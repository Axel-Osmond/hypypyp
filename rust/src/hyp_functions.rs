use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyOverflowError, PyValueError};
use pyo3::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HypergraphError {
    PairNodeOutOfBounds {
        tie: usize,
        node: usize,
        nodes: usize,
    },
    PairLinkOutOfBounds {
        tie: usize,
        link: usize,
        links: usize,
    },
    MapLength {
        map: &'static str,
        expected: usize,
        actual: usize,
    },
    MapValueOutOfBounds {
        map: &'static str,
        index: usize,
        value: usize,
        upper_bound: usize,
    },
    IncidenceNotPreserved {
        tie: usize,
    },
    ArithmeticOverflow {
        context: &'static str,
    },
    IndexOutOfBounds,
}

impl fmt::Display for HypergraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PairNodeOutOfBounds { tie, node, nodes } => write!(
                formatter,
                "pairs[{tie}] contains node index {node}, outside range({nodes})"
            ),
            Self::PairLinkOutOfBounds { tie, link, links } => write!(
                formatter,
                "pairs[{tie}] contains link index {link}, outside range({links})"
            ),
            Self::MapLength {
                map,
                expected,
                actual,
            } => write!(
                formatter,
                "{map} has length {actual}, but its domain has cardinality {expected}"
            ),
            Self::MapValueOutOfBounds {
                map,
                index,
                value,
                upper_bound,
            } => write!(
                formatter,
                "{map}[{index}] is {value}, outside range({upper_bound})"
            ),
            Self::IncidenceNotPreserved { tie } => write!(
                formatter,
                "the morphism does not preserve the node or link of source tie {tie}"
            ),
            Self::ArithmeticOverflow { context } => {
                write!(formatter, "usize overflow while computing {context}")
            }
            Self::IndexOutOfBounds => {
                write!(formatter, "morphism index is outside the hom-set")
            }
        }
    }
}

impl Error for HypergraphError {}

/// Pure Rust representation of a finite hypergraph.
///
/// A tie is represented by its position in `pairs`; its value is the pair of
/// ranks `(node, link)` incident to that tie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypergraphRustic {
    nodes: usize,
    links: usize,
    pairs: Vec<(usize, usize)>,
}

impl HypergraphRustic {
    pub fn new(
        nodes: usize,
        links: usize,
        pairs: Vec<(usize, usize)>,
    ) -> Result<Self, HypergraphError> {
        for (tie, &(node, link)) in pairs.iter().enumerate() {
            if node >= nodes {
                return Err(HypergraphError::PairNodeOutOfBounds { tie, node, nodes });
            }
            if link >= links {
                return Err(HypergraphError::PairLinkOutOfBounds { tie, link, links });
            }
        }

        Ok(Self {
            nodes,
            links,
            pairs,
        })
    }

    pub fn nodes(&self) -> usize {
        self.nodes
    }

    pub fn links(&self) -> usize {
        self.links
    }

    pub fn ties(&self) -> usize {
        self.pairs.len()
    }

    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    pub fn incidences(&self) -> Vec<(usize, usize)> {
        let mut incidences = self.pairs.clone();
        incidences.sort_unstable();
        incidences.dedup();
        incidences
    }

    pub fn test_simple(&self) -> bool {
        self.incidences().len() == self.pairs.len()
    }

    pub fn occurrences_ties(&self, node_index: usize) -> Vec<usize> {
        self.pairs
            .iter()
            .enumerate()
            .filter_map(|(tie, (node, _))| (*node == node_index).then_some(tie))
            .collect()
    }

    pub fn occurrences_links(&self, node_index: usize) -> Vec<usize> {
        self.incidences()
            .into_iter()
            .filter_map(|(node, link)| (node == node_index).then_some(link))
            .collect()
    }

    pub fn support_ties(&self, link_index: usize) -> Vec<usize> {
        self.pairs
            .iter()
            .enumerate()
            .filter_map(|(tie, (_, link))| (*link == link_index).then_some(tie))
            .collect()
    }

    pub fn support_nodes(&self, link_index: usize) -> Vec<usize> {
        self.incidences()
            .into_iter()
            .filter_map(|(node, link)| (link == link_index).then_some(node))
            .collect()
    }

    pub fn valence_ties(&self, node_index: usize, link_index: usize) -> Vec<usize> {
        self.pairs
            .iter()
            .enumerate()
            .filter_map(|(tie, (node, link))| {
                (*node == node_index && *link == link_index).then_some(tie)
            })
            .collect()
    }

    pub fn valence(&self, node_index: usize, link_index: usize) -> usize {
        self.pairs
            .iter()
            .filter(|(node, link)| *node == node_index && *link == link_index)
            .count()
    }

    pub fn cooccurrences(&self, nodes: &[usize]) -> Vec<usize> {
        (0..self.links)
            .filter(|&link| {
                let support = self.support_nodes(link);
                nodes.iter().all(|node| support.contains(node))
            })
            .collect()
    }

    pub fn loops(&self) -> Vec<(usize, usize, usize)> {
        (0..self.links)
            .filter_map(|link| {
                let support = self.support_nodes(link);
                (support.len() == 1).then(|| (support[0], link, self.valence(support[0], link)))
            })
            .collect()
    }

    pub fn valence_matrix(&self) -> Vec<Vec<usize>> {
        let mut valences = vec![vec![0; self.links]; self.nodes];
        for &(node, link) in &self.pairs {
            valences[node][link] += 1;
        }
        valences
    }
}

/// Pure Rust representation of a hypergraph morphism.
///
/// Source and target are reference-counted so cloning a morphism is cheap and
/// preserves sharing of the underlying hypergraphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorphismRustic {
    source: Arc<HypergraphRustic>,
    target: Arc<HypergraphRustic>,
    node_map: Vec<usize>,
    tie_map: Vec<usize>,
    link_map: Vec<usize>,
}

impl MorphismRustic {
    pub fn new(
        source: Arc<HypergraphRustic>,
        target: Arc<HypergraphRustic>,
        node_map: Vec<usize>,
        tie_map: Vec<usize>,
        link_map: Vec<usize>,
    ) -> Result<Self, HypergraphError> {
        validate_map("node_map", &node_map, source.nodes(), target.nodes())?;
        validate_map("tie_map", &tie_map, source.ties(), target.ties())?;
        validate_map("link_map", &link_map, source.links(), target.links())?;

        let morphism = Self {
            source,
            target,
            node_map,
            tie_map,
            link_map,
        };

        if let Some(tie) = morphism.first_non_preserved_tie() {
            return Err(HypergraphError::IncidenceNotPreserved { tie });
        }

        Ok(morphism)
    }

    pub fn source(&self) -> &Arc<HypergraphRustic> {
        &self.source
    }

    pub fn target(&self) -> &Arc<HypergraphRustic> {
        &self.target
    }

    pub fn node_map(&self) -> &[usize] {
        &self.node_map
    }

    pub fn tie_map(&self) -> &[usize] {
        &self.tie_map
    }

    pub fn link_map(&self) -> &[usize] {
        &self.link_map
    }

    pub fn test_morphism(&self) -> bool {
        self.first_non_preserved_tie().is_none()
    }

    pub fn test_mono(&self) -> bool {
        is_injective(&self.node_map) && is_injective(&self.tie_map) && is_injective(&self.link_map)
    }

    pub fn test_epi(&self) -> bool {
        is_surjective(&self.node_map, self.target.nodes())
            && is_surjective(&self.tie_map, self.target.ties())
            && is_surjective(&self.link_map, self.target.links())
    }

    pub fn test_iso(&self) -> bool {
        self.test_mono() && self.test_epi()
    }

    fn first_non_preserved_tie(&self) -> Option<usize> {
        self.source
            .pairs()
            .iter()
            .enumerate()
            .find_map(|(tie, &(source_node, source_link))| {
                let target_tie = self.tie_map[tie];
                let (target_node, target_link) = self.target.pairs()[target_tie];
                (self.node_map[source_node] != target_node
                    || self.link_map[source_link] != target_link)
                    .then_some(tie)
            })
    }
}

fn validate_map(
    name: &'static str,
    map: &[usize],
    expected_len: usize,
    upper_bound: usize,
) -> Result<(), HypergraphError> {
    if map.len() != expected_len {
        return Err(HypergraphError::MapLength {
            map: name,
            expected: expected_len,
            actual: map.len(),
        });
    }

    if let Some((index, &value)) = map
        .iter()
        .enumerate()
        .find(|(_, value)| **value >= upper_bound)
    {
        return Err(HypergraphError::MapValueOutOfBounds {
            map: name,
            index,
            value,
            upper_bound,
        });
    }

    Ok(())
}

fn is_injective(map: &[usize]) -> bool {
    let mut values = map.to_vec();
    values.sort_unstable();
    values.dedup();
    values.len() == map.len()
}

fn is_surjective(map: &[usize], target_cardinality: usize) -> bool {
    if target_cardinality == 0 {
        return map.is_empty();
    }

    let mut seen = vec![false; target_cardinality];
    for &value in map {
        seen[value] = true;
    }
    seen.into_iter().all(|value| value)
}

fn python_hypergraph_error(error: HypergraphError) -> PyErr {
    match error {
        HypergraphError::ArithmeticOverflow { .. } => PyOverflowError::new_err(error.to_string()),
        HypergraphError::IndexOutOfBounds => PyIndexError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

// Auxiliary functions for counting morphisms and constructing homsets

/// Count morphisms by explicitly indexing node maps and summing link images.
pub fn cardinality_indexed_by_nodes(
    source: &HypergraphRustic,
    target: &HypergraphRustic,
) -> Result<usize, HypergraphError> {
    let source_valences = source.valence_matrix();
    let target_valences = target.valence_matrix();
    cardinality_indexed_by_nodes_from_valences(
        &source_valences,
        &target_valences,
        source.nodes(),
        target.nodes(),
        source.links(),
        target.links(),
    )
}

/// Count morphisms by explicitly indexing link maps and summing node images.
pub fn cardinality_indexed_by_links(
    source: &HypergraphRustic,
    target: &HypergraphRustic,
) -> Result<usize, HypergraphError> {
    let source_valences = source.valence_matrix();
    let target_valences = target.valence_matrix();
    cardinality_indexed_by_links_from_valences(
        &source_valences,
        &target_valences,
        source.nodes(),
        target.nodes(),
        source.links(),
        target.links(),
    )
}

/// Count all morphisms and select the cheaper of the node- and link-indexed
/// versions using the same estimate as the Python prototype.
pub fn homgraph_cardinality_fast(
    source: &HypergraphRustic,
    target: &HypergraphRustic,
) -> Result<usize, HypergraphError> {
    let source_valences = source.valence_matrix();
    let target_valences = target.valence_matrix();
    let support_size = source_valences
        .iter()
        .flatten()
        .filter(|&&multiplicity| multiplicity != 0)
        .count();

    let node_function_count = saturating_power(target.nodes(), source.nodes());
    let link_function_count = saturating_power(target.links(), source.links());
    let node_cost = node_function_count
        .saturating_mul(target.links())
        .saturating_mul(support_size.saturating_add(source.links()));
    let link_cost = link_function_count
        .saturating_mul(target.nodes())
        .saturating_mul(support_size.saturating_add(source.nodes()));

    if node_cost <= link_cost {
        cardinality_indexed_by_nodes_from_valences(
            &source_valences,
            &target_valences,
            source.nodes(),
            target.nodes(),
            source.links(),
            target.links(),
        )
    } else {
        cardinality_indexed_by_links_from_valences(
            &source_valences,
            &target_valences,
            source.nodes(),
            target.nodes(),
            source.links(),
            target.links(),
        )
    }
}

fn cardinality_indexed_by_nodes_from_valences(
    source_valences: &[Vec<usize>],
    target_valences: &[Vec<usize>],
    source_nodes: usize,
    target_nodes: usize,
    source_links: usize,
    target_links: usize,
) -> Result<usize, HypergraphError> {
    let incidences_by_link: Vec<Vec<(usize, usize)>> = (0..source_links)
        .map(|link| {
            (0..source_nodes)
                .filter_map(|node| {
                    let multiplicity = source_valences[node][link];
                    (multiplicity != 0).then_some((node, multiplicity))
                })
                .collect()
        })
        .collect();
    let function_count = checked_power(target_nodes, source_nodes, "the number of node maps")?;
    let mut total = 0usize;

    for code in 0..function_count {
        let node_map = decode_function_values(source_nodes, target_nodes, code);
        let mut number_for_node_map = 1usize;

        for incidences in &incidences_by_link {
            let mut image_sum = 0usize;

            for target_link in 0..target_links {
                let mut term = 1usize;

                for &(source_node, multiplicity) in incidences {
                    let target_multiplicity = target_valences[node_map[source_node]][target_link];
                    if target_multiplicity == 0 {
                        term = 0;
                        break;
                    }
                    let choices =
                        checked_power(target_multiplicity, multiplicity, "tie-map choices")?;
                    term = checked_multiply(term, choices, "the homgraph cardinality")?;
                }

                image_sum = checked_add(image_sum, term, "the homgraph cardinality")?;
            }

            number_for_node_map =
                checked_multiply(number_for_node_map, image_sum, "the homgraph cardinality")?;
            if number_for_node_map == 0 {
                break;
            }
        }

        total = checked_add(total, number_for_node_map, "the homgraph cardinality")?;
    }

    Ok(total)
}

fn cardinality_indexed_by_links_from_valences(
    source_valences: &[Vec<usize>],
    target_valences: &[Vec<usize>],
    source_nodes: usize,
    target_nodes: usize,
    source_links: usize,
    target_links: usize,
) -> Result<usize, HypergraphError> {
    let incidences_by_node: Vec<Vec<(usize, usize)>> = (0..source_nodes)
        .map(|node| {
            (0..source_links)
                .filter_map(|link| {
                    let multiplicity = source_valences[node][link];
                    (multiplicity != 0).then_some((link, multiplicity))
                })
                .collect()
        })
        .collect();
    let function_count = checked_power(target_links, source_links, "the number of link maps")?;
    let mut total = 0usize;

    for code in 0..function_count {
        let link_map = decode_function_values(source_links, target_links, code);
        let mut number_for_link_map = 1usize;

        for incidences in &incidences_by_node {
            let mut image_sum = 0usize;

            for target_node in 0..target_nodes {
                let mut term = 1usize;

                for &(source_link, multiplicity) in incidences {
                    let target_multiplicity = target_valences[target_node][link_map[source_link]];
                    if target_multiplicity == 0 {
                        term = 0;
                        break;
                    }
                    let choices =
                        checked_power(target_multiplicity, multiplicity, "tie-map choices")?;
                    term = checked_multiply(term, choices, "the homgraph cardinality")?;
                }

                image_sum = checked_add(image_sum, term, "the homgraph cardinality")?;
            }

            number_for_link_map =
                checked_multiply(number_for_link_map, image_sum, "the homgraph cardinality")?;
            if number_for_link_map == 0 {
                break;
            }
        }

        total = checked_add(total, number_for_link_map, "the homgraph cardinality")?;
    }

    Ok(total)
}

fn decode_function_values(card_domain: usize, card_codomain: usize, mut code: usize) -> Vec<usize> {
    debug_assert!(card_codomain != 0 || card_domain == 0);
    let mut images = vec![0; card_domain];
    for image in &mut images {
        *image = code % card_codomain;
        code /= card_codomain;
    }
    images
}

fn checked_power(
    mut base: usize,
    mut exponent: usize,
    context: &'static str,
) -> Result<usize, HypergraphError> {
    let mut result = 1usize;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = checked_multiply(result, base, context)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = checked_multiply(base, base, context)?;
        }
    }
    Ok(result)
}

fn saturating_power(mut base: usize, mut exponent: usize) -> usize {
    let mut result = 1usize;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result.saturating_mul(base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base.saturating_mul(base);
        }
    }
    result
}

fn checked_multiply(
    left: usize,
    right: usize,
    context: &'static str,
) -> Result<usize, HypergraphError> {
    left.checked_mul(right)
        .ok_or(HypergraphError::ArithmeticOverflow { context })
}

fn checked_add(left: usize, right: usize, context: &'static str) -> Result<usize, HypergraphError> {
    left.checked_add(right)
        .ok_or(HypergraphError::ArithmeticOverflow { context })
}

/// Decode the `index`-th morphism, using the node-indexed block decomposition.
///
/// Node maps are visited in little-endian lexicographic order.  Inside the
/// block of a node map `f`, source link `l` has radix
///
/// `C_l = sum_j product_s val_target(f(s), j) ^ val_source(s, l)`.
///
/// Source links, source nodes, and ties inside a valence fiber are all decoded
/// least-significant coordinate first.  This is the same convention as
/// `seth_functions::decode_function`.
pub fn decode_hypergraph_morphism_by_nodes(
    source: Arc<HypergraphRustic>,
    target: Arc<HypergraphRustic>,
    mut index: BigUint,
) -> Result<MorphismRustic, HypergraphError> {
    let number_of_node_maps = big_power(target.nodes(), source.nodes());
    let mut node_map_code = BigUint::ZERO;
    let mut selected = None;

    while node_map_code < number_of_node_maps {
        let node_map = decode_big_function(source.nodes(), target.nodes(), node_map_code.clone());
        let (link_radices, link_image_blocks) =
            block_sizes_for_node_map(&source, &target, &node_map);
        let node_map_block_size: BigUint = link_radices.iter().product();

        if index < node_map_block_size {
            selected = Some((node_map, link_radices, link_image_blocks));
            break;
        }

        index -= node_map_block_size;
        node_map_code += 1u8;
    }

    let (node_map, link_radices, link_image_blocks) =
        selected.ok_or(HypergraphError::IndexOutOfBounds)?;
    let mut link_map = Vec::with_capacity(source.links());
    let mut local_tie_maps = vec![vec![Vec::new(); source.links()]; source.nodes()];

    for source_link in 0..source.links() {
        let link_radix = &link_radices[source_link];
        let mut link_code = &index % link_radix;
        index /= link_radix;

        let target_link = link_image_blocks[source_link]
            .iter()
            .position(|block_size| {
                if link_code < *block_size {
                    true
                } else {
                    link_code -= block_size;
                    false
                }
            })
            .expect("a code below C_l belongs to one of its additive blocks");
        link_map.push(target_link);

        for source_node in 0..source.nodes() {
            let source_valence = source.valence(source_node, source_link);
            let target_valence = target.valence(node_map[source_node], target_link);
            let local_radix = big_power(target_valence, source_valence);
            let local_code = &link_code % &local_radix;
            link_code /= local_radix;
            local_tie_maps[source_node][source_link] =
                decode_big_function(source_valence, target_valence, local_code);
        }
    }

    debug_assert!(index.is_zero());
    let mut positions_in_source_fibers = vec![vec![0usize; source.links()]; source.nodes()];
    let target_fibers: Vec<Vec<Vec<usize>>> = (0..target.nodes())
        .map(|target_node| {
            (0..target.links())
                .map(|target_link| target.valence_ties(target_node, target_link))
                .collect()
        })
        .collect();
    let mut tie_map = Vec::with_capacity(source.ties());

    for &(source_node, source_link) in source.pairs() {
        let fiber_position = positions_in_source_fibers[source_node][source_link];
        positions_in_source_fibers[source_node][source_link] += 1;
        let target_node = node_map[source_node];
        let target_link = link_map[source_link];
        let target_fiber_position = local_tie_maps[source_node][source_link][fiber_position];
        tie_map.push(target_fibers[target_node][target_link][target_fiber_position]);
    }

    MorphismRustic::new(source, target, node_map, tie_map, link_map)
}

fn block_sizes_for_node_map(
    source: &HypergraphRustic,
    target: &HypergraphRustic,
    node_map: &[usize],
) -> (Vec<BigUint>, Vec<Vec<BigUint>>) {
    let link_image_blocks: Vec<Vec<BigUint>> = (0..source.links())
        .map(|source_link| {
            (0..target.links())
                .map(|target_link| {
                    (0..source.nodes())
                        .map(|source_node| {
                            big_power(
                                target.valence(node_map[source_node], target_link),
                                source.valence(source_node, source_link),
                            )
                        })
                        .product()
                })
                .collect()
        })
        .collect();
    let link_radices = link_image_blocks
        .iter()
        .map(|blocks| blocks.iter().sum())
        .collect();
    (link_radices, link_image_blocks)
}

fn big_power(base: usize, mut exponent: usize) -> BigUint {
    let mut result = BigUint::from(1u8);
    let mut big_base = BigUint::from(base);
    while exponent != 0 {
        if exponent & 1 == 1 {
            result *= &big_base;
        }
        exponent >>= 1;
        if exponent != 0 {
            big_base = &big_base * &big_base;
        }
    }
    result
}

fn decode_big_function(card_domain: usize, card_codomain: usize, mut code: BigUint) -> Vec<usize> {
    debug_assert!(card_codomain != 0 || card_domain == 0);
    let radix = BigUint::from(card_codomain);
    let mut images = Vec::with_capacity(card_domain);
    for _ in 0..card_domain {
        let image = (&code % &radix)
            .to_usize()
            .expect("a remainder modulo usize always fits in usize");
        images.push(image);
        code /= &radix;
    }
    debug_assert!(code.is_zero());
    images
}

/// Thin Python wrapper around [`HypergraphRustic`].
#[pyclass(frozen, module = "rustic.rustic")]
#[derive(Debug)]
pub struct HypergraphPy {
    inner: Arc<HypergraphRustic>,
}

impl HypergraphPy {
    fn from_inner(inner: Arc<HypergraphRustic>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl HypergraphPy {
    #[new]
    fn new(nodes: usize, links: usize, pairs: Vec<(usize, usize)>) -> PyResult<Self> {
        HypergraphRustic::new(nodes, links, pairs)
            .map(Arc::new)
            .map(Self::from_inner)
            .map_err(python_hypergraph_error)
    }

    #[getter]
    fn nodes(&self) -> usize {
        self.inner.nodes()
    }

    #[getter]
    fn links(&self) -> usize {
        self.inner.links()
    }

    #[getter]
    fn ties(&self) -> usize {
        self.inner.ties()
    }

    #[getter]
    fn pairs(&self) -> Vec<(usize, usize)> {
        self.inner.pairs().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "HypergraphPy(nodes={}, links={}, pairs={:?})",
            self.inner.nodes(),
            self.inner.links(),
            self.inner.pairs()
        )
    }

    fn incidences(&self) -> Vec<(usize, usize)> {
        self.inner.incidences()
    }

    fn test_simple(&self) -> bool {
        self.inner.test_simple()
    }

    fn occurrences_ties(&self, node_index: usize) -> Vec<usize> {
        self.inner.occurrences_ties(node_index)
    }

    fn occurrences_links(&self, node_index: usize) -> Vec<usize> {
        self.inner.occurrences_links(node_index)
    }

    fn support_ties(&self, link_index: usize) -> Vec<usize> {
        self.inner.support_ties(link_index)
    }

    fn support_nodes(&self, link_index: usize) -> Vec<usize> {
        self.inner.support_nodes(link_index)
    }

    fn valence_ties(&self, node_index: usize, link_index: usize) -> Vec<usize> {
        self.inner.valence_ties(node_index, link_index)
    }

    fn valence(&self, node_index: usize, link_index: usize) -> usize {
        self.inner.valence(node_index, link_index)
    }

    fn cooccurrences(&self, nodes: Vec<usize>) -> Vec<usize> {
        self.inner.cooccurrences(&nodes)
    }

    fn loops(&self) -> Vec<(usize, usize, usize)> {
        self.inner.loops()
    }
}

/// Thin Python wrapper around [`MorphismRustic`].
#[pyclass(frozen, module = "rustic.rustic")]
#[derive(Debug)]
pub struct MorphismPy {
    inner: MorphismRustic,
}

#[pymethods]
impl MorphismPy {
    #[new]
    fn new(
        source: PyRef<'_, HypergraphPy>,
        target: PyRef<'_, HypergraphPy>,
        node_map: Vec<usize>,
        tie_map: Vec<usize>,
        link_map: Vec<usize>,
    ) -> PyResult<Self> {
        MorphismRustic::new(
            Arc::clone(&source.inner),
            Arc::clone(&target.inner),
            node_map,
            tie_map,
            link_map,
        )
        .map(|inner| Self { inner })
        .map_err(python_hypergraph_error)
    }

    #[getter]
    fn source(&self) -> HypergraphPy {
        HypergraphPy::from_inner(Arc::clone(self.inner.source()))
    }

    #[getter]
    fn target(&self) -> HypergraphPy {
        HypergraphPy::from_inner(Arc::clone(self.inner.target()))
    }

    #[getter]
    fn node_map(&self) -> Vec<usize> {
        self.inner.node_map().to_vec()
    }

    #[getter]
    fn tie_map(&self) -> Vec<usize> {
        self.inner.tie_map().to_vec()
    }

    #[getter]
    fn link_map(&self) -> Vec<usize> {
        self.inner.link_map().to_vec()
    }

    #[getter]
    fn mapping(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        (self.node_map(), self.tie_map(), self.link_map())
    }

    fn __repr__(&self) -> String {
        format!(
            "MorphismPy(node_map={:?}, tie_map={:?}, link_map={:?})",
            self.inner.node_map(),
            self.inner.tie_map(),
            self.inner.link_map()
        )
    }

    fn test_morphism(&self) -> bool {
        self.inner.test_morphism()
    }

    fn test_mono(&self) -> bool {
        self.inner.test_mono()
    }

    fn test_epi(&self) -> bool {
        self.inner.test_epi()
    }

    fn test_iso(&self) -> bool {
        self.inner.test_iso()
    }
}

#[pyfunction(name = "_cardinality_indexed_by_nodes")]
fn cardinality_indexed_by_nodes_py(
    source: PyRef<'_, HypergraphPy>,
    target: PyRef<'_, HypergraphPy>,
) -> PyResult<usize> {
    cardinality_indexed_by_nodes(&source.inner, &target.inner).map_err(python_hypergraph_error)
}

#[pyfunction(name = "_cardinality_indexed_by_links")]
fn cardinality_indexed_by_links_py(
    source: PyRef<'_, HypergraphPy>,
    target: PyRef<'_, HypergraphPy>,
) -> PyResult<usize> {
    cardinality_indexed_by_links(&source.inner, &target.inner).map_err(python_hypergraph_error)
}

#[pyfunction(name = "homgraph_cardinality_fast")]
fn homgraph_cardinality_fast_py(
    source: PyRef<'_, HypergraphPy>,
    target: PyRef<'_, HypergraphPy>,
) -> PyResult<usize> {
    homgraph_cardinality_fast(&source.inner, &target.inner).map_err(python_hypergraph_error)
}

#[pyfunction(name = "decode_hypergraph_morphism_by_nodes")]
fn decode_hypergraph_morphism_by_nodes_py(
    source: PyRef<'_, HypergraphPy>,
    target: PyRef<'_, HypergraphPy>,
    index: BigUint,
) -> PyResult<MorphismPy> {
    decode_hypergraph_morphism_by_nodes(Arc::clone(&source.inner), Arc::clone(&target.inner), index)
        .map(|inner| MorphismPy { inner })
        .map_err(python_hypergraph_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn brute_force_cardinality(source: &HypergraphRustic, target: &HypergraphRustic) -> usize {
        let node_function_count = target.nodes().pow(source.nodes() as u32);
        let link_function_count = target.links().pow(source.links() as u32);
        let mut total = 0usize;

        for node_code in 0..node_function_count {
            let node_map = decode_function_values(source.nodes(), target.nodes(), node_code);
            for link_code in 0..link_function_count {
                let link_map = decode_function_values(source.links(), target.links(), link_code);
                let number_of_tie_maps =
                    source
                        .pairs()
                        .iter()
                        .fold(1usize, |number, &(source_node, source_link)| {
                            number * target.valence(node_map[source_node], link_map[source_link])
                        });
                total += number_of_tie_maps;
            }
        }

        total
    }

    fn brute_force_morphisms(
        source: &HypergraphRustic,
        target: &HypergraphRustic,
    ) -> HashSet<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let mut morphisms = HashSet::new();
        let node_function_count = target.nodes().pow(source.nodes() as u32);
        let link_function_count = target.links().pow(source.links() as u32);

        for node_code in 0..node_function_count {
            let node_map = decode_function_values(source.nodes(), target.nodes(), node_code);
            for link_code in 0..link_function_count {
                let link_map = decode_function_values(source.links(), target.links(), link_code);
                let target_fibers: Vec<Vec<usize>> = source
                    .pairs()
                    .iter()
                    .map(|&(source_node, source_link)| {
                        target.valence_ties(node_map[source_node], link_map[source_link])
                    })
                    .collect();
                let tie_map_count = target_fibers.iter().map(Vec::len).product();

                for mut tie_code in 0..tie_map_count {
                    let tie_map = target_fibers
                        .iter()
                        .map(|fiber| {
                            let image = fiber[tie_code % fiber.len()];
                            tie_code /= fiber.len();
                            image
                        })
                        .collect();
                    morphisms.insert((node_map.clone(), tie_map, link_map.clone()));
                }
            }
        }

        morphisms
    }

    fn all_hypergraphs_with_at_most_two_of_each_kind() -> Vec<HypergraphRustic> {
        let mut hypergraphs = Vec::new();
        for nodes in 0..=2usize {
            for links in 0..=2usize {
                let pair_count = nodes * links;
                if pair_count == 0 {
                    hypergraphs.push(HypergraphRustic::new(nodes, links, vec![]).unwrap());
                    continue;
                }

                for ties in 0..=2usize {
                    for mut code in 0..pair_count.pow(ties as u32) {
                        let pairs = (0..ties)
                            .map(|_| {
                                let pair = code % pair_count;
                                code /= pair_count;
                                (pair % nodes, pair / nodes)
                            })
                            .collect();
                        hypergraphs.push(HypergraphRustic::new(nodes, links, pairs).unwrap());
                    }
                }
            }
        }
        hypergraphs
    }

    #[test]
    fn hypergraph_preserves_ties_and_computes_supports() {
        let hypergraph = HypergraphRustic::new(2, 2, vec![(0, 0), (0, 0), (1, 0), (1, 1)]).unwrap();

        assert_eq!(hypergraph.ties(), 4);
        assert_eq!(hypergraph.incidences(), vec![(0, 0), (1, 0), (1, 1)]);
        assert!(!hypergraph.test_simple());
        assert_eq!(hypergraph.occurrences_ties(0), vec![0, 1]);
        assert_eq!(hypergraph.occurrences_links(0), vec![0]);
        assert_eq!(hypergraph.support_ties(0), vec![0, 1, 2]);
        assert_eq!(hypergraph.support_nodes(0), vec![0, 1]);
        assert_eq!(hypergraph.valence_ties(0, 0), vec![0, 1]);
        assert_eq!(hypergraph.valence(0, 0), 2);
        assert_eq!(hypergraph.cooccurrences(&[0, 1]), vec![0]);
        assert_eq!(hypergraph.loops(), vec![(1, 1, 1)]);
    }

    #[test]
    fn hypergraph_accepts_empty_data_and_rejects_bad_ranks() {
        assert!(HypergraphRustic::new(0, 0, vec![]).is_ok());
        assert!(HypergraphRustic::new(1, 1, vec![(1, 0)]).is_err());
        assert!(HypergraphRustic::new(1, 1, vec![(0, 1)]).is_err());
    }

    #[test]
    fn homgraph_cardinality_matches_exhaustive_counting() {
        let cases = vec![
            (
                HypergraphRustic::new(0, 0, vec![]).unwrap(),
                HypergraphRustic::new(0, 0, vec![]).unwrap(),
            ),
            (
                HypergraphRustic::new(1, 0, vec![]).unwrap(),
                HypergraphRustic::new(2, 0, vec![]).unwrap(),
            ),
            (
                HypergraphRustic::new(0, 1, vec![]).unwrap(),
                HypergraphRustic::new(0, 2, vec![]).unwrap(),
            ),
            (
                HypergraphRustic::new(1, 1, vec![(0, 0), (0, 0)]).unwrap(),
                HypergraphRustic::new(1, 1, vec![(0, 0), (0, 0), (0, 0)]).unwrap(),
            ),
            (
                HypergraphRustic::new(2, 2, vec![(0, 0), (0, 0), (1, 0), (1, 1)]).unwrap(),
                HypergraphRustic::new(2, 2, vec![(0, 0), (1, 0), (1, 1), (1, 1)]).unwrap(),
            ),
        ];

        for (source, target) in cases {
            let expected = brute_force_cardinality(&source, &target);
            assert_eq!(cardinality_indexed_by_nodes(&source, &target), Ok(expected));
            assert_eq!(cardinality_indexed_by_links(&source, &target), Ok(expected));
            assert_eq!(homgraph_cardinality_fast(&source, &target), Ok(expected));
        }
    }

    #[test]
    fn homgraph_cardinality_reports_overflow() {
        let source = HypergraphRustic::new(1, 1, vec![(0, 0); 100]).unwrap();
        let target = HypergraphRustic::new(1, 1, vec![(0, 0); 2]).unwrap();

        assert!(matches!(
            homgraph_cardinality_fast(&source, &target),
            Err(HypergraphError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn decoder_enumerates_every_small_morphism_exactly_once() {
        let hypergraphs = all_hypergraphs_with_at_most_two_of_each_kind();

        for source in &hypergraphs {
            for target in &hypergraphs {
                let expected = brute_force_morphisms(source, target);
                let source = Arc::new(source.clone());
                let target = Arc::new(target.clone());
                let mut decoded = HashSet::new();

                for index in 0..expected.len() {
                    let morphism = decode_hypergraph_morphism_by_nodes(
                        Arc::clone(&source),
                        Arc::clone(&target),
                        BigUint::from(index),
                    )
                    .unwrap();
                    assert!(Arc::ptr_eq(&source, morphism.source()));
                    assert!(Arc::ptr_eq(&target, morphism.target()));
                    decoded.insert((
                        morphism.node_map().to_vec(),
                        morphism.tie_map().to_vec(),
                        morphism.link_map().to_vec(),
                    ));
                }

                assert_eq!(decoded, expected);
                assert!(matches!(
                    decode_hypergraph_morphism_by_nodes(
                        source,
                        target,
                        BigUint::from(expected.len()),
                    ),
                    Err(HypergraphError::IndexOutOfBounds)
                ));
            }
        }
    }

    #[test]
    fn decoder_uses_little_endian_tie_coordinates() {
        let source = Arc::new(HypergraphRustic::new(1, 1, vec![(0, 0), (0, 0)]).unwrap());
        let target = Arc::new(HypergraphRustic::new(1, 1, vec![(0, 0), (0, 0)]).unwrap());
        let expected_tie_maps = [vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]];

        for (index, expected_tie_map) in expected_tie_maps.into_iter().enumerate() {
            let morphism = decode_hypergraph_morphism_by_nodes(
                Arc::clone(&source),
                Arc::clone(&target),
                BigUint::from(index),
            )
            .unwrap();
            assert_eq!(morphism.tie_map(), expected_tie_map);
        }
    }

    #[test]
    fn decoder_orders_node_blocks_then_mixed_radix_link_coordinates() {
        let source = Arc::new(HypergraphRustic::new(1, 2, vec![]).unwrap());
        let target = Arc::new(HypergraphRustic::new(2, 2, vec![]).unwrap());
        let expected = [
            (vec![0], vec![0, 0]),
            (vec![0], vec![1, 0]),
            (vec![0], vec![0, 1]),
            (vec![0], vec![1, 1]),
            (vec![1], vec![0, 0]),
            (vec![1], vec![1, 0]),
            (vec![1], vec![0, 1]),
            (vec![1], vec![1, 1]),
        ];

        for (index, (expected_node_map, expected_link_map)) in expected.into_iter().enumerate() {
            let morphism = decode_hypergraph_morphism_by_nodes(
                Arc::clone(&source),
                Arc::clone(&target),
                BigUint::from(index),
            )
            .unwrap();
            assert_eq!(morphism.node_map(), expected_node_map);
            assert_eq!(morphism.link_map(), expected_link_map);
        }
    }

    #[test]
    fn decoder_accepts_indices_larger_than_usize() {
        let source = Arc::new(HypergraphRustic::new(1, 1, vec![(0, 0); 100]).unwrap());
        let target = Arc::new(HypergraphRustic::new(1, 1, vec![(0, 0); 2]).unwrap());
        let index = BigUint::from(1u8) << 99usize;
        let morphism = decode_hypergraph_morphism_by_nodes(source, target, index).unwrap();

        assert!(morphism.tie_map()[..99].iter().all(|&image| image == 0));
        assert_eq!(morphism.tie_map()[99], 1);
    }

    #[test]
    fn morphism_validates_and_shares_its_hypergraphs() {
        let hypergraph =
            Arc::new(HypergraphRustic::new(2, 2, vec![(0, 0), (1, 0), (1, 1)]).unwrap());
        let morphism = MorphismRustic::new(
            Arc::clone(&hypergraph),
            Arc::clone(&hypergraph),
            vec![0, 1],
            vec![0, 1, 2],
            vec![0, 1],
        )
        .unwrap();

        assert!(Arc::ptr_eq(&hypergraph, morphism.source()));
        assert!(Arc::ptr_eq(morphism.source(), morphism.target()));
        assert!(morphism.test_morphism());
        assert!(morphism.test_iso());
    }

    #[test]
    fn morphism_rejects_bad_maps_and_non_commuting_data() {
        let source = Arc::new(HypergraphRustic::new(2, 1, vec![(0, 0), (1, 0)]).unwrap());
        let target = Arc::new(HypergraphRustic::new(2, 1, vec![(0, 0), (1, 0)]).unwrap());

        assert!(
            MorphismRustic::new(
                Arc::clone(&source),
                Arc::clone(&target),
                vec![0],
                vec![0, 1],
                vec![0],
            )
            .is_err()
        );
        assert!(
            MorphismRustic::new(
                Arc::clone(&source),
                Arc::clone(&target),
                vec![0, 2],
                vec![0, 1],
                vec![0],
            )
            .is_err()
        );
        assert!(MorphismRustic::new(source, target, vec![1, 0], vec![0, 1], vec![0],).is_err());
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<HypergraphPy>()?;
    module.add_class::<MorphismPy>()?;
    module.add_function(wrap_pyfunction!(cardinality_indexed_by_nodes_py, module)?)?;
    module.add_function(wrap_pyfunction!(cardinality_indexed_by_links_py, module)?)?;
    module.add_function(wrap_pyfunction!(homgraph_cardinality_fast_py, module)?)?;
    module.add_function(wrap_pyfunction!(decode_hypergraph_morphism_by_nodes_py, module)?)?;
    Ok(())
}
