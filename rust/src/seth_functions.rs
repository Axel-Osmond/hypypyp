use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

pub fn factorial(n: usize) -> usize {
    (1..=n).product()
}

pub fn stirling_2(n: usize, m: usize) -> usize {
    if m == 0 {
        if n == 0 {
            return 1;
        } else {
            return 0;
        }
    }
    if m > n {
        return 0;
    }
    if m == 1 || m == n {
        return 1;
    }
    stirling_2(n - 1, m - 1) + m * stirling_2(n - 1, m)
}

#[pyfunction]
pub fn encode_function(card_codomain: usize, values: Vec<usize>) -> PyResult<usize> {
    let mut code: usize = 0;
    let mut factor: usize = 1;
    for (position, &value) in values.iter().enumerate() {
        if value >= card_codomain {
            return Err(PyValueError::new_err("invalid codomain index"));
        }
        let term = value
            .checked_mul(factor)
            .ok_or_else(|| PyOverflowError::new_err("function code exceeds usize"))?;
        code = code
            .checked_add(term)
            .ok_or_else(|| PyOverflowError::new_err("function code exceeds usize"))?;
        if position + 1 < values.len() {
            factor = factor
                .checked_mul(card_codomain)
                .ok_or_else(|| PyOverflowError::new_err("function code exceeds usize"))?;
        }
    }
    Ok(code)
}

#[pyfunction]
pub fn decode_function(
    card_domain: usize,
    card_codomain: usize,
    mut value: usize,
) -> PyResult<Vec<usize>> {
    if card_codomain == 0 {
        if card_domain == 0 && value == 0 {
            return Ok(Vec::new());
        }
        return Err(PyValueError::new_err("codomain is empty"));
    }

    let mut result = vec![0; card_domain];

    for image in &mut result {
        *image = value % card_codomain;
        value /= card_codomain;
    }

    if value != 0 {
        return Err(PyOverflowError::new_err("index exceeds homset cardinality"));
    }

    Ok(result)
}

#[pyfunction]
pub fn injection_count(card_domain: usize, card_codomain: usize) -> usize {
    if card_domain > card_codomain {
        return 0;
    }

    (0..card_domain).map(|i| card_codomain - i).product()
}

#[pyfunction]
pub fn count_injections(card_domain: usize, card_codomain: usize) -> usize {
    injection_count(card_domain, card_codomain)
}

#[pyfunction]
pub fn decode_injection(
    card_domain: usize,
    card_codomain: usize,
    mut value: usize,
) -> PyResult<Vec<usize>> {
    if card_domain > card_codomain {
        return Err(PyValueError::new_err("domain is larger than codomain"));
    }

    let number_of_injections = injection_count(card_domain, card_codomain);

    if value >= number_of_injections {
        return Err(PyValueError::new_err("index exceeds injection cardinality"));
    }

    let denominator = factorial(card_codomain - card_domain);
    let mut available: Vec<usize> = (0..card_codomain).collect();

    let mut result = Vec::with_capacity(card_domain);

    for i in 0..card_domain {
        let block_size = factorial(card_codomain - i - 1) / denominator;

        let q = value / block_size;
        value %= block_size;

        result.push(available.remove(q));
    }

    Ok(result)
}

#[pyfunction]
fn encode_injection(card_codomain: usize, images: Vec<usize>) -> PyResult<usize> {
    if images.len() > card_codomain {
        return Err(PyValueError::new_err("an injection requires n <= m"));
    }

    let mut available: Vec<usize> = (0..card_codomain).collect();

    let mut rank = 0usize;

    for (i, image) in images.into_iter().enumerate() {
        let q_i = available
            .iter()
            .position(|&value| value == image)
            .ok_or_else(|| PyValueError::new_err("image outside the codomain or repeated"))?;

        rank = rank
            .checked_mul(card_codomain - i)
            .and_then(|r| r.checked_add(q_i))
            .ok_or_else(|| PyOverflowError::new_err("injection rank exceeds usize"))?;

        available.remove(q_i);
    }

    Ok(rank)
}

#[pyfunction]
fn unrank_partition(n: usize, m: usize, rank: usize) -> PyResult<Vec<usize>> {
    let number_of_partitions = stirling_2(n, m);

    if rank >= number_of_partitions {
        return Err(PyValueError::new_err("rank exceeds partition cardinality"));
    }

    if n == 0 {
        return Ok(Vec::new());
    }

    let singleton_count = stirling_2(n - 1, m - 1);

    if rank < singleton_count {
        let mut labels = unrank_partition(n - 1, m - 1, rank)?;

        // Le nouvel élément forme le dernier bloc.
        labels.push(m - 1);
        Ok(labels)
    } else {
        let remaining_rank = rank - singleton_count;
        let previous_rank = remaining_rank / m;
        let block_index = remaining_rank % m;

        let mut labels = unrank_partition(n - 1, m, previous_rank)?;

        // Le nouvel élément rejoint ce bloc.
        labels.push(block_index);
        Ok(labels)
    }
}

fn rank_partition(labels: &[usize], m: usize) -> PyResult<usize> {
    let n = labels.len();

    if n == 0 {
        return if m == 0 {
            Ok(0)
        } else {
            Err(PyValueError::new_err(
                "an empty set has no non-empty partition",
            ))
        };
    }

    if m == 0 || m > n {
        return Err(PyValueError::new_err("invalid number of partition blocks"));
    }

    let last_block = labels[n - 1];
    if last_block >= m {
        return Err(PyValueError::new_err(
            "partition contains an invalid block index",
        ));
    }

    let prefix = &labels[..n - 1];
    let creates_last_block = last_block == m - 1 && !prefix.contains(&last_block);

    if creates_last_block {
        rank_partition(prefix, m - 1)
    } else {
        let previous_rank = rank_partition(prefix, m)?;
        let singleton_count = stirling_2(n - 1, m - 1);

        previous_rank
            .checked_mul(m)
            .and_then(|rank| rank.checked_add(last_block))
            .and_then(|rank| rank.checked_add(singleton_count))
            .ok_or_else(|| PyOverflowError::new_err("partition rank exceeds usize"))
    }
}

#[pyfunction]
fn unrank_permutation(n: usize, mut rank: usize) -> PyResult<Vec<usize>> {
    let total = factorial(n);

    if rank >= total {
        return Err(PyValueError::new_err("permutation rank out of range"));
    }

    let mut available: Vec<usize> = (0..n).collect();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let block_size = factorial(n - i - 1);
        let q = rank / block_size;
        rank %= block_size;

        result.push(available.remove(q));
    }

    Ok(result)
}

#[pyfunction]
pub fn count_surjections(card_domain: usize, card_codomain: usize) -> usize {
    if card_domain < card_codomain {
        return 0;
    }
    stirling_2(card_domain, card_codomain) * factorial(card_codomain)
}

#[pyfunction]
pub fn decode_surjection(
    card_domain: usize,
    card_codomain: usize,
    value: usize,
) -> PyResult<Vec<usize>> {
    let number_of_surjections = count_surjections(card_domain, card_codomain);

    if value >= number_of_surjections {
        return Err(PyValueError::new_err("rank exceeds surjection cardinality"));
    }
    let partition_rank = value / factorial(card_codomain);
    let permutation_rank = value % factorial(card_codomain);

    let partition = unrank_partition(card_domain, card_codomain, partition_rank)?;

    let permutation = unrank_permutation(card_codomain, permutation_rank)?;

    let table = partition
        .iter()
        .map(|&block_index| permutation[block_index])
        .collect();

    Ok(table)
}

#[pyfunction]
pub fn encode_surjection(
    card_domain: usize,
    card_codomain: usize,
    images: Vec<usize>,
) -> PyResult<usize> {
    if images.len() != card_domain {
        return Err(PyValueError::new_err(
            "a surjection must provide one image for every domain element",
        ));
    }

    if card_codomain == 0 {
        return if images.is_empty() {
            Ok(0)
        } else {
            Err(PyValueError::new_err(
                "a non-empty domain cannot map surjectively to an empty codomain",
            ))
        };
    }

    if images.len() < card_codomain {
        return Err(PyValueError::new_err("a surjection requires n >= m"));
    }

    let mut image_to_block = vec![usize::MAX; card_codomain];
    let mut partition = Vec::with_capacity(images.len());
    let mut permutation = Vec::with_capacity(card_codomain);

    for image in images {
        if image >= card_codomain {
            return Err(PyValueError::new_err("image outside the codomain"));
        }

        let block_index = if image_to_block[image] == usize::MAX {
            let new_block = permutation.len();
            image_to_block[image] = new_block;
            permutation.push(image);
            new_block
        } else {
            image_to_block[image]
        };

        partition.push(block_index);
    }

    if permutation.len() != card_codomain {
        return Err(PyValueError::new_err("the function is not surjective"));
    }

    let partition_rank = rank_partition(&partition, card_codomain)?;
    let mut available: Vec<usize> = (0..card_codomain).collect();
    let mut permutation_rank = 0usize;

    for (i, image) in permutation.into_iter().enumerate() {
        let position = available
            .iter()
            .position(|&value| value == image)
            .expect("the block images form a validated permutation");

        permutation_rank = permutation_rank
            .checked_mul(card_codomain - i)
            .and_then(|rank| rank.checked_add(position))
            .ok_or_else(|| PyOverflowError::new_err("permutation rank exceeds usize"))?;

        available.remove(position);
    }

    partition_rank
        .checked_mul(factorial(card_codomain))
        .and_then(|rank| rank.checked_add(permutation_rank))
        .ok_or_else(|| PyOverflowError::new_err("surjection rank exceeds usize"))
}

#[pyfunction]
pub fn bijection_number(card_domain: usize, card_codomain: usize) -> usize {
    if card_domain != card_codomain {
        return 0;
    }
    factorial(card_domain)
}

#[pyfunction]
pub fn decode_bijection(
    card_domain: usize,
    card_codomain: usize,
    value: usize,
) -> PyResult<Vec<usize>> {
    if card_domain != card_codomain {
        return Err(PyValueError::new_err(
            "domain and codomain must have the same cardinality",
        ));
    }

    let number_of_bijections = bijection_number(card_domain, card_codomain);

    if value >= number_of_bijections {
        return Err(PyValueError::new_err("rank exceeds bijection cardinality"));
    }

    unrank_permutation(card_domain, value)
}

#[pyfunction]
pub fn encode_bijection(
    card_domain: usize,
    card_codomain: usize,
    images: Vec<usize>,
) -> PyResult<usize> {
    if card_domain != card_codomain {
        return Err(PyValueError::new_err(
            "domain and codomain must have the same cardinality",
        ));
    }

    if images.len() != card_domain {
        return Err(PyValueError::new_err(
            "a bijection must provide one image for every domain element",
        ));
    }

    encode_injection(card_codomain, images)
}

#[pyfunction]
pub fn canonical_surjection_fibers(
    mut fibers: Vec<Vec<usize>>,
) -> PyResult<(Vec<Vec<usize>>, usize)> {
    if fibers.iter().any(Vec::is_empty) {
        return Err(PyValueError::new_err(
            "every fiber of a surjection must be non-empty",
        ));
    }

    let card_domain = fibers.iter().try_fold(0usize, |cardinality, fiber| {
        cardinality
            .checked_add(fiber.len())
            .ok_or_else(|| PyOverflowError::new_err("domain cardinality exceeds usize"))
    })?;
    let mut seen = vec![false; card_domain];

    for fiber in &mut fibers {
        fiber.sort_unstable();
        for &image in fiber.iter() {
            if image >= card_domain {
                return Err(PyValueError::new_err(
                    "fibers must contain domain ranks in range(n)",
                ));
            }
            if seen[image] {
                return Err(PyValueError::new_err(
                    "the fibers of a surjection must be pairwise disjoint",
                ));
            }
            seen[image] = true;
        }
    }

    Ok((fibers, card_domain))
}

pub fn number_of_sections_impl(fibers: &[Vec<usize>]) -> PyResult<usize> {
    fibers.iter().try_fold(1usize, |number, fiber| {
        number
            .checked_mul(fiber.len())
            .ok_or_else(|| PyOverflowError::new_err("number of sections exceeds usize"))
    })
}

#[pyfunction]
pub fn number_of_sections(fibers: Vec<Vec<usize>>) -> PyResult<usize> {
    let (fibers, _) = canonical_surjection_fibers(fibers)?;
    number_of_sections_impl(&fibers)
}

pub fn decode_section_impl(fibers: &[Vec<usize>], mut section_code: usize) -> PyResult<Vec<usize>> {
    let number = number_of_sections_impl(fibers)?;
    if section_code >= number {
        return Err(PyValueError::new_err(
            "section code outside range(number_of_sections)",
        ));
    }

    let mut section = Vec::with_capacity(fibers.len());
    for fiber in fibers {
        let choice = section_code % fiber.len();
        section_code /= fiber.len();
        section.push(fiber[choice]);
    }

    Ok(section)
}

#[pyfunction]
pub fn decode_section(fibers: Vec<Vec<usize>>, section_code: usize) -> PyResult<Vec<usize>> {
    let (fibers, _) = canonical_surjection_fibers(fibers)?;
    decode_section_impl(&fibers, section_code)
}

#[pyfunction]
pub fn encode_section(fibers: Vec<Vec<usize>>, section: Vec<usize>) -> PyResult<usize> {
    let (fibers, _) = canonical_surjection_fibers(fibers)?;
    if section.len() != fibers.len() {
        return Err(PyValueError::new_err(
            "a section must provide one image for every fiber",
        ));
    }

    let mut code = 0usize;
    let mut weight = 1usize;
    for (fiber, image) in fibers.iter().zip(section) {
        let choice = fiber
            .binary_search(&image)
            .map_err(|_| PyValueError::new_err("the function is not a section"))?;
        let term = choice
            .checked_mul(weight)
            .ok_or_else(|| PyOverflowError::new_err("section code exceeds usize"))?;
        code = code
            .checked_add(term)
            .ok_or_else(|| PyOverflowError::new_err("section code exceeds usize"))?;
        weight = weight
            .checked_mul(fiber.len())
            .ok_or_else(|| PyOverflowError::new_err("number of sections exceeds usize"))?;
    }

    Ok(code)
}

#[pyfunction]
pub fn section_code_to_injection_code(
    fibers: Vec<Vec<usize>>,
    section_code: usize,
) -> PyResult<usize> {
    let (fibers, card_domain) = canonical_surjection_fibers(fibers)?;
    let section = decode_section_impl(&fibers, section_code)?;
    encode_injection(card_domain, section)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_ranking_round_trip() {
        for n in 0..=7 {
            for m in 0..=n {
                for rank in 0..stirling_2(n, m) {
                    let labels = unrank_partition(n, m, rank).unwrap();
                    assert_eq!(rank_partition(&labels, m).unwrap(), rank);
                }
            }
        }
    }

    #[test]
    fn surjection_ranking_round_trip() {
        for n in 0..=6 {
            for m in 0..=n {
                for rank in 0..count_surjections(n, m) {
                    let images = decode_surjection(n, m, rank).unwrap();
                    assert_eq!(encode_surjection(n, m, images).unwrap(), rank);
                }
            }
        }
    }

    #[test]
    fn rejects_non_surjections() {
        assert!(encode_surjection(3, 3, vec![0, 1, 1]).is_err());
        assert!(encode_surjection(2, 2, vec![0, 2]).is_err());
        assert!(encode_surjection(1, 0, vec![0]).is_err());
        assert!(encode_surjection(3, 2, vec![0, 1]).is_err());
    }

    #[test]
    fn bijection_ranking_round_trip() {
        for n in 0..=8 {
            for rank in 0..bijection_number(n, n) {
                let images = decode_bijection(n, n, rank).unwrap();
                assert_eq!(encode_bijection(n, n, images).unwrap(), rank);
            }
        }
    }

    #[test]
    fn rejects_non_bijections() {
        assert!(encode_bijection(2, 3, vec![0, 1]).is_err());
        assert!(encode_bijection(3, 3, vec![0, 1]).is_err());
        assert!(encode_bijection(3, 3, vec![0, 1, 1]).is_err());
        assert!(encode_bijection(3, 3, vec![0, 1, 3]).is_err());
    }

    fn fibers_from_surjection(images: &[usize], card_codomain: usize) -> Vec<Vec<usize>> {
        let mut fibers = vec![Vec::new(); card_codomain];
        for (domain_rank, &codomain_rank) in images.iter().enumerate() {
            fibers[codomain_rank].push(domain_rank);
        }
        fibers
    }

    #[test]
    fn section_encoding_has_a_stable_mixed_radix_order() {
        let fibers = vec![vec![2, 0], vec![4, 1, 3]];
        let expected = [
            vec![0, 1],
            vec![2, 1],
            vec![0, 3],
            vec![2, 3],
            vec![0, 4],
            vec![2, 4],
        ];

        assert_eq!(number_of_sections(fibers.clone()).unwrap(), expected.len());
        for (code, section) in expected.into_iter().enumerate() {
            assert_eq!(decode_section(fibers.clone(), code).unwrap(), section);
            assert_eq!(encode_section(fibers.clone(), section).unwrap(), code);
        }
    }

    #[test]
    fn section_ranking_round_trips_for_all_small_surjections() {
        for n in 0..=6 {
            for m in 0..=n {
                for surjection_code in 0..count_surjections(n, m) {
                    let images = decode_surjection(n, m, surjection_code).unwrap();
                    let fibers = fibers_from_surjection(&images, m);
                    let section_count = number_of_sections(fibers.clone()).unwrap();

                    for section_code in 0..section_count {
                        let section = decode_section(fibers.clone(), section_code).unwrap();
                        assert_eq!(
                            encode_section(fibers.clone(), section.clone()).unwrap(),
                            section_code
                        );

                        let injection_code =
                            section_code_to_injection_code(fibers.clone(), section_code).unwrap();
                        assert_eq!(decode_injection(m, n, injection_code).unwrap(), section);
                    }
                }
            }
        }
    }

    #[test]
    fn the_empty_surjection_has_one_empty_section() {
        assert_eq!(number_of_sections(vec![]).unwrap(), 1);
        assert_eq!(decode_section(vec![], 0).unwrap(), Vec::<usize>::new());
        assert_eq!(encode_section(vec![], vec![]).unwrap(), 0);
        assert_eq!(section_code_to_injection_code(vec![], 0).unwrap(), 0);
    }

    #[test]
    fn section_functions_reject_invalid_inputs() {
        assert!(number_of_sections(vec![vec![0], vec![]]).is_err());
        assert!(number_of_sections(vec![vec![0], vec![0]]).is_err());
        assert!(number_of_sections(vec![vec![1]]).is_err());
        assert!(decode_section(vec![vec![0, 1]], 2).is_err());
        assert!(encode_section(vec![vec![0, 1]], vec![]).is_err());
        assert!(encode_section(vec![vec![0], vec![1]], vec![1, 0]).is_err());
        assert!(section_code_to_injection_code(vec![vec![0, 1]], 2).is_err());
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_function, m)?)?;
    m.add_function(wrap_pyfunction!(decode_function, m)?)?;
    m.add_function(wrap_pyfunction!(injection_count, m)?)?;
    m.add_function(wrap_pyfunction!(count_injections, m)?)?;
    m.add_function(wrap_pyfunction!(decode_injection, m)?)?;
    m.add_function(wrap_pyfunction!(encode_injection, m)?)?;
    m.add_function(wrap_pyfunction!(count_surjections, m)?)?;
    m.add_function(wrap_pyfunction!(decode_surjection, m)?)?;
    m.add_function(wrap_pyfunction!(encode_surjection, m)?)?;
    m.add_function(wrap_pyfunction!(bijection_number, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bijection, m)?)?;
    m.add_function(wrap_pyfunction!(encode_bijection, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_surjection_fibers, m)?)?;
    m.add_function(wrap_pyfunction!(number_of_sections, m)?)?;
    m.add_function(wrap_pyfunction!(decode_section, m)?)?;
    m.add_function(wrap_pyfunction!(encode_section, m)?)?;
    m.add_function(wrap_pyfunction!(section_code_to_injection_code, m)?)?;
    Ok(())
}
