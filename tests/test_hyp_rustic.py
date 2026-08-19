import pytest

import hyp.core as hyp
import rustic
import seth.core as seth


def test_hypergraph_py_delegates_to_the_rust_core():
    hypergraph = rustic.HypergraphPy(
        nodes=2,
        links=2,
        pairs=[(0, 0), (0, 0), (1, 0), (1, 1)],
    )

    assert hypergraph.nodes == 2
    assert hypergraph.links == 2
    assert hypergraph.ties == 4
    assert hypergraph.incidences() == [(0, 0), (1, 0), (1, 1)]
    assert not hypergraph.test_simple()
    assert hypergraph.loops() == [(1, 1, 1)]

    with pytest.raises(ValueError, match="outside range"):
        rustic.HypergraphPy(nodes=1, links=1, pairs=[(1, 0)])


def test_morphism_py_uses_three_maps_and_validates_commutativity():
    hypergraph = rustic.HypergraphPy(
        nodes=2,
        links=1,
        pairs=[(0, 0), (1, 0)],
    )
    identity = rustic.MorphismPy(
        source=hypergraph,
        target=hypergraph,
        node_map=[0, 1],
        tie_map=[0, 1],
        link_map=[0],
    )

    assert identity.mapping == ([0, 1], [0, 1], [0])
    assert identity.source.pairs == hypergraph.pairs
    assert identity.target.pairs == hypergraph.pairs
    assert identity.test_morphism()
    assert identity.test_iso()

    with pytest.raises(ValueError, match="does not preserve"):
        rustic.MorphismPy(
            source=hypergraph,
            target=hypergraph,
            node_map=[1, 0],
            tie_map=[0, 1],
            link_map=[0],
        )


def test_native_homgraph_cardinality_counts_tie_map_choices():
    source = rustic.HypergraphPy(nodes=1, links=1, pairs=[(0, 0), (0, 0)])
    target = rustic.HypergraphPy(
        nodes=1,
        links=1,
        pairs=[(0, 0), (0, 0), (0, 0)],
    )

    assert rustic._cardinality_indexed_by_nodes(source, target) == 9
    assert rustic._cardinality_indexed_by_links(source, target) == 9
    assert rustic.homgraph_cardinality_fast(source, target) == 9


def test_native_homgraph_cardinality_reports_overflow():
    source = rustic.HypergraphPy(nodes=1, links=1, pairs=[(0, 0)] * 100)
    target = rustic.HypergraphPy(nodes=1, links=1, pairs=[(0, 0)] * 2)

    with pytest.raises(OverflowError, match="homgraph cardinality|tie-map choices"):
        rustic.homgraph_cardinality_fast(source, target)


def test_native_hypergraph_morphism_decoder_supports_python_bigints():
    source = rustic.HypergraphPy(nodes=1, links=1, pairs=[(0, 0)] * 100)
    target = rustic.HypergraphPy(nodes=1, links=1, pairs=[(0, 0)] * 2)

    morphism = rustic.decode_hypergraph_morphism_by_nodes(source, target, 1 << 99)

    assert morphism.node_map == [0]
    assert morphism.link_map == [0]
    assert morphism.tie_map == [0] * 99 + [1]
    assert morphism.test_morphism()

    with pytest.raises(IndexError, match="outside the hom-set"):
        rustic.decode_hypergraph_morphism_by_nodes(source, target, 1 << 100)


def test_hyp_objects_are_connected_to_the_python_wrappers():
    nodes = seth.finset(2)
    ties = seth.finset(2)
    links = seth.finset(1)
    hypergraph = hyp.Hypergraph(
        Nodes=nodes,
        Ties=ties,
        Links=links,
        node_map=seth.NamedFunction(
            dom=ties,
            cod=nodes,
            table={0: 0, 1: 1},
            name="node_map",
        ),
        link_map=seth.NamedFunction(
            dom=ties,
            cod=links,
            table={0: 0, 1: 0},
            name="link_map",
        ),
        name="H",
    )
    identity = hyp.HypergraphMorphism(
        dom=hypergraph,
        cod=hypergraph,
        map=(nodes.identity, ties.identity, links.identity),
        name="id_H",
    )

    assert isinstance(hypergraph.rustic, rustic.HypergraphPy)
    assert isinstance(identity.rustic, rustic.MorphismPy)
    assert identity.rustic.test_iso()

    homset = hyp.CartesianHomSet(hypergraph, hypergraph)
    exhaustive_cardinality = len(homset.generate())
    assert hyp._cardinality_indexed_by_nodes(hypergraph, hypergraph) == exhaustive_cardinality
    assert hyp._cardinality_indexed_by_links(hypergraph, hypergraph) == exhaustive_cardinality
    assert hyp.homgraph_cardinality_fast(hypergraph, hypergraph) == exhaustive_cardinality
    assert len(homset) == exhaustive_cardinality
