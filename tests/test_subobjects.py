import pytest

import seth


def test_subset_representations_round_trip_in_code_order():
    A = seth.NamedSet({"first", "second", "third"}, "A", enum=("third", "first", "second"))
    PA = seth.powerset(A)

    assert len(PA) == 2 ** len(A)
    for index in range(2 ** len(A)):
        bits = seth.index_to_subset_charmap(A, index)
        subset = seth.subset_charmap_to_subset(A, bits)
        subobject = seth.subset_charmap_to_subobject(A, bits)

        assert seth.subset_charmap_to_index(A, bits) == index
        assert seth.subset_to_subset_charmap(A, subset) == bits
        assert seth.subset_to_index(A, subset) == index
        assert seth.subobject_to_subset_charmap(subobject) == bits
        assert seth.subobject_to_index(subobject) == index
        assert seth.index_to_subobject(A, index) == subobject
        assert PA.unrank(index) == subset
        assert PA.rank(subset) == index


def test_subset_representation_validation():
    A = seth.finset(3)

    with pytest.raises(ValueError):
        seth.index_to_subset_charmap(A, -1)
    with pytest.raises(ValueError):
        seth.index_to_subset_charmap(A, 8)
    with pytest.raises(ValueError):
        seth.subset_charmap_to_index(A, [1, 0])
    with pytest.raises(ValueError):
        seth.subset_charmap_to_index(A, [1, 2, 0])


def test_contravariant_powerset_uses_raw_rank_preimages():
    A = seth.NamedSet({"a", "b", "c"}, "A", enum=("c", "a", "b"))
    B = seth.NamedSet({"x", "y"}, "B", enum=("y", "x"))
    PB = seth.powerset(B)

    for function_index in range(len(B) ** len(A)):
        f = seth.NamedFunction.from_index(A, B, function_index, name=f"f_{function_index}")
        inverse_image = seth.powerset_contravariant(f)

        for subset_index in range(2 ** len(B)):
            codomain_bits = seth.index_to_subset_charmap(B, subset_index)
            expected_bits = [codomain_bits[f._raw[i]] for i in range(len(A))]
            expected_index = seth.subset_charmap_to_index(A, expected_bits)
            actual_subset = inverse_image(PB.unrank(subset_index))

            assert seth.preim(f, subset_index) == expected_bits
            assert seth.subset_to_subset_charmap(A, actual_subset) == expected_bits
            assert seth.powerset(A).rank(actual_subset) == expected_index


def test_subobject_factorization_uses_subset_codes_not_injection_ranks():
    A = seth.finset(4)
    small = seth.index_to_subobject(A, 0b0101)
    large = seth.index_to_subobject(A, 0b1101)
    other = seth.index_to_subobject(A, 0b1010)

    assert seth.injections_factorization(A, small, large)
    assert not seth.injections_factorization(A, small, other)


def test_empty_set_has_one_subset_with_code_zero():
    A = seth.finset(0)

    assert seth.index_to_subset_charmap(A, 0) == []
    assert seth.subset_charmap_to_index(A, []) == 0
    assert len(seth.powerset(A)) == 1
