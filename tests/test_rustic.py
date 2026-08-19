import rustic
import seth.core as seth
import pytest
import hyp.core as hyp

# print(rustic.encode_function(3, [1,2]))
# print(rustic.count_injections(6,11))
# # for i in range(rustic.count_injections(6,11)):
# #     print(rustic.decode_injection(6,11, i))

# print(rustic.count_injections(6,11))
# print(rustic.encode_injection(11, [1,2,3,4,5,6]))
# print(rustic.decode_injection(6,11, 500))
# print(rustic.encode_injection(11, rustic.decode_injection(6,11, 500)))

# print(rustic.count_surjections(8,5))
# print(rustic.decode_surjection(8,5, 500))


# print(rustic.bijection_number(4,4))
# for i in range(rustic.bijection_number(4,4)):
#     print(rustic.decode_bijection(4,4, i))


X = seth.NamedSet({'x', 2, 3}, "X")
Y = seth.NamedSet({'x', 2, X}, "Y")
T = seth.finset(4)
S6 = seth.finset(6)
S7 = seth.finset(7)

f3 = seth.NamedFunction(dom=X, cod=Y, table={2: 2, 'x': 'x', 3: X}, name="f3")

HomY4 = seth.HomSet(Y, T)

f5 = HomY4.access(5)

# print(X)
# print(X.enumeration)
# print(Y)
# print(Y.enumeration)
# print(T)



# print(f3.__repr__())
# print(f3._raw)
# print(f3.rustic_code)

# print(seth.Injection.from_function(f3).rustic_code_injection)

# print(seth.HomSet(X, Y))
# print(seth.HomSet(X, Y).generate().__repr__())
# print(seth.HomSet(X, Y).access(19).__repr__())

# print(HomY4.__repr__())
# print(f5.__repr__())


# print(seth.composition(f3, f5).__repr__())

# print(seth.HomSet(X, T).generate().__repr__())
# print(seth.Inj(X, T).__repr__())
# print(seth.Surj(X, T).__repr__())
# print(seth.Inj_inclusion(X, T).__repr__())

# S3= seth.finset(3)
# S4= seth.finset(4)

# print(seth.HomSet(S3, S4).__repr__())
# print(seth.Inj(S3, S4).__repr__())
# print(seth.Surj(S4, S3).__repr__())
# print(seth.Surj_inclusion(S4, S3).__repr__())
# print(seth.Bij(S4, S4).__repr__())

# print()


# def test_section_code_native_round_trip():
#     fibers = [[2, 0], [4, 1, 3]]
#     expected = [
#         [0, 1],
#         [2, 1],
#         [0, 3],
#         [2, 3],
#         [0, 4],
#         [2, 4],
#     ]

#     assert rustic.number_of_sections(fibers) == len(expected)
#     for code, section in enumerate(expected):
#         assert rustic.decode_section(fibers, code) == section
#         assert rustic.encode_section(fibers, section) == code
#         injection_code = rustic.section_code_to_injection_code(fibers, code)
#         assert rustic.decode_injection(2, 5, injection_code) == section

#     with pytest.raises(ValueError):
#         rustic.decode_section(fibers, len(expected))


# def test_surjection_section_code_api():
#     surjection = seth.Surjection.from_raw(
#         seth.finset(5),
#         seth.finset(2),
#         [0, 1, 0, 1, 1],
#         name="p",
#     )

#     assert surjection.number_of_sections() == 6
#     assert len(surjection.sections()) == 6

#     for code in range(surjection.number_of_sections()):
#         section = surjection.section_from_index(code)
#         assert surjection.section_index(section) == code
#         assert all(surjection(section(y)) == y for y in surjection.cod)
#         injection = seth.Injection.from_function(section)
#         assert (
#             surjection.section_code_to_injection_code(code)
#             == injection.rustic_code_injection
#         )



# inj0 = seth.Injection.from_injection_index(S3, S4, 1)
# inj1 = seth.Injection.from_injection_index(S3, S4, 2)


# print(seth.Inj(S3,S4).name, seth.Inj(S3, S4).__len__())
# print(seth.Inj(S3,S4).__repr__())
# print(inj0.__repr__())
# print(inj1.__repr__())
# print(seth.injections_factorization(S3, inj0, inj1))
# print(inj0.rustic_code_injection)

# print(seth.index_to_subset(S3, 7).__repr__())

# print(seth.powerset(S4).__repr__())

# print(seth.charmap_list(inj0))
# print(seth.list_to_subset(S4, [1, 1, 0, 1]).__repr__())

# print(seth.powerset(S4).__repr__())
# print(seth.index_to_binary(S4, 7))
# print(seth.list_to_subset(S4, [1, 1, 0, 1]).__repr__())

# f = seth.NamedFunction(dom=S3, cod=S4, table={0: 0, 1: 1, 2: 3}, name="f")

# print(f.__repr__())
# print(seth.powerset_contravariant(f).__repr__())
# print(seth.powerset_covariant(f).__repr__())
# #print(seth.powerset(seth.powerset(S4)).__repr__())

# print(seth.Rel(S3, S4).__repr__())

# print(hyp.hypset(seth.finset(3), seth.finset(4), seth.finset(4)).__repr__())
# print(len(hyp.hypset(seth.finset(3), seth.finset(4), seth.finset(4))))
# print(hyp.decode_hyp(seth.finset(4), seth.finset(3), seth.finset(8), 111122225).__repr__())


print(seth.HomSet(seth.finset(5), seth.finset(5)).generate().__repr__())