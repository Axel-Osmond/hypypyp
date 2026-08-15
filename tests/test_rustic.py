import seth._native as rustic
import seth.core as seth

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

S3= seth.finset(3)
S4= seth.finset(4)

print(seth.HomSet(S3, S4).__repr__())
print(seth.Inj(S3, S4).__repr__())
print(seth.Surj(S4, S3).__repr__())
print(seth.Surj_inclusion(S4, S3).__repr__())
print(seth.Bij(S4, S4).__repr__())

print()