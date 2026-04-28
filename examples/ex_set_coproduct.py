import seth
def main():
    X = seth.NamedSet(name = "X", elements = {1, 2, 3})
    Y = seth.NamedSet(name = "Y", elements = {"a", "b"})
    Z = seth.NamedSet(name = "Z", elements = {'x', 'y', 'z'})
    I = seth.NamedSet(name = "I", elements = set())

    XY = seth.Coproduct(X,Y)

    print(seth.Initial().display())
    print(seth.Initial().unique_map(X).display())
    print(seth.Initial().unique_map(I).display())
    assert seth.initial_detection(I)== True
    assert seth.initial_detection(X)== False


    print(XY.content())
    print(XY.inj_0().display())
    print(XY.inj_1().display())

    u1_table = {1: 'x', 2: 'y', 3: 'z'}
    u1 = seth.NamedFunction(name= 'u1', dom = X, cod = Z, table = u1_table)
    u2_table = {'a': 'x', 'b': 'y'}
    u2 = seth.NamedFunction(name= 'u2', dom = Y, cod = Z, table = u2_table)

    print(XY.universal_solution(u1,u2).display())
    print(seth.composition(XY.inj_0(),XY.universal_solution(u1,u2)).display())

    print(XY.braiding().display())
    print(seth.unitor_coproduct_left(X).display())
    print(seth.associator_coproduct(X,Y,Y).display())

    print(seth.codiagonal(X).display())


    f0_table = {
        1: "a",
        2: "a",
        3: "b"
    }
    f0 = seth.NamedFunction(name = "f0", dom = X, cod = Y, table = f0_table)

    f1_table = {
        1: "a",
        2: "b",
        3: "b"
    }
    f1 = seth.NamedFunction(name = "f1", dom = X, cod = Y, table = f1_table)

    print(seth.coproduct_maps(f0,f1).display())

    print(seth.FiniteCoproduct([X,X,X]).display())
    print(seth.FiniteCoproduct([X,X,X]).inj(1).display())
    print(seth.finite_coproduct_maps([f0,f0,f0]).display())

    print(seth.Coproduct(X,seth.FiniteCoproduct([X,X,X])).display())

    print(seth.left_distributivity_isomorphism(ext=X,paire=(Y,Z)).display())

if __name__ == "__main__":
    main()