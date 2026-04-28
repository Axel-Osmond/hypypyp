import seth
def main():
    X = seth.NamedSet(name = "X", elements = {1, 2, 3})
    Y = seth.NamedSet(name = "Y", elements = {"a", "b"})
    T = seth.NamedSet(name = "T", elements = {1})

    XY = seth.Product(X,Y)

    print(seth.Terminal().display())
    print(seth.Terminal().unique_map(X).display())
    print(seth.Terminal().unique_map(T).display())
    assert seth.terminal_detection(T)== True
    assert seth.terminal_detection(X)== False


    print(XY.content())
    print(XY.proj_0().display())
    print(XY.proj_1().display())

    p1 = seth.NamedFunction(name= '1', dom = T, cod = X, table = {1: 1})
    pa = seth.NamedFunction(name= 'a', dom = T, cod = Y, table = {1: 'a'})

    print(XY.universal_solution(p1,pa).display())
    print(seth.composition(XY.universal_solution(p1,pa), XY.proj_0()).display())

    print(XY.braiding().display())
    print(seth.unitor_cartesian_left(X).display())
    print(seth.associator_cartesian(X,Y,Y).display())

    print(seth.diagonal(X).display())



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

    print(seth.product_maps(f0,f1).display())

    print(seth.FiniteProduct([X,X,X]).display())
    print(seth.FiniteProduct([X,X,X]).proj(1).display())
    print(seth.finite_product_maps([f0,f0,f0]).display())

    print(seth.Product(X,seth.FiniteProduct([X,X,X])).display())

    print(seth.detection_binary_product(XY))
    print(seth.decomposition_binary_product(XY).display())

    XYobj = XY.obj
    print(seth.finite_product_detection(XYobj))
    print(seth.finite_product_decomposition(XYobj).display())

if __name__ == "__main__":
    main()