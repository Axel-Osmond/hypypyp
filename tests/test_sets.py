import seth

def test_sets():
    X = seth.NamedSet(name = "X", elements = {1, 2, 3})
    Y = seth.NamedSet(name = "Y", elements = {"a", "b"})


    print(X.content())
    print(Y.content())
    print(f"|{X}| = {X.__len__()}")
    print(f"|{Y}| = {Y.__len__()}")

    Z = seth.NamedSet(name = "Z", elements = {X,Y})
    U = seth.NamedSet(name = "U", elements = {X, Y, Z})

    PX = seth.powerset(X)

    print(Z.content())
    print(Z.display())
    print(U.content())
    print(U.display())
    print(PX.content())

