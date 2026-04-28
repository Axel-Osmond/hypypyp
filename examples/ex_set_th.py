import seth

def main():
    X = seth.NamedSet(name = "X", elements = {1, 2, 3})
    Y = seth.NamedSet(name = "Y", elements = {"a", "b"})
    U = seth.NamedSet(name = "U", elements = {X, Y})

    W = seth.union_of_elements(U)
    i = seth.inclusion(X, W)


    print(X.content())
    PX = seth.powerset(X)
    print(PX.content())

    print(seth.singleton(X).content())

    print(seth.pair(X, Y).content())
    print(seth.ordered_pair(X, Y).content())
    print(seth.union_of_elements(U).content())
    print(seth.ordinal(5))

    print(seth.Omega_set().content())
    print(seth.Top().display())
    print(i.display())
    print(seth.charmap(i).display())

if __name__ == "__main__":
    main()