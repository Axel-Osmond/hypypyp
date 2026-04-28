import seth
def main():
    X = seth.NamedSet(name = "X", elements = {1, 2, 3})
    Y = seth.NamedSet(name = "Y", elements = {"a", "b"})
    Z = seth.NamedSet(name = "Z", elements = {"x", "y", "z"})



    f_table = {
        1: "a",
        2: "a",
        3: "b"
    }
    f = seth.NamedFunction(name = "f", dom = X, cod = Y, table = f_table)

    g_table = {
        "a": "x",
        "b": "y"
    }
    g = seth.NamedFunction(name = "g", dom = Y, cod = Z, table = g_table)

    h_table = {
        "x" : 1,
        "y" : 2,
        "z" : 3
    }
    h = seth.NamedFunction(name = "h", dom = Z, cod = X, table = h_table)   

    print(f.display())
    print(f"f(1) = {f(1)}")

    print(g.display())
    print(seth.composition(f,g).display())

    print(seth.composition_chaine([f,g,h]).display())


    print(f.fiber('a').content())
    print(f.kernel().display())
    print(f'{f.image_factorization()[0].display()} \n {f.image_factorization()[1].display()}')

    print(f.fibers_decomposition().display())
    print(f.fibers_decomposition_isomorphism().display())

    print(f'inj : {f.injectivity_test_fast()}')
    print(f'surj : {f.surjectivity_test_fast()}')
    print(seth.powerset_contravariant(f).display())
    print(seth.powerset_covariant(f).display())

    f = seth.Surjection.from_function(f)
    print(type(f))
    print(f.sections().display())
    for s in f.sections():
        print(seth.composition(s,f).display())
        assert seth.composition(s,f) == Y.identity

    assert h.bijectivity_test_fast()
    h = seth.Bijection.from_function(h)
    print(h.inverse.display())

if __name__ == "__main__":
    main()