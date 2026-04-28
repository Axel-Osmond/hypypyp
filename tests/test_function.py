import seth



def test_function():
    X = seth.NamedSet(name = "X", elements = {1, 2, 3})
    Y = seth.NamedSet(name = "Y", elements = {"a", "b"})



    f_table = {
        1: "a",
        2: "a",
        3: "b"
    }
    f = seth.NamedFunction(name = "f", dom = X, cod = Y, table = f_table)

    print(f.display())
    print(f"f(1) = {f(1)}")
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