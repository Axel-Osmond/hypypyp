import seth
def main():
    X = seth.NamedSet(name = "X0", elements = {'x0', 'x1', 'x2'})
    Y = seth.NamedSet(name = "X1", elements = {'y0', 'y1', 'y2', 'y3'})
    Z = seth.NamedSet(name = "X2", elements = {'z0', 'z1'})
    W = seth.NamedSet(name = "X3", elements = {'w0', 'w1'})

    f0 = seth.NamedFunction(name = "f0", dom = X, cod = Z, table = {'x0': 'z0', 'x1': 'z0', 'x2': 'z1'})
    f1 = seth.NamedFunction(name = "f1", dom = Y, cod = Z, table = {'y0': 'z0', 'y1': 'z0', 'y2': 'z1', 'y3': 'z1'})

    g0 = seth.NamedFunction(name = "g0", dom = W, cod = X, table = {'w0': 'x0', 'w1': 'x0'})
    g1 = seth.NamedFunction(name = "g1", dom = W, cod = Y, table = {'w0': 'y0', 'w1': 'y1'})

    assert seth.composition(g0,f0) == seth.composition(g1,f1)

    print(f0.display())
    print(f1.display())

    print(seth.Pushout(g0,g1).display())
    print(seth.Pushout(g0,g1).inj_0().display())
    print(seth.Pushout(g0,g1).inj_1().display())

    print(seth.Pushout(g0,g1).universal_solution(f0,f1).display())

    u0 = seth.NamedFunction(name = "u0", dom = X, cod = Y, table = {'x0': 'y0', 'x1': 'y1', 'x2': 'y2'})
    u1 = seth.NamedFunction(name = "u1", dom = X, cod = Y, table = {'x0': 'y0', 'x1': 'y1', 'x2': 'y3'})

    print(seth.Coequalizer(u0,u1).display())
    print(seth.Coequalizer(u0,u1).projection().display())

    w = seth.NamedFunction(name = "w", dom = Y, cod = W, table = {'y0': 'w0', 'y1': 'w1', 'y2': 'w0', 'y3': 'w0'})
    assert seth.composition(u0,w) == seth.composition(u1,w)
    print(seth.Coequalizer(u0,u1).universal_solution(w).display())

if __name__ == "__main__":
    main()