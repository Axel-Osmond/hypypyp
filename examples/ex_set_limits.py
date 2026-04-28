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

    print(seth.Pullback(f0,f1).display())
    print(seth.Pullback(f0,f1).proj_0().display())
    print(seth.Pullback(f0,f1).proj_1().display())

    print(seth.Pullback(f0,f1).universal_solution(g0,g1).display())

    u0 = seth.NamedFunction(name = "u0", dom = X, cod = Y, table = {'x0': 'y0', 'x1': 'y1', 'x2': 'y2'})
    u1 = seth.NamedFunction(name = "u1", dom = X, cod = Y, table = {'x0': 'y0', 'x1': 'y1', 'x2': 'y3'})

    print(seth.Equalizer(u0,u1).display())
    print(seth.Equalizer(u0,u1).inclusion().display())

    w = seth.NamedFunction(name = "w", dom = W, cod = X, table = {'w0': 'x0', 'w1': 'x1'})
    assert seth.composition(w,u0) == seth.composition(w,u1)
    print(seth.Equalizer(u0,u1).universal_solution(w).display())

if __name__ == "__main__":
    main()