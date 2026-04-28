import seth
import hyp

def main():
    SX = seth.NamedSet(elements = {'x0', 'x1'}, name = 'SX')
    TX = seth.NamedSet(elements = {'t0', 't1'}, name = 'TX')
    LX = seth.NamedSet(elements = {'lx'}, name = 'LX')

    HX = hyp.Hypergraph(
    Nodes = SX,
    Ties = TX,
    Links = LX,
    node_map = seth.NamedFunction(
        dom = TX,
        cod = SX,
        table = {'t0': 'x0', 't1': 'x1'},
        name = 'node_map'),
    link_map = seth.NamedFunction(
        dom = TX,
        cod = LX,
        table = {'t0': 'lx', 't1': 'lx'},
        name = 'link_map'),
    name = 'HX'
    )

    SY = seth.NamedSet(elements = {'y0', 'y1'}, name = 'SY')
    TY = seth.NamedSet(elements = {'r0', 'r1'}, name = 'TY')
    LY = seth.NamedSet(elements = {'ly'}, name = 'LY')

    HY = hyp.Hypergraph(
    Nodes = SY,
    Ties = TY,
    Links = LY,
    node_map = seth.NamedFunction(
        dom = TY,
        cod = SY,
        table = {'r0': 'y0', 'r1': 'y1'},
        name = 'node_map'),
    link_map = seth.NamedFunction(
        dom = TY,
        cod = LY,
        table = {'r0': 'ly', 'r1': 'ly'},
        name = 'link_map'),
    name = 'HY'
    )

    SZ = seth.NamedSet(elements = {'z0', 'z1'}, name = 'SZ')
    TZ = seth.NamedSet(elements = {'w0', 'w1'}, name = 'TZ')
    LZ = seth.NamedSet(elements = {'lz'}, name = 'LZ')

    HZ = hyp.Hypergraph(
    Nodes = SZ,
    Ties = TZ,
    Links = LZ,
    node_map = seth.NamedFunction(
        dom = TZ,
        cod = SZ,
        table = {'w0': 'z0', 'w1': 'z1'},
        name = 'node_map'),
    link_map = seth.NamedFunction(
        dom = TZ,
        cod = LZ,
        table = {'w0': 'lz', 'w1': 'lz'},
        name = 'link_map'),
    name = 'HZ'
    )

    Sf = seth.NamedFunction(
    dom = SX,
    cod = SY,
    table = {'x0': 'y0', 'x1': 'y1'},
    name = 'Sf'
    )
    Tf = seth.NamedFunction(
        dom = TX,
        cod = TY,
        table = {'t0': 'r0', 't1': 'r1'},
        name = 'Tf'
    )
    Lf = seth.NamedFunction(
        dom = LX,
        cod = LY,
        table = {'lx': 'ly'},
        name = 'Lf'
    )

    f = hyp.HypergraphMorphism(
    dom = HX,
    cod = HY,
    map = (Sf, Tf, Lf),
    name = 'f'
    )

    Sg = seth.NamedFunction(
    dom = SY,
    cod = SZ,
    table = {'y0': 'z0', 'y1': 'z1'},
    name = 'Sg'
    )
    Tg = seth.NamedFunction(
        dom = TY,
        cod = TZ,
        table = {'r0': 'w0', 'r1': 'w1'},
        name = 'Tg'
    )
    Lg = seth.NamedFunction(
        dom = LY,
        cod = LZ,
        table = {'ly': 'lz'},
        name = 'Lg'
    )

    g = hyp.HypergraphMorphism(
    dom = HY,
    cod = HZ,
    map = (Sg, Tg, Lg),
    name = 'g'
    )


    print(f)
    print(f.display())

    print(f.test_morphisme_concret())

    print(f.eval_occurences('x0').display())
    print(f.eval_support('lx').display())

    print(g)
    print(g.display())

    print(hyp.composition(f, g).display())

    f.test_iso()
    f = hyp.HypergraphIsomorphism.from_morphism(f)
    print(f.inverse.display())

    g.test_epi()
    g = hyp.HypergraphEpimorphism.from_morphism(g)
    print(g.sections().display())

    for section in g.sections():
        assert hyp.composition(section, g) == HZ.identity


if __name__ == "__main__":
    main()