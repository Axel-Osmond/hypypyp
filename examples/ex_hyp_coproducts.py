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

    print(hyp.Initial().display())
    print(hyp.Initial().unique_map(HX).display())

    f0 =  hyp.HypergraphMorphism(
        dom = HX,
        cod = HZ,
        map = ( seth.NamedFunction(
            dom = SX,
            cod = SZ,
            table = {'x0': 'z0', 'x1': 'z1'},
            name = 'node_map'),
                seth.NamedFunction(
            dom = TX,
            cod = TZ,
            table = {'t0': 'w0', 't1': 'w1'},
            name = 'tie_map'),
                seth.NamedFunction(
            dom = LX,
            cod = LZ,
            table = {'lx': 'lz'},
            name = 'link_map')), 
        name = 'f0' 
    )

    f1 =  hyp.HypergraphMorphism(
        dom = HY,
        cod = HZ,
        map = ( seth.NamedFunction(
            dom = SY,
            cod = SZ,
            table = {'y0': 'z0', 'y1': 'z1'},
            name = 'node_map'),
                seth.NamedFunction(
            dom = TY,
            cod = TZ,
            table = {'r0': 'w0', 'r1': 'w1'},
            name = 'tie_map'),
                seth.NamedFunction(
            dom = LY,
            cod = LZ,
            table = {'ly': 'lz'},
            name = 'link_map')), 
        name = 'f1' 
    )

    print(f0.display())
    print(f1.display())

    HXY = hyp.Coproduct(HX, HY)
    print(HXY.display())
    print(HXY.inj_0().display())
    print(HXY.universal_solution(f0,f1).display())

    print(HXY.braiding().display())

    print(hyp.unitor_coproduct_left(HX).display())

    print(hyp.associator_coproduct(HX, HY, HZ).display())

    print(hyp.FiniteCoproduct([HX, HY, HZ]).display())
    print(hyp.FiniteCoproduct([HX, HY, HZ]).inj(1).display())

    print(hyp.FiniteCoproduct([HX,HY]).universal_solution([f0, f1]).display())

if __name__ == "__main__":
    main()