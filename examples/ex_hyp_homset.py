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

    print(HX.display())
    print(HY.display())

    print(hyp.CartesianHomSet(HX, HY).display())

    print(hyp.hom_coconforme(HX, HY).display())
    print(hyp.hom_conforme(HY, HX).display())
    print(hyp.hom_mono(HX, HY).display())
    print(hyp.hom_epi(HX, HY).display())
    print(hyp.Iso(HX, HY).display())

if __name__ == "__main__":
    main()