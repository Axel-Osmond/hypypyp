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

    HXY = hyp.FunnyTensor(HX, HY)
    print(HXY.display())
    print(HXY.canonical_elements_naming().display())

    print(HXY.braiding().display())
    print(hyp.funny_left_unitor(HX).display())
    print(hyp.funny_right_unitor(HX).display())
    print(hyp.funny_associator(HX, HY, HXY).display())

    H2 = hyp.walking_link(2)
    print(H2.display())
    H22 = hyp.FunnyTensor(H2, H2).canonical_elements_naming()
    print(H22.display())
    


if __name__ == "__main__":
    main()