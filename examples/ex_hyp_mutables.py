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

    print(HX)

    HXMut = HX.hypergraph_to_mutable()
    print(HXMut.__repr__())
    HXMut.add(S={'x2'}, T={'t2' : ('x2', 'lx2')}, L={'lx2'})
    print(HXMut.__repr__())
    HXMut.remove(S={'x0'}, T={'t0'}, L={'lx'})
    print(HXMut.__repr__())

    HXMut.rename_elements(S_map={'x1': 'x1bis'}, T_map={'t1': 't1bis'}, L_map={'lx': 'lxbis'})
    print(HXMut.__repr__())
    HXMut.increment_node()
    print(HXMut.__repr__())
    HXMut.increment_link()
    print(HXMut.__repr__())

if __name__ == "__main__":
    main()