import hyp
import seth

def main():
    SX = seth.NamedSet(elements = {'x0', 'x1', 'y0', 'y1', 'z', 's'}, name = 'SX')
    TX = seth.NamedSet(elements = {'tx0l0', 
                                   'tx1l1', 
                                   'ty0l0', 
                                   'ty1l1',
                                   'tx0lx',
                                   'tx1lx',
                                   'ty0ly',
                                   'ty1ly', 
                                   'tx0lxy', 
                                   'tx1lxy', 
                                   'ty0lxy', 
                                   'ty1lxy', 
                                   'ts0', 
                                   'ts1'}, name = 'TX')
    LX = seth.NamedSet(elements = {'lx', 'ly', 'l0', 'l1', 'lxy', 'l', 'ls'}, name = 'LX')
    HX = hyp.Hypergraph(
        Nodes = SX,
        Ties = TX,
        Links = LX,
        node_map = seth.NamedFunction(
            dom = TX,
            cod = SX,
            table = {'tx0l0': 'x0', 
                     'tx1l1': 'x1', 
                     'ty0l0': 'y0', 
                     'ty1l1': 'y1',
                     'tx0lx': 'x0',
                     'tx1lx': 'x1',
                     'ty0ly': 'y0',
                     'ty1ly': 'y1',
                     'tx0lxy': 'x0', 
                     'tx1lxy': 'x1', 
                     'ty0lxy': 'y0', 
                     'ty1lxy': 'y1',
                     'ts0': 's',
                     'ts1': 's'},
            name = 'node_map'),
        link_map = seth.NamedFunction(
            dom = TX,
            cod = LX,
            table = {'tx0l0': 'l0', 
                     'tx1l1': 'l1', 
                     'ty0l0': 'l0', 
                     'ty1l1': 'l1',
                     'tx0lx': 'lx',
                     'tx1lx': 'lx',
                     'ty0ly': 'ly',
                     'ty1ly': 'ly', 
                     'tx0lxy': 'lxy', 
                     'tx1lxy': 'lxy', 
                     'ty0lxy': 'lxy', 
                     'ty1lxy': 'lxy',
                     'ts0': 'ls',
                     'ts1': 'ls'},
            name = 'link_map'),
        name = 'HX'
    )

    print(HX.display())
    print(HX.sizes())
    print(HX.node_map.display())
    print(HX.link_map.display())

    print(HX.dictionnaire)
    print(HX.nodes_support)

    print(HX.support_ties('lx').content())
    print(HX.support_nodes('lx').content())
    print(HX.occurrences_ties('x0').content())
    print(HX.occurrences_links('x0').content())

    print(HX.valence_set('x0','lx').content())
    print(HX.bipartite())
    print(HX.emptylinks())
    print(HX.nakednodes())

    print(HX.intersections().content())
    print(HX.intersection_nodes('lx', 'l0').content())

    print(HX.incidences().content())
    print(HX.cooccurences_links('x0', 'x1').content())

    print(HX.dual.display())
    print(HX.bidual_isomorphism.display())

if __name__ == "__main__":
    main()