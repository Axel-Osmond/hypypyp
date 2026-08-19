import rustic
import seth.core as seth
import pytest
import hyp.core as hyp

Wedge = hyp.Hypergraph(
            Nodes = seth.finset(3).rename("S_┌"),
            Links = seth.NamedSet(elements = {'01', '02'}, name = "L_┌"),
            Ties = seth.NamedSet(elements = {'t_1', 't_2', 'r_1', 'r_2'}, name = "T_┌"),
            node_map = seth.NamedFunction(
                dom = seth.NamedSet(elements = {'t_1', 't_2', 'r_1', 'r_2'}, name = "T_┌"),
                cod = seth.finset(3).rename("S_┌"),
                table = {'t_1': 0, 't_2': 0, 'r_1': 1, 'r_2': 2},
                name = "node_map_┌"
            ),
            link_map = seth.NamedFunction(
                dom = seth.NamedSet(elements = {'t_1', 't_2', 'r_1', 'r_2'}, name = "T_┌"),
                cod = seth.NamedSet(elements = {'01', '02'}, name = "L_┌"),
                table = {'t_1': '01', 't_2': '02', 'r_1': '01', 'r_2': '02'},
                name = "link_map_┌" 
            ),
            name = "┌"
)

Square = hyp.Hypergraph(
            Nodes = seth.finset(4).rename("S_□"),
            Links = seth.NamedSet(elements = {'01', '02', '13', '23'}, name = "L_□"),
            Ties = seth.NamedSet(elements = {'t_1', 't_2', 'r_1', 'r_2', "t_1'", "t_2'", "r_1'", "r_2'"}, name = "T_□"),
            node_map = seth.NamedFunction(
                dom = seth.NamedSet(elements = {'t_1', 't_2', 'r_1', 'r_2', "t_1'", "t_2'", "r_1'", "r_2'"}, name = "T_□"),
                cod = seth.finset(4).rename("S_□"),
                table = {'t_1': 0, 't_2': 0, 'r_1': 1, 'r_2': 2, "t_1'": 3, "t_2'": 3, "r_1'":1, "r_2'": 2},
                name = "node_map_□"
            ),
            link_map = seth.NamedFunction(
                dom = seth.NamedSet(elements = {'t_1', 't_2', 'r_1', 'r_2', "t_1'", "t_2'", "r_1'", "r_2'"}, name = "T_□"),
                cod = seth.NamedSet(elements = {'01', '02', '13', '23'}, name = "L_□"),
                table = {'t_1': '01', 't_2': '02', 'r_1': '01', 'r_2': '02', "t_1'": '13', "t_2'": '23', "r_1'": '13', "r_2'": '23'},
                name = "link_map_□" 
            ),
            name = "Square"
)

hyp.test_inclusion_sub_fast(Wedge, Square)

def cartesian_homset_functoriality_left(f : hyp.HypergraphMorphism, H : hyp.Hypergraph):
    return seth.NamedFunction(dom = hyp.CartesianHomSet(f.cod, H), 
                              cod = hyp.CartesianHomSet(f.dom, H), 
                              table = {g: hyp.composition(f,g) for g in hyp.CartesianHomSet(f.cod, H)}, 
                              name = f"[{f.name}, {H.name}]")


H = hyp.reticulation(2,2).canonical_elements_naming()




iota = hyp.inclusion_hypergraph(Wedge, Square)




X = seth.finset(4)
Y = seth.finset(3)


if __name__ == "__main__":
    print("Wedge:", Wedge.__repr__())
    print("Square:", Square.__repr__())
    print("H:", H.__repr__())
    print("Decomp(H):", hyp.decomp_hyp(H).__repr__())
    print(iota.__repr__())
    print(hyp.homgraph_cardinality_fast(Wedge, H).__repr__())
    print(hyp.CartesianHomSet(Wedge, H).__repr__())
    print(hyp.CartesianHomSet(H, H).__repr__())
    print(hyp.CartesianHomSet(H, H).__len__())
    print(hyp.CartesianHomSet(H, H).cardinality)
