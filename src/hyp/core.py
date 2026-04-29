# Hypypyp

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping, Callable, Sequence
import itertools as it
import numpy as np
from typing import TypeAlias

import seth
from seth import (
    NamedSet,
    NamedFunction,
    HomSet,
    Omega_set,
    subset_charmap,
    union,
    pair,
    ordinal,
)

#################################################################

# Base types

## -- Representables dans Hyp

"""A representable is an object with an underlying hypergraph
It will be the parent class of :
- Hypergraph, that is, a span of sets with a name and a custom display method
- Classes for universal constructions like limits and colimits
- Tensor products, which will have an underlying hypergraph together with structure
- homsets and some distributors"""


class Representable(ABC):
    """A representable is an object with an underlying hypergraph.
    This superclass promises:

    - that the underlying hypergraph is accessible via the obj property
    - that the representable has a name, representation based on the underlying hypergraph
    
    Representable are immutable, hence hashable,
    and equality is based on the underlying hypergraph, independently of the name.
    It also has identity morphism and dual construction"""

    @property
    @abstractmethod
    def obj(self) -> Hypergraph:
        """Underlying representing hypergraph."""
        pass

    @property
    def identity(self) -> HypergraphIsomorphism:
        """Return the identity function on the hypergraph.
        This is the triple made of the identities maps
        on nodes, ties and links,
        which is a morphism of hypergraphes,
        and in fact an isomorphism."""
        regle_id = lambda x: x
        S_id = NamedFunction(
            dom=self.Nodes,
            cod=self.Nodes,
            table=regle_id,
            name=f"id_S_{self.name}",
        )
        T_id = NamedFunction(
            dom=self.Ties,
            cod=self.Ties,
            table=regle_id,
            name=f"id_T_{self.name}",
        )
        L_id = NamedFunction(
            dom=self.Links,
            cod=self.Links,
            table=regle_id,
            name=f"id_L_{self.name}",
        )
        return HypergraphIsomorphism(
            dom=self, cod=self, map=(S_id, T_id, L_id), name=f"id_{self.name}"
        )

    @property
    def dual(self) -> Hypergraph:
        """Return the dual of the hypergraph,
        obtained by swapping nodes and links
        
        H* satisfies:

            - S(H*) = L(H)
            - L(H*) = S(H)
            - T(H*) = T(H)
        """
        if isinstance(self, Binary_Type):
            name = f"({self.name})*"
        else:
            name = f"{self.name}*"
        return Hypergraph(
            Nodes=self.Links,
            Ties=self.Ties,
            Links=self.Nodes,
            node_map=self.link_map,
            link_map=self.node_map,
            name=name,
        )

    @property
    def bidual_isomorphism(self) -> HypergraphIsomorphism:
        """Return the isomorphism between the hypergraph and its bidual,
        which is the identity on the underlying sets and maps."""
        return HypergraphIsomorphism(
            dom=self,
            cod=self.dual.dual,
            map=(
                self.Nodes.identity,
                self.Ties.identity,
                self.Links.identity,
            ),
            name=f"bidual_iso_{self.name}",
        )

    @property
    def name(self) -> str:
        """Invoque the name of the underlying hypergraph."""
        return self.obj.name

    @property
    def Nodes(self) -> NamedSet:
        """Invoque the nodes of the underlying hypergraph."""
        return self.obj.Nodes

    @property
    def Ties(self) -> NamedSet:
        """Invoque the ties of the underlying hypergraph."""
        return self.obj.Ties

    @property
    def Links(self) -> NamedSet:
        """Invoque the links of the underlying hypergraph."""
        return self.obj.Links

    @property
    def node_map(self) -> NamedFunction:
        """Invoque the node_map of the underlying hypergraph."""
        return self.obj.node_map

    @property
    def link_map(self) -> NamedFunction:
        """Invoque the link_map of the underlying hypergraph."""
        return self.obj.link_map

    def __eq__(self, other: object) -> bool:
        """Equality based on the underlying set, independently of the name."""
        if not isinstance(other, Representable):
            return NotImplemented
        return self.obj == other.obj

    def __hash__(self):
        """Hash based on the underlying set."""
        return hash(self.obj)

    def __repr__(self) -> str:
        """Repr based on the underlying set."""
        return repr(self.obj)

    def __str__(self) -> str:
        """Str based on the underlying set."""
        return str(self.obj)

    def display(self) -> str:
        """Display based on the underlying set."""
        return self.obj.display()

    def rename(self, new_name: str) -> Representable:
        """Rename based on the underlying set."""
        return self.obj.rename(new_name)


## -- Constructs --


class Construct(Representable):
    """Construct represent the result of a universal construction
    It is a representable, hence has a name
    .obj returns an underlying object which is always a Hypergraph.
    It is eligible as a domain or codomain of functions,
    and can be used in further constructions,
    without losing the structure supported by the underlying hypergraph."""

    _obj: Hypergraph
    _name: str

    @property
    def obj(self) -> Hypergraph:
        """Invoque the underlying hypergraph."""
        return self._obj

    @property
    def name(self) -> str:
        """Invoque the name of the underlying hypergraph."""
        return self._name

    @property
    def Nodes(self):
        """Invoque the nodes of the underlying hypergraph."""
        return getattr(self, "_Nodes", self.obj.Nodes)

    @property
    def Ties(self):
        """Invoque the ties of the underlying hypergraph."""
        return getattr(self, "_Ties", self.obj.Ties)

    @property
    def Links(self):
        """Invoque the links of the underlying hypergraph."""
        return getattr(self, "_Links", self.obj.Links)

    @property
    def node_map(self):
        """Invoque the node_map of the underlying hypergraph."""
        return getattr(self, "_node_map", self.obj.node_map)

    @property
    def link_map(self):
        """Invoque the link_map of the underlying hypergraph."""
        return getattr(self, "_link_map", self.obj.link_map)

    def __repr__(self) -> str:
        """Repr based on the underlying set."""
        return repr(self.obj)

    def __str__(self) -> str:
        """Str based on the underlying set."""
        return str(self.obj)


# Hypergraphes : classe, présentation


class Hypergraph(Representable):
    """
    A (finite) hypergraph represented as a span of Named Sets and NamedFunctions,
    with a name and a custom display method.

    A hypergraph consists of:

    - a set of nodes ``S``;
    - a set of links ``L``;
    - a set of ties ``T``;
    - a node map ``σ : T → S``;
    - a link map ``λ : T → L``.

    Args:
        nodes: The named set of nodes.
        links: The named set of links.
        ties: The named set of incidence witnesses.
        node_map: The function sending each tie to its node.
        link_map: The function sending each tie to its link.

    Notes:
        A node may occur several times in the same link, because incidences
        are represented by explicit ties.
    """

    def __init__(
        self,
        Nodes: seth.Representable,
        Ties: seth.Representable,
        Links: seth.Representable,
        node_map: NamedFunction,
        link_map: NamedFunction,
        name: str,
    ):

        self._Nodes = Nodes
        self._Links = Links
        self._Ties = Ties
        self._node_map = node_map
        self._link_map = link_map
        self._name = name if name else f"Hyp({Nodes.name}, {Links.name}, {Ties.name})"

        self.nodes_dict = {}
        self.links_dict = {}
        self.nodes_support: dict = {s: set() for s in self.Nodes}
        self.links_support: dict = {l: set() for l in self.Links}
        self.valences: dict = {(s, l): set() for s in self.Nodes for l in self.Links}

        for t in self.Ties:
            s = self.node_map(t)
            l = self.link_map(t)
            self.nodes_dict[t] = s
            self.links_dict[t] = l
            self.nodes_support[s].add(t)
            self.links_support[l].add(t)
            self.valences[(s, l)].add(t)

        self.node_to_links = {
            s: {self.links_dict[t] for t in self.nodes_support[s]} for s in self.Nodes
        }
        self.link_to_nodes = {
            l: {self.nodes_dict[t] for t in self.links_support[l]} for l in self.Links
        }

    @property
    def obj(self) -> Hypergraph:
        """Invoque the underlying hypergraph, which is itself."""
        return self  # The underlying representing object is itself

    @property
    def name(self) -> str:
        """Invoque the name of the hypergraph."""
        return self._name

    @property
    def Nodes(self):
        """Invoque the nodes of the hypergraph."""
        return self._Nodes

    @property
    def Links(self):
        """Invoque the links of the hypergraph."""
        return self._Links

    @property
    def Ties(self):
        """Invoque the ties of the hypergraph."""
        return self._Ties

    @property
    def node_map(self):
        """Invoque the node_map of the hypergraph."""
        return self._node_map

    @property
    def link_map(self):
        """Invoque the link_map of the hypergraph."""
        return self._link_map

    def dictionnaire(self) -> dict:
        """Return a dictionary coding the relation T subs S x L, of the form {t : (s, l)}."""
        dic = dict()
        Tsorted = sorted(self.Ties, key=lambda t: repr(t))
        for t in Tsorted:
            dic[t] = (self.node_map(t), self.link_map(t))
        return dic

    def sizes(self) -> tuple[int, int, int]:
        """Return the sizes of the sets of nodes, links and ties as a tuple."""
        return (len(self.Nodes), len(self.Links), len(self.Ties))

    def __repr__(self) -> str:
        """Return a string representation of the hypergraph."""
        presentation_tie = str()
        presentation_nodes = str()
        presentation_links = str()

        safe_key = lambda x: (type(x).__name__, repr(x))

        S = sorted(self.Nodes, key=safe_key)
        L = sorted(self.Links, key=safe_key)
        T = sorted(self.Ties, key=safe_key)

        for s in S:
            presentation_nodes = presentation_nodes + f" \t {s} \n"
        for l in L:
            presentation_links = presentation_links + f" \t {l} \n"
        for t in T:
            presentation_tie = (
                presentation_tie
                + f" \t {t} : {self.nodes_dict[t]} ∈ {self.links_dict[t]} \n"
            )

        return (
            f"Hypergraph {self.name}\n"
            f"  Nodes ({len(S)}): \n {presentation_nodes}"
            f"  Links ({len(L)}): \n {presentation_links}"
            f"  Ties ({len(T)}): \n {presentation_tie}"
        )

    def display(self) -> str:
        """
        Return a string representation of the hypergraph.
        
        Example of display:
            Hypergraph HX
                Nodes (2): 
                        x0 
                        x1 
                Links (1): 
                        lx 
                Ties (2): 
                        t0 : x0 ∈ lx 
                        t1 : x1 ∈ lx """
        return self.__repr__()

    def __eq__(self, other) -> bool:
        """Equality based on the underlying sets and maps,
        independently of the name."""
        return (
            self.Nodes == other.Nodes
            and self.Links == other.Links
            and self.Ties == other.Ties
            and self.node_map == other.node_map
            and self.link_map == other.link_map
        )

    def __hash__(self) -> int:
        """Hash based on the underlying sets and maps,
        independently of the name."""
        return hash(
            (
                hash(self.Nodes),
                hash(self.Links),
                hash(self.Ties),
                hash(self.node_map),
                hash(self.link_map),
            )
        )

    def __str__(self) -> str:
        """Display the name"""
        return self.name

    def rename_global(self, new_name: str) -> Hypergraph:
        """Return a new hypergraph with the same underlying sets and maps,
        but a new name."""
        return Hypergraph(
            Nodes=self.Nodes,
            Ties=self.Ties,
            Links=self.Links,
            node_map=self.node_map,
            link_map=self.link_map,
            name=new_name,
        )

    # les fonctions suivantes requièrent que S, L, T soient des ensembles nommés

    def support_ties(self, l) -> NamedSet:
        """
        For a link l, return the support 

            |l| = {t in T | link_map(t) = l}
        """
        return NamedSet(self.links_support[l], name=f"|{l}|")

    def support_nodes(self, l) -> NamedSet:
        """
        For a link l, return the support 

            ||l|| = {s in S | s_map(t) = s for some t in |l|}
        """
        return NamedSet(self.link_to_nodes[l], name=f"||{l}||")

    def valence_len(self, s, l) -> int:
        """For a node s and a link l, return the valence of s in l,
        that is the number of ties t such that node_map(t) = s and link_map(t) = l
        """
        return len(self.valences[(s, l)])

    def occurrences_ties(self, s) -> NamedSet:
        """
        For a node s, return the set 

            Occ(s) = {t in T | node_map(t) = s}
        """
        return NamedSet(self.nodes_support[s], name=f"Occ({s})")

    def occurrences_links(self, s) -> NamedSet:
        """
        For a node s, return the set 

            Occ_L(s) = {l in L | s is in the support of l}
        """
        return NamedSet(self.node_to_links[s], name=f"Occ_L({s})")

    def valence_set(self, s, l) -> NamedSet:
        """
        For a node s and a link l, return the set

            Val(s, l) = {t in T | node_map(t) = s and link_map(t) = l}
        """
        return NamedSet(self.valences[(s, l)], name=f"<{s} | {l}>")

    def test_simplicité(self) -> bool:
        """Return True if the hypergraph is simple,
        that is, if for every node s and every link l,
        there is at most one tie t
        such that node_map(t) = s and link_map(t) = l."""
        return all(len(self.valences[(s, l)]) <= 1 for (s, l) in self.valences.keys())

    def bipartite(self) -> np.ndarray:
        """Return the bipartite adjacency matrix of the hypergraph,
        with rows indexed by nodes and columns indexed by links,
        and entries given by the valence of the node in the link."""
        return np.array(
            [[self.valence_len(s, l) for l in self.Links] for s in self.Nodes]
        )
    
    def emptylinks(self) -> set:
        """Return the set of empty links, that is, the set of links l such that |l| = ∅."""
        return {l for l in self.Links if len(self.support_ties(l)) == 0}

    def nakednodes(self) -> set:
        """Return the set of naked nodes, that is, the set of nodes s such that Occ(s) = ∅."""
        return {s for s in self.Nodes if len(self.occurrences_ties(s)) == 0}
    
    def intersections(self) -> seth.Pullback:
        """Return the pullback of the node_map and the link_map,
        which is the set of pairs (t0, t1) such that lambda(t0) = lambda(t1). 
        This set can be seen as the set of intersections between ties"""
        return seth.Pullback(self.link_map, self.link_map)
    
    def incidences(self) -> seth.Pullback:
        """Return the pullback of the node_map and the link_map,
        which is the set of pairs (t0, t1) such that node_map(t0) = node_map(t1). 
        This set can be seen as the set of incidences between ties"""
        return seth.Pullback(self.node_map, self.node_map)

    def intersection_nodes(self, l1, l2) -> NamedSet:
        """
        For two links l1 and l2, return the set 

             ||l1|| ∩ ||l2|| = {s in S | s is in the support of l1 and s is in the support of l2}
        """
        return NamedSet(
            self.link_to_nodes[l1].intersection(self.link_to_nodes[l2]),
            name=f"||{l1}|| ∩ ||{l2}||",
        )
    
    def cooccurences_links(self, s1, s2) -> NamedSet:
        """
        For two nodes s1 and s2, return the set 

            Occ_L(s1) ∩ Occ_L(s2) = {l in L | s1 is in the support of l and s2 is in the support of l}
        """
        return NamedSet(
            self.node_to_links[s1].intersection(self.node_to_links[s2]),
            name=f"Occ_L({s1}) ∩ Occ_L({s2})",
        )

    ## conversion entre présentations nommées et brutes

    def hypergraph_to_mutable(self) -> MutableHypergraph:
        """Return a MutableHypergraph with the same underlying sets and maps,
        but with mutable sets."""
        X = set(self.Nodes)
        L = set(self.Links)
        dictbrut = self.dictionnaire()
        return MutableHypergraph(
            Nodes=X,
            Links=L,
            Data=dictbrut,
            name=self.name + "_brut" if self.name else None,
        )


# présentations


def hypergraph_from_dict_brut(D: dict, S0=None, L0=None, name=None) -> Hypergraph:
    """
    Return a Hypergraph from a dictionary specifying ties in an ergonomic way.
    Here the specified nodes and links are native sets, do not require names.
    The values of the dictionary are pairs of the form (s, l) where s is a node and l is a link.
    The values s and l need not be already in S0 nor L0, they will be added to the support of the hypergraph.
    
    Args:
        D: A dictionary coding the span T ↣ S x L, of the form {t : (s, l)}.
        S0: An optional set of nodes to add to the support of the hypergraph.
        L0: An optional set of links to add to the support of the hypergraph.
        name: An optional name for the hypergraph.
    """
    T = NamedSet(set(D.keys()), name=f"T_{name}" if name else "T")
    S = set()
    L = set()
    for v in D.values():
        S.add(v[0])
        L.add(v[1])
    for s in S0 or []:
        S.add(s)
    for l in L0 or []:
        L.add(l)
    S1 = NamedSet(S, name=f"S_{name}" if name else "S")
    L1 = NamedSet(L, name=f"L_{name}" if name else "L")
    s_map = NamedFunction(
        dom=T,
        cod=S1,
        table=lambda t: D[t][0],
        name=f"s_map({name})" if name else "s_map",
    )
    l_map = NamedFunction(
        dom=T,
        cod=L1,
        table=lambda t: D[t][1],
        name=f"l_map({name})" if name else "l_map",
    )
    return Hypergraph(
        Nodes=S1, Ties=T, Links=L1, node_map=s_map, link_map=l_map, name=name
    )


def hypergraph_from_dict(
    dic: dict, Nodes: seth.Representable, Links: seth.Representable, name=None
) -> Hypergraph:
    """
    Return a Hypergraph from a dictionary. 
    Here the values of the dictionary are elements of the given sets of nodes and links.

    Args:
        dic: A dictionary coding the span T ↣ S x L, of the form {t : (s, l)}.
        Nodes: A NamedSet (or more generally, a seth.Representable) of nodes.
        Links: A NamedSet (or more generally, a seth.Representable) of links.
        name: An optional name for the hypergraph.
    """
    T = NamedSet(set(dic.keys()), f"T_{name}" if name else "T")
    s_map = NamedFunction(
        dom=T,
        cod=Nodes,
        table=lambda t: dic[t][0],
        name=f"s_map_{name}" if name else "s_map",
    )
    l_map = NamedFunction(
        dom=T,
        cod=Links,
        table=lambda t: dic[t][1],
        name=f"l_map_{name}" if name else "l_map",
    )
    return Hypergraph(
        Nodes=Nodes, Ties=T, Links=Links, node_map=s_map, link_map=l_map, name=name
    )


def hypergraph_from_set(set_input: set | seth.Representable, name: str) -> Hypergraph:
    """Return a codiscrete Hypergraph from a set of nodes,
    with a single link and ties corresponding to the nodes."""
    if not isinstance(set_input, (set, seth.Representable)):
        raise TypeError("set_input doit être un set ou un seth.Representable")
    S = NamedSet(elements=set_input, name="S") if type(set_input) == set else set_input
    L = NamedSet(elements={f"l_{name}"}, name="L")
    T = NamedSet(elements={("t", element) for element in S}, name="T")
    s_map = NamedFunction(
        dom=T, cod=S, table=lambda t: t[1], name=f"s_map_{name}" if name else "s_map"
    )
    l_map = NamedFunction(
        dom=T,
        cod=L,
        table=lambda t: f"l_{name}",
        name=f"l_map_{name}" if name else "l_map",
    )
    return Hypergraph(
        Nodes=S, Ties=T, Links=L, node_map=s_map, link_map=l_map, name=name
    )


## présentation brut ; les ensembles sont mutables et sans name


class MutableHypergraph:
    """
    Class of mutable hypergraphs,
    whose objects of Nodes and Links are mutable native sets
    and ties are coded by a dictionary of the form {t : (s, l)}.
    
    Elements can be added or removed from the sets of nodes and links, and from the dictionary of ties.

    Args:
        Nodes: A set of nodes.
        Links: A set of links.
        Data: A dictionary coding the span T ↣ S x L, of the form {t : (s, l)}.
        name: An optional name for the hypergraph.
        
    """

    def __init__(self, Nodes: set, Links: set, Data: dict, name=None):
        self.Nodes = Nodes  # set ou list
        self.Links = Links  # set ou list
        self.Data = Data  # dictionnaire de la forme {t : (s, l)} codant la relation fonctionnelle T subs S \times L
        self.Ties = set(self.Data.keys())
        self.name = name

    def repr_sorted(self) -> MutableHypergraph:
        X = sorted(self.Nodes)
        L = sorted(self.Links)
        D = dict()
        for t in sorted(self.Data.keys()):
            D[t] = self.Data[t]
        return MutableHypergraph(
            Nodes=X, Links=L, Data=D, name=self.name + "_sorted" if self.name else None
        )

    def mutable_to_hypergraph(self) -> Hypergraph:
        """Return a Hypergraph with the same underlying sets and maps,
        but with named sets and named functions."""
        S = NamedSet(set(self.Nodes), name=f"S({self.name})" if self.name else "S")
        L = NamedSet(set(self.Links), name=f"L({self.name})" if self.name else "L")
        T = NamedSet(
            set(self.Data.keys()), name=f"T({self.name})" if self.name else "T"
        )
        s_map = NamedFunction(
            dom=T,
            cod=S,
            table=lambda t: self.Data[t][0],
            name=f"s_map({self.name})" if self.name else "s_map",
        )
        l_map = NamedFunction(
            dom=T,
            cod=L,
            table=lambda t: self.Data[t][1],
            name=f"l_map({self.name})" if self.name else "l_map",
        )
        return Hypergraph(
            Nodes=S, Ties=T, Links=L, node_map=s_map, link_map=l_map, name=self.name
        )

    def sizes(self) -> tuple[int, int, int]:
        return (len(self.Nodes), len(self.Links), len(self.Ties))

    def __repr__(self) -> str:
        presentation_tie = str()
        presentation_nodes = str()
        presentation_links = str()

        safe_key = lambda x: (type(x).__name__, repr(x))

        S = sorted(self.Nodes, key=safe_key)
        L = sorted(self.Links, key=safe_key)
        T = sorted(self.Data.keys(), key=safe_key)

        for s in S:
            presentation_nodes = presentation_nodes + f" \t {s} \n"
        for l in L:
            presentation_links = presentation_links + f" \t {l} \n"
        for t in T:
            presentation_tie = (
                presentation_tie + f" \t {t} : {self.Data[t][0]} ∈ {self.Data[t][1]} \n"
            )

        return (
            f"Hypergraph {self.name}\n"
            f"  Nodes ({len(self.Nodes)}): \n {presentation_nodes}"
            f"  Liens ({len(self.Links)}): \n {presentation_links}"
            f"  Témoins ({len(self.Ties)}): \n {presentation_tie}"
        )

    def add(self, S: set = None, L: set = None, T: dict = None) -> "MutableHypergraph":
        """Add nodes, links and ties to the hypergraph,
        where S is a set of nodes, L is a set of links
        and T is a dictionary coding the ties to add."""
        if S is not None:
            self.Nodes = self.Nodes | S
        if L is not None:
            self.Links = self.Links | L
        if T is not None:
            self.Data = self.Data | T
            self.Ties = set(self.Data.keys())
            self.Nodes = self.Nodes | set(v[0] for v in T.values())
            self.Links = self.Links | set(v[1] for v in T.values())
        return self

    def remove(
        self, S: set = None, L: set = None, T: dict = None
    ) -> "MutableHypergraph":
        """Remove nodes, links and ties from the hypergraph,
        where S is a set of nodes, L is a set of links
        and T is a dictionary coding the ties to remove."""
        if S is not None:
            self.Nodes = self.Nodes - S
            self.Data = {t: v for t, v in self.Data.items() if v[0] not in S}
        if L is not None:
            self.Links = self.Links - L
            self.Data = {t: v for t, v in self.Data.items() if v[1] not in L}
        if T is not None:
            for t in T:
                if t in self.Data:
                    del self.Data[t]
            self.Ties = set(self.Data.keys())
        return self

    def rename_elements(
        self, S_map: dict = None, L_map: dict = None, T_map: dict = None
    ) -> "MutableHypergraph":
        """Rename nodes, links and ties of the hypergraph,
        with dictionaries for renaming separately node, links and ties."""
        if S_map is not None:
            self.Nodes = {S_map.get(s, s) for s in self.Nodes}
            for t in list(self.Data.keys()):
                s, l = self.Data[t]
                self.Data[t] = (S_map.get(s, s), l)
        if L_map is not None:
            self.Links = {L_map.get(l, l) for l in self.Links}
            for t in list(self.Data.keys()):
                s, l = self.Data[t]
                self.Data[t] = (s, L_map.get(l, l))
        if T_map is not None:
            self.Data = {T_map.get(t, t): v for t, v in self.Data.items()}
            self.Ties = set(self.Data.keys())
        return self

    def increment_node(self) -> "MutableHypergraph":
        """Add a new node to the hypergraph, with a new name based on the current number of nodes."""
        self.Nodes = self.Nodes | {f"x_{len(self.Nodes)}"}
        return self

    def increment_link(self) -> "MutableHypergraph":
        """Add a new link to the hypergraph, with a new name based on the current number of links."""
        self.Links = self.Links | {f"l_{len(self.Links)}"}
        return self

    def put_node_in_link(self, s, l, n: int = 1) -> "MutableHypergraph":
        """Add n new ties between a node s and a link l,
        with new names based on the current number of ties."""
        for i in range(n):
            t = f"t_{len(self.Ties)}"
            self.Data[t] = (s, l)
            self.Ties.add(t)
        return self

    def rename_sorted(self) -> "MutableHypergraph":
        """Rename nodes, links and ties of the hypergraph with sorted names."""
        S_map = {x: i for i, x in enumerate(sorted(self.Nodes))}
        self.Nodes = {S_map[s] for s in self.Nodes}
        L_map = {l: i for i, l in enumerate(sorted(self.Links))}
        self.Links = {L_map[l] for l in self.Links}
        T_map = {
            t: (t, (S_map[self.Data[t][0]], L_map[self.Data[t][1]]))
            for t in self.Data.keys()
        }
        Données2 = {T_map[t]: (S_map[v[0]], L_map[v[1]]) for t, v in self.Data.items()}
        self.Data = Données2
        return self

    def copy(self) -> "MutableHypergraph":
        """Return a copy of the hypergraph,
        with the same underlying sets and maps,"""
        return MutableHypergraph(
            Nodes=set(self.Nodes),
            Links=set(self.Links),
            Data=dict(self.Data),
            name=self.name + "_copy" if self.name else None,
        )


###########################################

## -- Terminal object --


class Terminal(Construct):
    """
    Terminal object in the category of hypergraphs,
    which is the hypergraph with a single node, a single link and a single tie between them.
        
        S1 = T1 = L1 = {∗}
    """

    def __init__(self):
        terminal_set = seth.Terminal()
        self._name = "1"
        self._obj = Hypergraph(
            Nodes=terminal_set,
            Ties=terminal_set,
            Links=terminal_set,
            node_map=terminal_set.identity,
            link_map=terminal_set.identity,
            name=self._name,
        )

    def unique_map(self, H: Representable) -> HypergraphMorphism:
        """
        Return the unique morphism from H to the terminal object

            !_H : H -> 1
        """
        t = HypergraphMorphism(
            dom=H,
            cod=self,
            map=(
                seth.Terminal().unique_map(H.Nodes),
                seth.Terminal().unique_map(H.Ties),
                seth.Terminal().unique_map(H.Links),
            ),
            name=f"unique_1_to_{H.name}",
        )
        if self.test_terminal(H):
            return HypergraphIsomorphism.from_morphism(t)
        else:
            return t

    @classmethod
    def test_terminal(cls, H) -> bool:
        """Test whether H is terminal, that is, whether it has a single node, a single link and a single tie between them."""
        return len(H.Nodes) == 1 and len(H.Links) == 1 and len(H.Ties) == 1


## -- Initial object --


class Initial(Construct):
    """
    Initial object in the category of hypergraphs,
    which is the hypergraph with no nodes, no links and no ties.

        S0 = T0 = L0 = ∅
    """

    def __init__(self):
        initial_set = seth.Initial()
        self._name = "∅"
        self._obj = Hypergraph(
            Nodes=initial_set,
            Ties=initial_set,
            Links=initial_set,
            node_map=initial_set.identity,
            link_map=initial_set.identity,
            name=self._name,
        )

    def unique_map(self, H: Representable) -> HypergraphMorphism:
        """
        Return the unique morphism from the initial object to H

            !_H : ∅ -> H
         
        """
        i = HypergraphMorphism(
            dom=self,
            cod=H,
            map=(
                seth.Initial().unique_map(H.Nodes),
                seth.Initial().unique_map(H.Ties),
                seth.Initial().unique_map(H.Links),
            ),
            name=f"unique_0_to_{H.name}",
        )
        if self.test_initial(H):
            return HypergraphIsomorphism.from_morphism(i)
        else:
            return i

    @classmethod
    def test_initial(cls, H) -> bool:
        """Test whether H is initial, that is, whether it has no nodes, no links and no ties."""
        return len(H.Nodes) == 0 and len(H.Links) == 0 and len(H.Ties) == 0


## -- Funny Unit --


class Unit_funny(Construct):
    """
    Unit of the funny and strong tensor products,
    which is the hypergraph with a single node and no links

        - S1 = 1, T1 = ∅, 
        - L1 = ∅
    """

    def __init__(self):
        terminal_set = seth.Terminal()
        initial_set = seth.Initial()
        self._name = "I"
        self._obj = Hypergraph(
            Nodes=terminal_set,
            Ties=initial_set,
            Links=initial_set,
            node_map=seth.Initial().unique_map(terminal_set),
            link_map=initial_set.identity,
            name=self._name,
        )


class Unit_Straight(Construct):
    """
    Unit of the straight tensor product,
    which is the hypergraph with a single link and no nodes

        - S1 = ∅, 
        - T1 = ∅, L1 = 1
    """

    def __init__(self):
        terminal_set = seth.Terminal()
        initial_set = seth.Initial()
        self._name = "I*"
        self._obj = Hypergraph(
            Nodes=initial_set,
            Ties=initial_set,
            Links=terminal_set,
            node_map=initial_set.identity,
            link_map=seth.Initial().unique_map(terminal_set),
            name=self._name,
        )


##########################################################################################################

# Morphismes d'hypergraphes


class HypergraphMorphism:
    """Class representing morphisms of hypergraphs as triples of functions
    with a name and a custom display method.

    Args:
        dom : the domain hypergraph
        cod : the codomain hypergraph
        node_map : a named function mapping nodes of the domain to nodes of the codomain
        tie_map : a named function mapping ties of the domain to ties of the codomain
        link_map : a named function mapping links of the domain to links of the codomain
        name : the name of the morphism

    Enter the triple of functions as a tuple (node_map, tie_map, link_map) in the parameter 'map' of the constructor.
    """

    def __init__(
        self,
        dom: Representable,
        cod: Representable,
        map: tuple[NamedFunction, NamedFunction, NamedFunction],
        name=None,
    ):

        if not (
            isinstance(map, tuple)
            and len(map) == 3
            and all(isinstance(m, NamedFunction) for m in map)
        ):
            raise TypeError(
                "'map must be a tuple of three NamedFunction instances (node_map, tie_map, link_map)'."
            )

        if dom.Nodes != map[0].dom or cod.Nodes != map[0].cod:
            raise ValueError(
                "The node_map must be a function from dom.Nodes to cod.Nodes."
            )
        if dom.Ties != map[1].dom or cod.Ties != map[1].cod:
            raise ValueError(
                "The tie_map must be a function from dom.Ties to cod.Ties."
            )
        if dom.Links != map[2].dom or cod.Links != map[2].cod:
            raise ValueError(
                "The link_map must be a function from dom.Links to cod.Links."
            )

        # the triple of functions (node_map, tie_map, link_map)
        self.dom = dom
        self.cod = cod
        self.node_map = map[0]
        self.tie_map = map[1]
        self.link_map = map[2]
        self.name = name
        if not self.test_morphisme_formel():
            raise ValueError("The morphism is not a hypergraph morphism.")

    def test_morphisme_formel(self) -> bool:
        """Test whether the data of the morphism satisfies the morphism condition, that is,
        as a formal commutativity condition on the functions"""
        if not seth.composition(self.dom.node_map, self.node_map) == seth.composition(
            self.tie_map, self.cod.node_map
        ):
            return False
        if not seth.composition(self.dom.link_map, self.link_map) == seth.composition(
            self.tie_map, self.cod.link_map
        ):
            return False
        return True

    # test concret ;
    # a doubler d'un test formel via les eq de fonctio name composées
    def test_morphisme_concret(self) -> bool:
        """Test whether the triple of functions (node_map, tie_map, link_map)
        satisfies the morphism condition, that is,
        whether for every tie t in dom.Ties

            - node_map(dom.node_map(t)) = cod.node_map(tie_map(t))
            - link_map(dom.link_map(t)) = cod.link_map(tie_map(t))
            
        """
        dom_node_of_tie = self.dom.obj.nodes_dict
        dom_link_of_tie = self.dom.obj.links_dict
        cod_node_of_tie = self.cod.obj.nodes_dict
        cod_link_of_tie = self.cod.obj.links_dict

        node_image = self.node_map.values
        tie_image = self.tie_map.values
        link_image = self.link_map.values

        for t in self.dom.Ties:
            t_img = tie_image[t]

            if node_image[dom_node_of_tie[t]] != cod_node_of_tie[t_img]:
                return False
            if link_image[dom_link_of_tie[t]] != cod_link_of_tie[t_img]:
                return False

        return True

    # ici dom et cod servent à differentier des methodes dom, cod de la classe NamedFunction

    def symbolic_repr(self) -> str:
        """Return a symbolic representation of the function, e.g., 

            f : X → Y 
        """
        return f"{self.name} : {self.dom.name} → {self.cod.name} "

    def __repr__(self) -> str:
        return (
            self.symbolic_repr() + "\n"
            f"  S_{self.name} = {repr(self.node_map)} \n"
            f"  T_{self.name} = {repr(self.tie_map)} \n"
            f"  L_{self.name} = {repr(self.link_map)} \n"
        )

    def __str__(self) -> str:
        return self.symbolic_repr()

    def display(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        """Equality based on domaim, codomain,
        and the underlying functions, independently of the name."""
        return (
            self.dom == other.dom
            and self.cod == other.cod
            and self.node_map == other.node_map
            and self.tie_map == other.tie_map
            and self.link_map == other.link_map
        )

    def __lt__(self, other):
        if self.dom != other.dom or self.cod != other.cod:
            raise ValueError("Les morphismes ne sont pas comparables.")
        return (self.node_map, self.tie_map, self.link_map) < (
            other.node_map,
            other.tie_map,
            other.link_map,
        )

    def __hash__(self) -> int:
        return hash(
            (
                hash(self.dom),
                hash(self.cod),
                hash(self.node_map),
                hash(self.tie_map),
                hash(self.link_map),
            )
        )

    def eval_support(self, l) -> NamedFunction:
        """For a link l in dom, return the support map of l,
        that is the function from the support of l in dom
        to the support of link_map(l) in cod,
        given by the restriction of node_map to the support of l."""
        return NamedFunction(
            dom=self.dom.obj.support_ties(l),
            cod=self.cod.obj.support_ties(self.link_map(l)),
            table=lambda t: self.tie_map(t),
            name=f"support_map_{self.name}" if self.name else "support_map",
        )

    def eval_occurences(self, s) -> NamedFunction:
        """For a node s in dom, return the occurrences map of s,
        that is the function from the occurrences of s in dom
        to the occurrences of node_map(s) in cod,
        given by the restriction of tie_map to the occurrences of s."""
        return NamedFunction(
            dom=self.dom.obj.occurrences_ties(s),
            cod=self.cod.obj.occurrences_ties(self.node_map(s)),
            table=lambda t: self.tie_map(t),
            name=f"occurrences_map_{self.name}" if self.name else "occurrences_map",
        )

    def test_mono(self) -> bool:
        """A monomorphisme in the category of hypergraphs
        is a morphism of hypergraphs whose component functions are injective."""
        return (
            self.node_map.injectivity_test_fast()
            and self.link_map.injectivity_test_fast()
            and self.tie_map.injectivity_test_fast()
        )

    def test_epi(self) -> bool:
        """An epimorphisme in the category of hypergraphs
        is a morphism of hypergraphs whose component functions are surjective."""
        return (
            self.node_map.surjectivity_test_fast()
            and self.link_map.surjectivity_test_fast()
            and self.tie_map.surjectivity_test_fast()
        )

    def test_iso(self) -> bool:
        """An isomorphisme in the category of hypergraphs
        is a morphism of hypergraphs whose component functions are bijective."""
        return (
            self.node_map.bijectivity_test_fast()
            and self.link_map.bijectivity_test_fast()
            and self.tie_map.bijectivity_test_fast()
        )

    def image(self) -> Hypergraph:
        """Pointwise calculation of the image of the morphism, that is the hypergraph 
        whose sets of nodes, links and ties are the images of the corresponding functions,"""
        return Hypergraph(
            Nodes=self.node_map.image(),
            Ties=self.tie_map.image(),
            Links=self.link_map.image(),
            node_map=NamedFunction(
                dom=self.tie_map.image(),
                cod=self.node_map.image(),
                table=lambda t: self.cod.node_map(t),
                name=f"s_image_{self.name}" if self.name else "s_image",
            ),
            link_map=NamedFunction(
                dom=self.tie_map.image(),
                cod=self.link_map.image(),
                table=lambda t: self.cod.link_map(t),
                name=f"l_image_{self.name}" if self.name else "l_image",
            ),
            name=f"Im({self.name})" if self.name else "Im",
        )


def composition(f1: HypergraphMorphism, f2: HypergraphMorphism) -> HypergraphMorphism:
    """ Return the composition of two morphisms of hypergraphs f1 and f2, that is the morphism of hypergraphs
     f2 ∘ f1 : dom(f1) → cod(f2) given by the composition of the underlying functions. Beware the order."""
    if f1.cod != f2.dom:
        raise ValueError("Morphisms do not compose")
    else:
        node_map = seth.composition(f1.node_map, f2.node_map)
        tie_map = seth.composition(f1.tie_map, f2.tie_map)
        link_map = seth.composition(f1.link_map, f2.link_map)
        return HypergraphMorphism(
            dom=f1.dom,
            cod=f2.cod,
            map=(node_map, tie_map, link_map),
            name=f"{f2.name} ∘ {f1.name}",
        )


# Monomorphismes d'hypergraphes


class HypergraphMonomorphism(HypergraphMorphism):
    """A monomorphism in the category of hypergraphs
    is a morphism of hypergraphs whose component functions are injective."""

    def __init__(self, dom: Representable, cod: Representable, map: tuple, name=None):
        super().__init__(dom=dom, cod=cod, map=map, name=name)
        if not self.test_mono():
            raise ValueError("Le morphisme n'est pas un monomorphisme d'hypergraphes.")

    def symbolic_repr(self) -> str:
        """
        Return a symbolic representation of the function, e.g.,

            f : X ↣ Y
        """
        return f"{self.name} : {self.dom.name} ↣ {self.cod.name} "

    @classmethod
    def from_morphism(cls, f: HypergraphMorphism) -> HypergraphMonomorphism:
        if not f.test_mono():
            raise ValueError("Le morphisme n'est pas un monomorphisme d'hypergraphes.")
        return cls(
            dom=f.dom,
            cod=f.cod,
            map=(f.node_map, f.tie_map, f.link_map),
            name=f.name,
        )


class HypergraphEpimorphism(HypergraphMorphism):
    """An epimorphism in the category of hypergraphs
    is a morphism of hypergraphs whose component functions are surjective."""

    def __init__(self, dom: Representable, cod: Representable, map: tuple, name=None):
        super().__init__(dom=dom, cod=cod, map=map, name=name)
        if not self.test_epi():
            raise ValueError("Le morphisme n'est pas un épimorphisme d'hypergraphes.")

    def symbolic_repr(self) -> str:
        """
        Return a symbolic representation of the function, e.g., 

            f : X ↠ Y 
        """
        return f"{self.name} : {self.dom.name} ↠ {self.cod.name} "

    @classmethod
    def from_morphism(cls, f: HypergraphMorphism) -> HypergraphEpimorphism:
        if not f.test_epi():
            raise ValueError("Le morphisme n'est pas un épimorphisme d'hypergraphes.")
        return cls(
            dom=f.dom,
            cod=f.cod,
            map=(f.node_map, f.tie_map, f.link_map),
            name=f.name,
        )

    def sections(self) -> NamedSet:
        """
        For an epimorphism f : G ↠ H,
        Return the set Γ(f) of sections of the ,
        that is the set of morphisms 

            g : H → G
        
        such that

            f ∘ g = id_H
        """
        sections = set()
        for f in CartesianHomSet(self.cod, self.dom):
            if composition(f, self) == self.cod.identity:
                sections.add(f)
        return NamedSet(sections, name=f"Γ({self.name})")


class HypergraphIsomorphism(HypergraphMorphism):
    """Un isomorphisme d'hypergraphes est un morphisme d'hypergraphes dont les fonctions composantes sont bijectives."""

    def __init__(self, dom: Representable, cod: Representable, map: tuple, name=None):
        super().__init__(dom=dom, cod=cod, map=map, name=name)
        if not self.test_iso():
            raise ValueError("Le morphisme n'est pas un isomorphisme d'hypergraphes.")
        Sf = seth.Bijection.from_function(self.node_map)
        Tf = seth.Bijection.from_function(self.tie_map)
        Lf = seth.Bijection.from_function(self.link_map)
        self.inverse = HypergraphMorphism(
            dom=cod,
            cod=dom,
            map=(Sf.inverse, Tf.inverse, Lf.inverse),
            name=f"{self.name}⁻¹" if self.name else "iso⁻¹",
        )

    def symbolic_repr(self) -> str:
        """Return a symbolic representation of the function, e.g.,

            f : X ≃ Y
        """
        return f"{self.name} : {self.dom.name} ≃ {self.cod.name} "

    @classmethod
    def from_morphism(cls, f: HypergraphMorphism) -> HypergraphIsomorphism:
        if not f.test_iso():
            raise ValueError("Le morphisme n'est pas un isomorphisme d'hypergraphes.")
        return cls(
            dom=f.dom,
            cod=f.cod,
            map=(f.node_map, f.tie_map, f.link_map),
            name=f.name,
        )


#######################################################################

# Ensemble des morphismes d'hypergraphes entre deux hypergraphes


class CartesianHomSet(seth.Construct):
    """Represent the homset in Hyp
    For H0,H1, return Hyp[H0,H1], the set of morphisms of hypergraphs from H0 to H1
    Object calculation is defered to the method generate, 
    which calculates the homset as a subset of the cartesian product of the homsets of the underlying sets.
    """

    def __init__(self, H0: Representable, H1: Representable):
        self.H0 = H0
        self.H1 = H1
        self.hom_S = HomSet(H0.Nodes, H1.Nodes)
        self.hom_T = HomSet(H0.Ties, H1.Ties)
        self.hom_L = HomSet(H0.Links, H1.Links)
        self._name = f"Hom[{H0.name}, {H1.name}]"

    def generate(self):
        """Calculate the homset as a subset of the cartesian product of the homsets of the underlying sets."""
        hom = set()
        for s, t, l in it.product(self.hom_S, self.hom_T, self.hom_L):
            try:
                f = HypergraphMorphism(
                    dom=self.H0, cod=self.H1, map=(s, t, l), name=f"f_{len(hom)}"
                )
            except ValueError:
                continue
            if f.test_morphisme_concret():
                hom.add(f)
        return NamedSet(hom, name=self._name)

    @property
    def obj(self) -> NamedSet:
        """Realizes Hom[H0,H1] as a Homset."""
        return self.generate()

    @property
    def name(self) -> str:
        """Return the name of the homset."""
        return self._name

    def __repr__(self) -> str:
        return self.obj.display()

    def display(self):
        return self.obj.display()

    def __str__(self) -> str:
        return self.name

    def __len__(self) -> int:
        return len(self.obj)

    def __iter__(self):
        return iter(self.obj)


# Ensemble des morphismes monomorphismes d'hypergraphes entre deux hypergraphes


def hom_mono(H_0: Representable, H_1: Representable) -> NamedSet:
    """Return the set of monomorphisms of hypergraphs from H_0 to H_1,
    as a subset of the homset Hom[H_0,H_1]."""
    homono = set()
    for f in CartesianHomSet(H_0, H_1).obj:
        if f.test_mono():
            homono.add(f)
    return NamedSet(homono, name=f"Hom_mono({H_0.name}, {H_1.name})")


def hom_epi(H_0: Representable, H_1: Representable) -> NamedSet:
    """Return the set of epimorphisms of hypergraphs from H_0 to H_1,
    as a subset of the homset Hom[H_0,H_1]."""
    homepi = set()
    for f in CartesianHomSet(H_0, H_1).obj:
        if f.test_epi():
            homepi.add(f)
    return NamedSet(homepi, name=f"Hom_epi({H_0.name}, {H_1.name})")


def Iso(H_0: Representable, H_1: Representable) -> NamedSet:
    """Return the set of isomorphisms of hypergraphs from H_0 to H_1,
    as a subset of the homset Hom[H_0,H_1]."""
    hom_iso = set()
    for f in CartesianHomSet(H_0, H_1).obj:
        if f.test_iso():
            hom_iso.add(f)
    return NamedSet(hom_iso, name=f"Iso({H_0.name}, {H_1.name})")


# Morphismes conformes et coconformes


def test_conforme(f: HypergraphMorphism) -> bool:
    """A morphism f : H → G of hypergraphs is conforme if for every link l in dom,
    the support map |l|_H → |f(l)|_G of l is surjective."""
    for l in f.dom.Links:
        dom_support = f.dom.obj.support_ties(l)
        cod_support = f.cod.obj.support_ties(f.link_map(l))
        support_map = f.eval_support(l)
        if not support_map.surjectivity_test_fast():
            return False
    return True


def hom_conforme(H_0: Representable, H_1: Representable) -> NamedSet:
    """Return the set of conforme morphisms of hypergraphs from H_0 to H_1,
    as a subset of the homset Hom[H_0,H_1]."""
    homconforme = set()
    for f in CartesianHomSet(H_0, H_1).obj:
        if test_conforme(f):
            homconforme.add(f)
    return NamedSet(homconforme, name=f"Hom_conforme({H_0.name}, {H_1.name})")


def test_coconforme(f: HypergraphMorphism) -> bool:
    """A morphism f : H → G of hypergraphs is coconforme if for every node s in dom,
    the occurrences map Occ(s)_H → Occ(f(s))_G of s is surjective."""
    for s in f.dom.Nodes:
        dom_occurences = f.dom.obj.occurrences_ties(s)
        cod_occurences = f.cod.obj.occurrences_ties(f.node_map(s))
        occurences_map = f.eval_occurences(s)
        if not occurences_map.surjectivity_test_fast():
            return False
    return True


def hom_coconforme(H_0: Representable, H_1: Representable) -> NamedSet:
    """Return the set of coconforme morphisms of hypergraphs from H_0 to H_1,
    as a subset of the homset Hom[H_0,H_1]."""
    homcoconforme = set()
    for f in CartesianHomSet(H_0, H_1).obj:
        if test_coconforme(f):
            homcoconforme.add(f)
    return NamedSet(homcoconforme, name=f"Hom_coconforme({H_0.name}, {H_1.name})")


def Restriction_nodes(H: Representable, X: NamedSet, f: NamedFunction) -> Hypergraph:
    pb = seth.Pullback(f, H.node_map)
    T_restr = pb
    L_restr = NamedSet(set(H.link_map(t[1]) for t in T_restr), name=f"L_restr_{H.name}")
    s_restr = pb.proj_0()
    l_restr = NamedFunction(
        dom=T_restr,
        cod=L_restr,
        table=lambda t: H.link_map(t[1]),
        name=f"l_restr_{H.name}",
    )
    return Hypergraph(
        Nodes=X,
        Ties=T_restr,
        Links=L_restr,
        node_map=s_restr,
        link_map=l_restr,
        name=f"Restriction_nodes({H.name}, {X.name})",
    )


############################################################################

# Limites d'hypergraphes

"""Hyp is a presheaf topos over the walking cospan,
hence it has all limits and colimits, 
and they are calculated pointwise on the underlying sets of nodes, links and ties,"""


class Product(Construct):
    """
    Return H0 × H1, the product of H0 and H1 in the category of hypergraphs,

    It is defined as:

        - S_{H0 × H1} = S_{H0} × S_{H1}
        - T_{H0 × H1} = T_{H0} × T_{H1}
        - L_{H0 × H1} = L_{H0} × L_{H1}
        - σ_{H0 × H1} = σ_{H0} × σ_{H1}
        - λ_{H0 × H1} = λ_{H0} × λ_{H1}
    """

    def __init__(self, H0: Representable, H1: Representable):
        self.H0 = H0
        self.H1 = H1

        if isinstance(H0, Binary_Type) and isinstance(H1, Binary_Type):
            self._name = f"({H0.name}) × ({H1.name})"
        elif isinstance(H0, Binary_Type) and not isinstance(H1, Binary_Type):
            self._name = f"({H0.name}) × {H1.name}"
        elif not isinstance(H0, Binary_Type) and isinstance(H1, Binary_Type):
            self._name = f"{H0.name} × ({H1.name})"
        else:
            self._name = f"{H0.name} × {H1.name}"

        self._obj = Hypergraph(
            Nodes=seth.Product(H0.Nodes, H1.Nodes),
            Ties=seth.Product(H0.Ties, H1.Ties),
            Links=seth.Product(H0.Links, H1.Links),
            node_map=seth.product_maps(H0.node_map, H1.node_map),
            link_map=seth.product_maps(H0.link_map, H1.link_map),
            name=self._name,
        )

    def proj_0(self) -> HypergraphMorphism:
        """
        Return the projection

            π0 : H0 × H1 ↠ H0
        """
        node_map = (
            self.Nodes.proj_0()
            if hasattr(self.Nodes, "proj_0")
            else seth.Surjection(
                dom=self.Nodes,
                cod=self.H0.Nodes,
                table=lambda s: s[0],
                name=f"π_0_{self.H0.name}",
            )
        )
        tie_map = (
            self.Ties.proj_0()
            if hasattr(self.Ties, "proj_0")
            else seth.Surjection(
                dom=self.Ties,
                cod=self.H0.Ties,
                table=lambda t: t[0],
                name=f"π_0_{self.H0.name}",
            )
        )
        link_map = (
            self.Links.proj_0()
            if hasattr(self.Links, "proj_0")
            else seth.Surjection(
                dom=self.Links,
                cod=self.H0.Links,
                table=lambda l: l[0],
                name=f"π_0_{self.H0.name}",
            )
        )
        return HypergraphEpimorphism(
            dom=self,
            cod=self.H0,
            map=(node_map, tie_map, link_map),
            name=f"π_0_{self.H0.name},{self.H1.name}",
        )

    def proj_1(self) -> HypergraphMorphism:
        """
        Return the projection 

            π1 : H0 × H1 ↠ H1
        """
        node_map = (
            self.Nodes.proj_1()
            if hasattr(self.Nodes, "proj_1")
            else seth.Surjection(
                dom=self.Nodes,
                cod=self.H1.Nodes,
                table=lambda s: s[1],
                name=f"π_1_{self.H1.name}",
            )
        )
        tie_map = (
            self.Ties.proj_1()
            if hasattr(self.Ties, "proj_1")
            else seth.Surjection(
                dom=self.Ties,
                cod=self.H1.Ties,
                table=lambda t: t[1],
                name=f"π_1_{self.H1.name}",
            )
        )
        link_map = (
            self.Links.proj_1()
            if hasattr(self.Links, "proj_1")
            else seth.Surjection(
                dom=self.Links,
                cod=self.H1.Links,
                table=lambda l: l[1],
                name=f"π_1_{self.H1.name}",
            )
        )
        return HypergraphEpimorphism(
            dom=self,
            cod=self.H1,
            map=(node_map, tie_map, link_map),
            name=f"π_1_{self.H0.name},{self.H1.name}",
        )

    def universal_solution(
        self, f1: HypergraphMorphism, f2: HypergraphMorphism
    ) -> HypergraphMorphism:
        """
        For a cone f1 : C → H0, f2 : C → H1, return the unique morphism 

            (f1,f2) : C → H0 × H1
        such that 

            - π_0 ∘ (f1,f2) = f1
            - π_1 ∘ (f1,f2) = f2

        It is computed from the universal solutions
        for the underlying sets of nodes, links and ties."""
        if f1.cod != self.H0 or f2.cod != self.H1:
            raise ValueError("Do not form a cone over the product.")
        else:
            node_map = self.Nodes.universal_solution(f0=f1.node_map, f1=f2.node_map)
            tie_map = self.Ties.universal_solution(f0=f1.tie_map, f1=f2.tie_map)
            link_map = self.Links.universal_solution(f0=f1.link_map, f1=f2.link_map)
        return HypergraphMorphism(
            dom=f1.dom,
            cod=self,
            map=(node_map, tie_map, link_map),
            name=f"({f1.name}, {f2.name})",
        )

    def braiding(self) -> HypergraphIsomorphism:
        """
        Return the braiding isomorphism 

            β : H0 × H1 ≃ H1 × H0
        """
        braid = Product(self.H1, self.H0)
        node_map = self.Nodes.braiding()
        tie_map = self.Ties.braiding()
        link_map = self.Links.braiding()
        return HypergraphIsomorphism(
            dom=self,
            cod=braid,
            map=(node_map, tie_map, link_map),
            name=f"β_{self.H0.name}_{self.H1.name}",
        )


def unitor_cartesian_left(H: Representable) -> HypergraphIsomorphism:
    """
    Return the left unitor isomorphism

        λ : 1 × H ≃ H
        """
    unitor = Product(Terminal(), H)
    node_map = seth.unitor_cartesian_left(H.Nodes)
    tie_map = seth.unitor_cartesian_left(H.Ties)
    link_map = seth.unitor_cartesian_left(H.Links)
    return HypergraphIsomorphism(
        dom=unitor, cod=H, map=(node_map, tie_map, link_map), name=f"λ_{H.name}"
    )


def unitor_cartesian_right(H: Representable) -> HypergraphIsomorphism:
    """
    Return the right unitor isomorphism

        ρ : H × 1 ≃ H
    """
    unitor = Product(H, Terminal())
    node_map = seth.unitor_cartesian_right(H.Nodes)
    tie_map = seth.unitor_cartesian_right(H.Ties)
    link_map = seth.unitor_cartesian_right(H.Links)
    return HypergraphIsomorphism(
        dom=unitor, cod=H, map=(node_map, tie_map, link_map), name=f"ρ_{H.name}"
    )


def associator_cartesian(
    H0: Representable, H1: Representable, H2: Representable
) -> HypergraphIsomorphism:
    """
    Return the associator isomorphism 

        α : (H0 × H1) × H2 ≃ H0 × (H1 × H2)
    """
    leftprod = Product(Product(H0, H1), H2)
    rightprod = Product(H0, Product(H1, H2))
    node_map = seth.associator_cartesian(H0.Nodes, H1.Nodes, H2.Nodes)
    tie_map = seth.associator_cartesian(H0.Ties, H1.Ties, H2.Ties)
    link_map = seth.associator_cartesian(H0.Links, H1.Links, H2.Links)
    return HypergraphIsomorphism(
        dom=leftprod,
        cod=rightprod,
        map=(node_map, tie_map, link_map),
        name=f"α_{H0.name}_{H1.name}_{H2.name}",
    )


def product_morphism(
    f0: HypergraphMorphism, f1: HypergraphMorphism
) -> HypergraphMorphism:
    """
    For two morphisms f0 : H0 → G0 and f1 : H1 → G1, 
    return the product morphism 

        f0 × f1 : H0 × H1 → G0 × G1
        """
    product_dom = Product(f0.dom, f1.dom)
    product_cod = Product(f0.cod, f1.cod)
    node_map = seth.product_maps(f0.node_map, f1.node_map)
    tie_map = seth.product_maps(f0.tie_map, f1.tie_map)
    link_map = seth.product_maps(f0.link_map, f1.link_map)
    return HypergraphMorphism(
        dom=product_dom,
        cod=product_cod,
        map=(node_map, tie_map, link_map),
        name=f"{f0.name} × {f1.name}",
    )


## -- Finite Cartesian Product --


class FiniteProduct(Construct):
    def __init__(self, hypergraph_list: list[Hypergraph]):
        """
        Return the finite product of a list of hypergraphs H0, H1, ..., Hn, defined as:

            - S_{H0 × H1 × ... × Hn} = S_{H0} × S_{H1} × ... × S_{Hn}
            - T_{H0 × H1 × ... × Hn} = T_{H0} × T_{H1} × ... × T_{Hn}
            - L_{H0 × H1 × ... × Hn} = L_{H0} × L_{H1} × ... × L_{Hn}
            - σ_{H0 × H1 × ... × Hn} = σ_{H0} × σ_{H1} × ... × σ_{Hn}
            - λ_{H0 × H1 × ... × Hn} = λ_{H0} × λ_{H1} × ... × λ_{Hn}
        """
        self.hypergraph_list = hypergraph_list
        self._name = " × ".join(f"{H.name}" for H in hypergraph_list)
        if not hypergraph_list:
            self._obj = Terminal()
        else:
            self._obj = Hypergraph(
                Nodes=seth.FiniteProduct([H.Nodes for H in hypergraph_list]),
                Ties=seth.FiniteProduct([H.Ties for H in hypergraph_list]),
                Links=seth.FiniteProduct([H.Links for H in hypergraph_list]),
                node_map=seth.finite_product_maps(
                    [H.node_map for H in hypergraph_list]
                ),
                link_map=seth.finite_product_maps(
                    [H.link_map for H in hypergraph_list]
                ),
                name=self._name,
            )

    def proj(self, index: int) -> HypergraphMorphism:
        """
        Return the projection morphism 

            π_i : H0 × H1 × ... × Hn → Hi
        """
        if index < 0 or index >= len(self.hypergraph_list):
            raise IndexError("Index de projection hors limites.")
        H = self.hypergraph_list[index]
        node_map = (
            self.obj.Nodes.proj(index)
            if hasattr(self.Nodes, "proj")
            else seth.Surjection(
                dom=self.Nodes,
                cod=H.Nodes,
                table=lambda s: s[index],
                name=f"S_π_{index}",
            )
        )
        tie_map = (
            self.obj.Ties.proj(index)
            if hasattr(self.Ties, "proj")
            else seth.Surjection(
                dom=self.Ties,
                cod=H.Ties,
                table=lambda t: t[index],
                name=f"T_π_{index}",
            )
        )
        link_map = (
            self.obj.Links.proj(index)
            if hasattr(self.Links, "proj")
            else seth.Surjection(
                dom=self.Links,
                cod=H.Links,
                table=lambda l: l[index],
                name=f"L_π_{index}",
            )
        )
        return HypergraphEpimorphism(
            dom=self.obj, cod=H, map=(node_map, tie_map, link_map), name=f"π_{index}"
        )

    def universal_solution(
        self, morphism_list: list[HypergraphMorphism]
    ) -> HypergraphMorphism:
        """
        For a cone f_i : C → Hi,
        return the unique morphism 

            (f_i) : C → H0 × H1 × ... × Hn

        such that for all i

            π_i ∘ (f_i) = f_i 

        It is computed from the universal solutions
        at the underlying sets of nodes, links and ties."""
        if len(morphism_list) != len(self.hypergraph_list):
            raise ValueError(
                "Number doesn't match the number of factors in the product."
            )
        for i, f in enumerate(morphism_list):
            if f.cod != self.hypergraph_list[i]:
                raise ValueError(f"Do not form a cone over the product.")

        node_map = self.Nodes.universal_solution(
            list_f=[f.node_map for f in morphism_list],
        )
        tie_map = self.Ties.universal_solution(
            list_f=[f.tie_map for f in morphism_list],
        )
        link_map = self.Links.universal_solution(
            list_f=[f.link_map for f in morphism_list],
        )
        return HypergraphMorphism(
            dom=morphism_list[0].dom,
            cod=self,
            map=(node_map, tie_map, link_map),
            name=f"({', '.join(f.name for f in morphism_list)})",
        )


def product_morphism_list(
    morphism_list: list[HypergraphMorphism],
) -> HypergraphMorphism:
    """
    From a list of morphisms f0, f1, ..., fn, return the product morphism 

        Π fi : H0 × H1 × ... × Hn → G0 × G1 × ... × Gn
    """
    product_dom = FiniteProduct([f.dom for f in morphism_list])
    product_cod = FiniteProduct([f.cod for f in morphism_list])
    node_map = seth.finite_product_maps([f.node_map for f in morphism_list])
    tie_map = seth.finite_product_maps([f.tie_map for f in morphism_list])
    link_map = seth.finite_product_maps([f.link_map for f in morphism_list])
    return HypergraphMorphism(
        dom=product_dom,
        cod=product_cod,
        map=(node_map, tie_map, link_map),
        name=f" × ".join(f"{f.name}" for f in morphism_list),
    )


## -- Pullback --


class Pullback(Construct):
    """
    Return the pullback of two morphisms f0 : H0 → H2 and f1 : H1 → H2
    H0 ×_{H2} H1, which is computed pointwise as
      
        - S_{H0 ×_{H2} H1} = S_{H0} ×_{S_{H2}} S_{H1}
        - T_{H0 ×_{H2} H1} = T_{H0} ×_{T_{H2}} T_{H1}
        - L_{H0 ×_{H2} H1} = L_{H0} ×_{L_{H2}} L_{H1}
    """

    def __init__(self, f0: HypergraphMorphism, f1: HypergraphMorphism):
        self.f0 = f0
        self.f1 = f1
        self.H0 = f0.dom
        self.H1 = f1.dom
        self.H2 = f0.cod if f0.cod == f1.cod else None

        same_target = (f0.cod is f1.cod) or (
            f0.cod.Nodes == f1.cod.Nodes
            and f0.cod.Links == f1.cod.Links
            and f0.cod.Ties == f1.cod.Ties
        )
        if not same_target:
            raise ValueError("Les hypergraphes cods des morphismes ne sont pas égaux.")

        self.S_pb = seth.Pullback(f0.node_map, f1.node_map)
        self.L_pb = seth.Pullback(f0.link_map, f1.link_map)
        self.T_pb = seth.Pullback(f0.tie_map, f1.tie_map)

        self._name = (
            f"{f0.dom.name} ×_{f0.cod.name}^({f0.name}, {f1.name}) {f1.dom.name}"
        )
        self._obj = Hypergraph(
            Nodes=self.S_pb,
            Ties=self.T_pb,
            Links=self.L_pb,
            node_map=NamedFunction(
                dom=self.T_pb,
                cod=self.S_pb,
                table=lambda t: (self.H0.node_map(t[0]), self.H1.node_map(t[1])),
                name=f"s_{f0.name}*{f1.name}",
            ),
            link_map=NamedFunction(
                dom=self.T_pb,
                cod=self.L_pb,
                table=lambda t: (self.H0.link_map(t[0]), self.H1.link_map(t[1])),
                name=f"l_{f0.name}*{f1.name}",
            ),
            name=self._name,
        )

    def proj_0(self) -> HypergraphMorphism:
        """
        Return the projection 

            π0 : H0 ×_{H2} H1 → H0
        """
        return HypergraphMorphism(
            dom=self,
            cod=self.H0,
            map=(
                (
                    self.S_pb.proj_0()
                    if hasattr(self.Nodes, "proj_0")
                    else NamedFunction(
                        dom=self.Nodes,
                        cod=self.H0.Nodes,
                        table=lambda s: s[0],
                        name=f"π_S_{self.H0.name}",
                    )
                ),
                (
                    self.T_pb.proj_0()
                    if hasattr(self.Ties, "proj_0")
                    else NamedFunction(
                        dom=self.Ties,
                        cod=self.H0.Ties,
                        table=lambda t: t[0],
                        name=f"π_T_{self.H0.name}",
                    )
                ),
                (
                    self.L_pb.proj_0()
                    if hasattr(self.Links, "proj_0")
                    else NamedFunction(
                        dom=self.Links,
                        cod=self.H0.Links,
                        table=lambda l: l[0],
                        name=f"π_L_{self.H0.name}",
                    )
                ),
            ),
            name=f"π0_{self.f0.name},{self.f1.name}",
        )

    def proj_1(self) -> HypergraphMorphism:
        """
        Return the projection

            π1 : H0 ×_{H2} H1 → H1
        """
        return HypergraphMorphism(
            dom=self,
            cod=self.H1,
            map=(
                (
                    self.S_pb.proj_1()
                    if hasattr(self.Nodes, "proj_1")
                    else NamedFunction(
                        dom=self.Nodes,
                        cod=self.H1.Nodes,
                        table=lambda s: s[1],
                        name=f"π_S_{self.H1.name}",
                    )
                ),
                (
                    self.T_pb.proj_1()
                    if hasattr(self.Ties, "proj_1")
                    else NamedFunction(
                        dom=self.Ties,
                        cod=self.H1.Ties,
                        table=lambda t: t[1],
                        name=f"π_T_{self.H1.name}",
                    )
                ),
                (
                    self.L_pb.proj_1()
                    if hasattr(self.Links, "proj_1")
                    else NamedFunction(
                        dom=self.Links,
                        cod=self.H1.Links,
                        table=lambda l: l[1],
                        name=f"π_L_{self.H1.name}",
                    )
                ),
            ),
            name=f"π1_{self.f0.name},{self.f1.name}",
        )

    def universal_solution(
        self, g0: HypergraphMorphism, g1: HypergraphMorphism
    ) -> HypergraphMorphism:
        """
        For a cone 
        g0 : C → H0, g1 : C → H1, 
        return the unique morphism 

            ⟨g0,g1⟩ : C → H0 ×_{H2} H1

        such that 

           -  π0 ∘ ⟨g0,g1⟩ = g0
           -  π1 ∘ ⟨g0,g1⟩ = g1
        """
        if g0.cod != self.H0 or g1.cod != self.H1:
            raise ValueError("Morphismes must have the right codomains")
        if g0.dom != g1.dom:
            raise ValueError("Morphismes must have the same domain.")
        if composition(g0, self.f0) != composition(g1, self.f1):
            raise ValueError("Morphisms must compose")

        node_map_pb = self.Nodes.universal_solution(g0.node_map, g1.node_map)
        tie_map_pb = self.Ties.universal_solution(g0.tie_map, g1.tie_map)
        link_map_pb = self.Links.universal_solution(g0.link_map, g1.link_map)

        unique_morphism = HypergraphMorphism(
            dom=g0.dom,
            cod=self,
            map=(node_map_pb, tie_map_pb, link_map_pb),
            name=f"({self.f0.name},{self.f1.name})",
        )
        return unique_morphism


def kernel(f: HypergraphMorphism) -> Pullback:
    return Pullback(f, f)


## -- Equalizers --


class Equalizer(Construct):
    """Return the equalizer of two morphisms f0 : H0 → H1 and f1 : H0 → H1"""
    def __init__(self, f0: HypergraphMorphism, f1: HypergraphMorphism):
        if f0.dom != f1.dom:
            raise ValueError(f"Domains do not match.")
        if f0.cod != f1.cod:
            raise ValueError(f"Codomains do not match.")
        self.f0 = f0
        self.f1 = f1
        self.H0 = f0.dom
        self.H1 = f0.cod

        self.S_eq = seth.Equalizer(f0.node_map, f1.node_map)
        self.T_eq = seth.Equalizer(f0.tie_map, f1.tie_map)
        self.L_eq = seth.Equalizer(f0.link_map, f1.link_map)

        self._name = f"Eq({f0.name}, {f1.name})"
        self._obj = Hypergraph(
            Nodes=self.S_eq,
            Ties=self.T_eq,
            Links=self.L_eq,
            node_map=self.S_eq.universal_solution(
                h=seth.composition(self.T_eq.inclusion(), self.H0.node_map)
            ),
            link_map=self.L_eq.universal_solution(
                h=seth.composition(self.T_eq.inclusion(), self.H0.link_map)
            ),
            name=self._name,
        )

    def inclusion(self) -> HypergraphMonomorphism:
        """
        Return the inclusion morphism

            i : Eq(f0,f1) ↣ H0
            """
        return HypergraphMonomorphism(
            dom=self,
            cod=self.H0,
            map=(
                self.S_eq.inclusion(),
                self.T_eq.inclusion(),
                self.L_eq.inclusion(),
            ),
            name=f"inclusion_{self.name}",
        )

    def universal_solution(self, g: HypergraphMorphism) -> HypergraphMorphism:
        """
        For a map g : C → H0 such that f0 ∘ g = f1 ∘ g, return the unique morphism 
        
            ⟨g⟩ : C → Eq(f0,f1)

        such that 

            i ∘ ⟨g⟩ = g
        """
        if g.cod != self.H0:
            raise ValueError("Morphisme must have the right codomain.")
        if not self.test_equalize(g):
            raise ValueError("Morphisme does not equalize f0 and f1.")

        node_map_eq = self.S_eq.universal_solution(g.node_map)
        tie_map_eq = self.T_eq.universal_solution(g.tie_map)
        link_map_eq = self.L_eq.universal_solution(g.link_map)

        unique_morphism = HypergraphMorphism(
            dom=g.dom,
            cod=self,
            map=(node_map_eq, tie_map_eq, link_map_eq),
            name=f"⟨{g.name}⟩",
        )
        return unique_morphism

    def test_equalize(self, g: HypergraphMorphism) -> bool:
        """Test if a morphism g : C → H0 equalizes f0 and f1, i.e. if f0 ∘ g = f1 ∘ g."""
        if g.cod != self.H0:
            raise ValueError("Morphisme must have the right codomain.")
        return composition(g, self.f0) == composition(g, self.f1)


###################################################################

# Colimits of hypergraphs

## -- Coproducts --


class Coproduct(Construct):
    """
    Return the coproduct of two hypergraphs H0 and H1:

        - S_{H0 + H1} = S_{H0} + S_{H1}
        - T_{H0 + H1} = T_{H0} + T_{H1}
        - L_{H0 + H1} = L_{H0} + L_{H1}
        - σ_{H0 + H1} = σ_{H0} + σ_{H1}
        - λ_{H0 + H1} = λ_{H0} + λ_{H1}
    """

    def __init__(self, H0: Representable, H1: Representable):
        self.H0 = H0
        self.H1 = H1

        # Insert parenthesis around binary types for better readability of names
        if isinstance(H0, Binary_Type) and isinstance(H1, Binary_Type):
            self._name = f"({H0.name}) + ({H1.name})"
        elif isinstance(H0, Binary_Type) and not isinstance(H1, Binary_Type):
            self._name = f"({H0.name}) + {H1.name}"
        elif not isinstance(H0, Binary_Type) and isinstance(H1, Binary_Type):
            self._name = f"{H0.name} + ({H1.name})"
        else:
            self._name = f"{H0.name} + {H1.name}"

        self._obj = Hypergraph(
            Nodes=seth.Coproduct(H0.Nodes, H1.Nodes),
            Ties=seth.Coproduct(H0.Ties, H1.Ties),
            Links=seth.Coproduct(H0.Links, H1.Links),
            node_map=seth.coproduct_maps(H0.node_map, H1.node_map),
            link_map=seth.coproduct_maps(H0.link_map, H1.link_map),
            name=self._name,
        )

    def inj_0(self) -> HypergraphMonomorphism:
        """
        Return the injection 
        
            in0 : H0 ↣ H0 + H1
        """
        node_map = self.Nodes.inj_0()
        tie_map = self.Ties.inj_0()
        link_map = self.Links.inj_0()
        return HypergraphMonomorphism(
            dom=self.H0,
            cod=self,
            map=(node_map, tie_map, link_map),
            name=f"in0_({self.H0.name},{self.H1.name})",
        )

    def inj_1(self) -> HypergraphMonomorphism:
        """
        Return the injection 
        
            in1 : H1 ↣ H0 + H1
        """
        node_map = self.Nodes.inj_1()
        tie_map = self.Ties.inj_1()
        link_map = self.Links.inj_1()
        return HypergraphMonomorphism(
            dom=self.H1,
            cod=self,
            map=(node_map, tie_map, link_map),
            name=f"in1_({self.H0.name},{self.H1.name})",
        )

    def universal_solution(
        self, f1: HypergraphMorphism, f2: HypergraphMorphism
    ) -> HypergraphMorphism:
        """
        For a cocone f1 : H0 → C, f2 : H1 → C,
        return the unique morphism 
        
            ⟨f1,f2⟩ : H0 + H1 → C


        such that 
        
            - ⟨f1,f2⟩ ∘ inj_0 = f1 
            - ⟨f1,f2⟩ ∘ inj_1 = f2."""
        if f1.dom != self.H0 or f2.dom != self.H1 or f1.cod != f2.cod:
            raise ValueError("Les morphismes ne sont pas adaptés au coproduct.")
        else:
            node_map = self.Nodes.universal_solution(f0=f1.node_map, f1=f2.node_map)
            tie_map = self.Ties.universal_solution(f0=f1.tie_map, f1=f2.tie_map)
            link_map = self.Links.universal_solution(f0=f1.link_map, f1=f2.link_map)
        return HypergraphMorphism(
            dom=self,
            cod=f1.cod,
            map=(node_map, tie_map, link_map),
            name=f"⟨{f1.name}, {f2.name}⟩",
        )
    
    def braiding(self) -> HypergraphIsomorphism:
        """
        Return the braiding isomorphism 

            β : H0 + H1 ≃ H1 + H0
        """
        braid = Coproduct(self.H1, self.H0)
        node_map = self.Nodes.braiding()
        tie_map = self.Ties.braiding()
        link_map = self.Links.braiding()
        return HypergraphIsomorphism(
            dom=self,
            cod=braid,
            map=(node_map, tie_map, link_map),
            name=f"β_{self.H0.name}_{self.H1.name}",
        )

def unitor_coproduct_left(H: Representable) -> HypergraphIsomorphism:
    """
    Return the left unitor isomorphism

        λ : 0 + H ≃ H
    """
    unitor = Coproduct(Initial(), H)
    node_map = seth.unitor_coproduct_left(H.Nodes)
    tie_map = seth.unitor_coproduct_left(H.Ties)
    link_map = seth.unitor_coproduct_left(H.Links)
    return HypergraphIsomorphism(
        dom=unitor, 
        cod=H, 
        map=(node_map, tie_map, link_map), 
        name=f"λ_{H.name}"
    )

def unitor_coproduct_right(H: Representable) -> HypergraphIsomorphism:
    """
    Return the right unitor isomorphism

        ρ : H + 0 ≃ H
    """
    unitor = Coproduct(H, Initial())
    node_map = seth.unitor_coproduct_right(H.Nodes)
    tie_map = seth.unitor_coproduct_right(H.Ties)
    link_map = seth.unitor_coproduct_right(H.Links)
    return HypergraphIsomorphism(
        dom=unitor, 
        cod=H, 
        map=(node_map, tie_map, link_map), 
        name=f"ρ_{H.name}"
    )

def associator_coproduct(
    H0: Representable, H1: Representable, H2: Representable
    ) -> HypergraphIsomorphism:
    """Return the associator isomorphism

        α : (H0 + H1) + H2 ≃ H0 + (H1 + H2)
    """
    leftcoprod = Coproduct(Coproduct(H0, H1), H2)
    rightcoprod = Coproduct(H0, Coproduct(H1, H2))
    node_map = seth.associator_coproduct(H0.Nodes, H1.Nodes, H2.Nodes)
    tie_map = seth.associator_coproduct(H0.Ties, H1.Ties, H2.Ties)
    link_map = seth.associator_coproduct(H0.Links, H1.Links, H2.Links)
    return HypergraphIsomorphism(
        dom=leftcoprod,
        cod=rightcoprod,
        map=(node_map, tie_map, link_map),
        name=f"α_{H0.name}_{H1.name}_{H2.name}",
    ) 
    


def coproduct_maps(
    f1: HypergraphMorphism, f2: HypergraphMorphism
) -> HypergraphMorphism:
    """
    Return the coproduct morphism

        f1 + f2 : H0 + H1 → G0 + G1
    """
    node_map = seth.coproduct_maps(f1.node_map, f2.node_map)
    tie_map = seth.coproduct_maps(f1.tie_map, f2.tie_map)
    link_map = seth.coproduct_maps(f1.link_map, f2.link_map)
    return HypergraphMorphism(
        dom=Coproduct(f1.dom, f2.dom),
        cod=Coproduct(f1.cod, f2.cod),
        map=(node_map, tie_map, link_map),
        name=f"coprod_maps_{f1.name}_{f2.name}",
    )


## -- Finite Coproducts --


class FiniteCoproduct(Construct):
    """
    Return the finite coproduct H0 + H1 + ... + Hn of a list of hypergraphs

        - S_{H0 + H1 + ... + Hn} = S_{H0} + S_{H1} + ... + S_{Hn}
        - T_{H0 + H1 + ... + Hn} = T_{H0} + T_{H1} + ... + T_{Hn}
        - L_{H0 + H1 + ... + Hn} = L_{H0} + L_{H1} + ... + L_{Hn}
        - σ_{H0 + H1 + ... + Hn} = σ_{H0} + σ_{H1} + ... + σ_{Hn}
        - λ_{H0 + H1 + ... + Hn} = λ_{H0} + λ_{H1} + ... + λ_{Hn}
    """

    def __init__(self, hypergraph_list: list[Representable]):

        self.hypergraph_list = hypergraph_list
        self._name = " + ".join(f"{H.name}" for H in hypergraph_list)

        if not hypergraph_list:
            self._obj = Initial()
        else:
            self._obj = Hypergraph(
                Nodes=seth.FiniteCoproduct([H.Nodes for H in hypergraph_list]),
                Ties=seth.FiniteCoproduct([H.Ties for H in hypergraph_list]),
                Links=seth.FiniteCoproduct([H.Links for H in hypergraph_list]),
                node_map=seth.finite_coproduct_maps(
                    [H.node_map for H in hypergraph_list]
                ),
                link_map=seth.finite_coproduct_maps(
                    [H.link_map for H in hypergraph_list]
                ),
                name=self._name,
            )

    def inj(self, index: int) -> HypergraphMorphism:
        """
        Return the injection

            in_i : Hi ↣ H0 + H1 + ... + Hn
        """
        if index < 0 or index >= len(self.hypergraph_list):
            raise IndexError("Index d'injection hors limites.")
        H = self.hypergraph_list[index]
        node_map = self.Nodes.inj(index)
        tie_map = self.Ties.inj(index)
        link_map = self.Links.inj(index)
        return HypergraphMorphism(
            dom=H, cod=self, map=(node_map, tie_map, link_map), name=f"in_{H.name}"
        )

    def universal_solution(
        self, morphism_list: list[HypergraphMorphism]
    ) -> HypergraphMorphism:
        """
        For a cocone f_i : Hi → C,
        return the unique morphism 

            ⟨f_i⟩ : H0 + H1 + ... + Hn → C

        such that for all i
            ⟨f_i⟩ ∘ inj_i = f_i 
        """
        if len(morphism_list) != len(self.hypergraph_list):
            raise ValueError(
                "Number doesn't match the number of factors in the coproduct."
            )
        for i, f in enumerate(morphism_list):
            if f.dom != self.hypergraph_list[i]:
                raise ValueError(
                    f"The morphism {f.name} does not have the correct domain hypergraph."
                )

        node_map = self.Nodes.universal_solution(
            list_f=[f.node_map for f in morphism_list],
        )
        tie_map = self.Ties.universal_solution(
            list_f=[f.tie_map for f in morphism_list],
        )
        link_map = self.Links.universal_solution(
            list_f=[f.link_map for f in morphism_list],
        )
        return HypergraphMorphism(
            dom=FiniteCoproduct([f.dom for f in morphism_list]),
            cod=morphism_list[0].cod,
            map=(node_map, tie_map, link_map),
            name=f"⟨{', '.join(f.name for f in morphism_list)}⟩",
        )


## -- Pushout --


class Pushout(Construct):
    def __init__(self, f0: HypergraphMorphism, f1: HypergraphMorphism):
        """Return the pushout of two morphisms f0 : H0 → H2 and f1 : H1 → H2"""
        if f0.dom != f1.dom:
            raise ValueError("Les morphismes n'ont pas le même domaine.")
        self.f0 = f0
        self.f1 = f1
        self.H0 = f0.cod
        self.H1 = f1.cod
        self.H2 = f0.dom

        self.S_po = seth.Pushout(f0.node_map, f1.node_map)
        self.L_po = seth.Pushout(f0.link_map, f1.link_map)
        self.T_po = seth.Pushout(f0.tie_map, f1.tie_map)

        self._name = f"{f0.cod.name} +^{f0.name},{f1.name}_{f0.dom.name} {f1.cod.name}"
        self._obj = Hypergraph(
            Nodes=self.S_po,
            Ties=self.T_po,
            Links=self.L_po,
            node_map=self.T_po.universal_solution(
                f=seth.composition(self.H0.node_map, self.S_po.inj_0()),
                g=seth.composition(self.H1.node_map, self.S_po.inj_1()),
            ),
            link_map=self.T_po.universal_solution(
                f=seth.composition(self.H0.link_map, self.L_po.inj_0()),
                g=seth.composition(self.H1.link_map, self.L_po.inj_1()),
            ),
            name=self._name,
        )

    def inj_0(self) -> HypergraphMorphism:
        """
        Return the injection 

            in0 : H0 → H0 +^f0,f1_H2 H1
        """
        return HypergraphMorphism(
            dom=self.H0,
            cod=self,
            map=(
                self.S_po.inj_0(),
                self.T_po.inj_0(),
                self.L_po.inj_0(),
            ),
            name=f"in0_({self.H0.name},{self.H1.name})",
        )

    def inj_1(self) -> HypergraphMorphism:
        """Return the injection

            in1 : H1 → H0 +^f0,f1_H2 H1
        """
        return HypergraphMorphism(
            dom=self.H1,
            cod=self,
            map=(
                self.S_po.inj_1(),
                self.T_po.inj_1(),
                self.L_po.inj_1(),
            ),
            name=f"in1_({self.H0.name},{self.H1.name})",
        )

    def universal_solution(
        self, g0: HypergraphMorphism, g1: HypergraphMorphism
    ) -> HypergraphMorphism:
        """For a cocone g0 : H0 → C, g1 : H1 → C such that g0 ∘ f0 = g1 ∘ f1,
        return the unique morphism 

             ⟨g0,g1⟩ : H0 +^f0,f1_H2 H1 → C

        such that 

             ⟨g0,g1⟩ ∘ inj_0 = g0 and ⟨g0,g1⟩ ∘ inj_1 = g1."""
        if g0.dom != self.H0:
            raise ValueError(f"{g0.name} does not have the correct domain hypergraph.")
        if g1.dom != self.H1:
            raise ValueError(f"{g1.name} does not have the correct domain hypergraph.")
        if g0.cod != g1.cod:
            raise ValueError(f"{g0.name} and {g1.name} do not have the same codomain.")
        if composition(self.f0, g0) != composition(self.f1, g1):
            raise ValueError("Morphisms do not form a cone over the pushout.")

        node_map = self.Nodes.universal_solution(f=g0.node_map, g=g1.node_map)
        tie_map = self.Ties.universal_solution(f=g0.tie_map, g=g1.tie_map)
        link_map = self.Links.universal_solution(f=g0.link_map, g=g1.link_map)
        return HypergraphMorphism(
            dom=self,
            cod=g0.cod,
            map=(node_map, tie_map, link_map),
            name=f"⟨{g0.name}, {g1.name}⟩",
        )


## -- Coequalizers --


class Coequalizer(Construct):
    def __init__(self, f0: HypergraphMorphism, f1: HypergraphMorphism):
        """Return the coequalizer of two morphisms f0 : H0 → H1 and f1 : H0 → H1"""
        if f0.dom != f1.dom:
            raise ValueError("Domains do not match.")
        if f0.cod != f1.cod:
            raise ValueError("Codomains do not match.")
        self.f0 = f0
        self.f1 = f1
        self.H0 = f0.dom
        self.H1 = f0.cod

        self.S_coeq = seth.Coequalizer(f0.node_map, f1.node_map)
        self.L_coeq = seth.Coequalizer(f0.link_map, f1.link_map)
        self.T_coeq = seth.Coequalizer(f0.tie_map, f1.tie_map)

        self._name = f"Coeq({f0.name},{f1.name})"
        self._obj = Hypergraph(
            Nodes=self.S_coeq,
            Ties=self.T_coeq,
            Links=self.L_coeq,
            node_map=self.T_coeq.universal_solution(
                seth.composition(self.H1.node_map, self.S_coeq.projection())
            ),
            link_map=self.T_coeq.universal_solution(
                seth.composition(self.H1.link_map, self.L_coeq.projection())
            ),
            name=self._name,
        )

    def proj(self) -> HypergraphEpimorphism:
        """Return the projection 
         π : H1 → H1 /~_f0,f1 H0"""
        return HypergraphEpimorphism(
            dom=self.H1,
            cod=self,
            map=(
                self.S_coeq.projection(),
                self.T_coeq.projection(),
                self.L_coeq.projection(),
            ),
            name=f"π_({self.H1.name}/~_({self.f0.name},{self.f1.name}))",
        )

    def test_coequalize(self, g: HypergraphMorphism) -> bool:
        """Test if a morphism g : H1 → C coequalizes f0 and f1, i.e. if g ∘ f0 = g ∘ f1."""
        if g.dom != self.H1:
            raise ValueError("Morphisme must have the right domain.")
        return composition(self.f0, g) == composition(self.f1, g)

    def universal_solution(self, g: HypergraphMorphism) -> HypergraphMorphism:
        """For a map g : H1 → C such that g ∘ f0 = g ∘ f1, return the unique morphism 

             ⟨g⟩ : Coeq(f0,f1) → C

        such that 

            ⟨g⟩ ∘ π = g."""
        if g.dom != self.H1:
            raise ValueError("Morphisme must have the right domain.")
        if not self.test_coequalize(g):
            raise ValueError("Morphisme does not coequalize f0 and f1.")

        node_map_coeq = self.S_coeq.universal_solution(g.node_map)
        tie_map_coeq = self.T_coeq.universal_solution(g.tie_map)
        link_map_coeq = self.L_coeq.universal_solution(g.link_map)

        unique_morphism = HypergraphMorphism(
            dom=self,
            cod=g.cod,
            map=(node_map_coeq, tie_map_coeq, link_map_coeq),
            name=f"⟨{g.name}⟩",
        )
        return unique_morphism


##############################################################################################

# Le topos des hypergraphes


def Omega_hyp() -> Hypergraph:
    """Return the subobject classifier Omega in the category of hypergraphs.

        - S_Ω = {True,False} (the subobject classifier of Set)
        - L_Ω = {True,False} (the subobject classifier of Set)
        - T_Ω = {t_∅, t_sigma, t_lambda, t_sigma_lambda, t_top}

    the 5 possible configurations for a tie t : x ∈ l :

    - t_∅ : neither t, x nor l are in the subobject
    - t_sigma : only x is in the subobject
    - t_lambda : only l is in the subobject
    - t_sigma_lambda : both x and l are in the subobject, but not t
    - t_top : t, x and l are all in the subobject
    """
    return hypergraph_from_dict(
        dic={
            "t_∅": (False, False),
            "t_sigma": (True, False),
            "t_lambda": (False, True),
            "t_sigma_lambda": (True, True),
            "t_top": (True, True),
        },
        Nodes=seth.Omega_set(),
        Links=seth.Omega_set(),
        name="Ω",
    )


def Top_hyp() -> HypergraphMorphism:
    """Return the morphism ⊤ : 1 → Ω,
    where 1 is the terminal hypergraph and Ω is the subobject classifier."""
    return HypergraphMorphism(
        dom=Terminal(),
        cod=Omega_hyp(),
        map=(
            NamedFunction(
                dom=Terminal().Nodes,
                cod=Omega_hyp().Nodes,
                table=lambda s: True,
                name="S_⊤",
            ),
            NamedFunction(
                dom=Terminal().Ties,
                cod=Omega_hyp().Ties,
                table=lambda t: "t_top",
                name="T_⊤",
            ),
            NamedFunction(
                dom=Terminal().Links,
                cod=Omega_hyp().Links,
                table=lambda l: True,
                name="L_⊤",
            ),
        ),
        name="⊤",
    )


def subobject_from_charmap(H: Representable, chi: HypergraphMorphism) -> Hypergraph:
    if chi.cod != Omega_hyp():
        raise ValueError("Le morphisme caractéristique ne cod pas l'hypergraphe Omega.")
    else:
        Sub = Pullback(chi, Top_hyp())
        S = NamedSet(set(s for (s, _) in Sub.obj.Nodes), name=f"S({chi.name})")
        L = NamedSet(set(l for (l, _) in Sub.obj.Links), name=f"L({chi.name})")
        T = NamedSet(set(t for (t, _) in Sub.obj.Ties), name=f"T({chi.name})")
        s_map = NamedFunction(
            dom=T,
            cod=S,
            table=lambda t: chi.dom.node_map(t),
            name=f"s_S({chi.name})",
        )
        l_map = NamedFunction(
            dom=T,
            cod=L,
            table=lambda t: chi.dom.link_map(t),
            name=f"l_S({chi.name})",
        )
        return Hypergraph(
            Nodes=S,
            Ties=T,
            Links=L,
            node_map=s_map,
            link_map=l_map,
            name=f"S({chi.name})",
        )


def test_inclusion_sub_fast(Sub1: Representable, Sub2: Representable) -> int:
    # On suppose Sub1, Sub2 déjà dans Sub(H) : pas de recomputation ici.
    return (
        set(Sub1.Nodes.set).issubset(set(Sub2.Nodes.set))
        and set(Sub1.Links.set).issubset(set(Sub2.Links.set))
        and set(Sub1.Ties.set).issubset(set(Sub2.Ties.set))
    )


def inclusion_hypergraph(
    Subgraph: Representable,
    H: Representable,
) -> HypergraphMorphism:
    """Return the inclusion morphism i : Subgraph ↣ H, where Subgraph is a subobject of H.
    It is computed from the pullback defining Subgraph as a subobject of H."""
    if not test_inclusion_sub_fast(Subgraph, H):
        raise ValueError("Le sous-hypergraphe n'est pas inclus dans l'hypergraphe.")
    else:
        node_map = NamedFunction(
            dom=Subgraph.Nodes,
            cod=H.Nodes,
            table=lambda s: s,
            name=f"S_({Subgraph.name}→{H.name})",
        )
        tie_map = NamedFunction(
            dom=Subgraph.Ties,
            cod=H.Ties,
            table=lambda t: t,
            name=f"T_({Subgraph.name}→{H.name})",
        )
        link_map = NamedFunction(
            dom=Subgraph.Links,
            cod=H.Links,
            table=lambda l: l,
            name=f"L_({Subgraph.name}→{H.name})",
        )
        return HypergraphMorphism(
            dom=Subgraph,
            cod=H,
            map=(node_map, tie_map, link_map),
            name=f"i_({Subgraph.name}→{H.name})",
        )


def poset_of_subobjects(H: Representable) -> NamedSet:
    """Return the set of subobjects of H
    It is computed from the hom-set Hom(H, Ω)
    and the pullback defining each subobject as a subobject of H."""
    hom_to_Omega = CartesianHomSet(H, Omega_hyp())
    subobjects = set()
    for chi in hom_to_Omega.obj:
        sub_H = subobject_from_charmap(H, chi)
        subobjects.add(sub_H)
    return NamedSet(subobjects, name=f"Sub({H.name})")


def subobjet_conforme(H: Representable) -> NamedSet:
    """Return the set of subobjects of H that are conformal,
    i.e. such that the inclusion morphism is a hypergraph monomorphism."""
    subobj_H = poset_of_subobjects(H)
    conforme_subs = set()
    for Subgraph in subobj_H:
        incl = inclusion_hypergraph(H, Subgraph)
        if test_conforme(incl):
            conforme_subs.add(Subgraph)
    return NamedSet(conforme_subs, name=f"Conforme_Sub({H.name})")


###################################################################################################

# Structures monoïdales


class FunnyTensor(Construct):
    """
    Return the funny tensor product of two hypergraphs H0 and H1

        - S_{H0 □ H1} = S_{H0} × S_{H1}
        - T_{H0 □ H1} = (S_{H0} × T_{H1}) + (T_{H0} × S_{H1})
        - L_{H0 □ H1} = (S_{H0} × L_{H1}) + (L_{H0} × S_{H1})

    """

    def __init__(self, H0: Representable, H1: Representable):
        self.H0 = H0
        self.H1 = H1
        self._Nodes = seth.Product(H0.Nodes, H1.Nodes)
        self._Links = seth.Coproduct(
            seth.Product(H0.Nodes, H1.Links), seth.Product(H0.Links, H1.Nodes)
        )
        self._Ties = seth.Coproduct(
            seth.Product(H0.Nodes, H1.Ties), seth.Product(H0.Ties, H1.Nodes)
        )
        sigma0 = seth.product_maps(H0.Nodes.identity, H1.node_map)
        sigma1 = seth.product_maps(H0.node_map, H1.Nodes.identity)
        self._node_map = self._Ties.universal_solution(f0=sigma0, f1=sigma1)
        lambda0 = seth.product_maps(H0.Nodes.identity, H1.link_map)
        lambda1 = seth.product_maps(H0.link_map, H1.Nodes.identity)
        self._link_map = seth.coproduct_maps(lambda0, lambda1)

        if isinstance(self.H0, Binary_Type) and isinstance(self.H1, Binary_Type):
            self._name = f"({H0.name}) □ ({H1.name})"
        elif isinstance(self.H0, Binary_Type) and not isinstance(self.H1, Binary_Type):
            self._name = f"({H0.name}) □ {H1.name}"
        elif not isinstance(self.H0, Binary_Type) and isinstance(self.H1, Binary_Type):
            self._name = f"{H0.name} □ ({H1.name})"
        else:
            self._name = f"{H0.name} □ {H1.name}"
        self._obj = Hypergraph(
            Nodes=self._Nodes,
            Ties=self._Ties,
            Links=self._Links,
            node_map=self._node_map,
            link_map=self._link_map,
            name=self._name,
        )

    def braiding(self) -> HypergraphIsomorphism:
        """Return the braiding isomorphism β : H0 □ H1 → H1 □ H0"""
        braid = FunnyTensor(self.H1, self.H0)
        node_map = self.Nodes.braiding()
        tie_map = seth.composition(
            self.Ties.braiding(),
            seth.coproduct_maps(
                seth.Product(self.H0.Ties, self.H1.Nodes).braiding(),
                seth.Product(self.H0.Nodes, self.H1.Ties).braiding(),
            ),
        )
        link_map = seth.composition(
            self.Links.braiding(),
            seth.coproduct_maps(
                seth.Product(self.H0.Links, self.H1.Nodes).braiding(),
                seth.Product(self.H0.Nodes, self.H1.Links).braiding(),
            ),
        )
        b = HypergraphMorphism(
            dom=self,
            cod=braid,
            map=(node_map, tie_map, link_map),
            name=f"β_{self.H0.name},{self.H1.name}",
        )
        return HypergraphIsomorphism.from_morphism(b)

    def canonical_elements_naming(self) -> Hypergraph:
        """Rename the links and ties of H0 □ H1 with their canonical form, i.e.

        - x ⊗ r : (x,y) ∈ x ⊗ k if x ∈ S_H0 and r : y ∈ k in H1
        - t ⊗ y : (x,y) ∈ t ⊗ k if t : x ∈ k in H0 and y ∈ S_H1

        """
        tie_renaming = lambda t: f"{t[1][0]} ⊗ {t[1][1]}"
        link_renaming = lambda l: f"{l[1][0]} ⊗ {l[1][1]}"
        new_ties = self.Ties.rename_elements(tie_renaming)
        new_links = self.Links.rename_elements(link_renaming)
        tie_renaming_map = seth.Bijection(
            dom=self.Ties,
            cod=new_ties,
            table=tie_renaming,
            name=f"rename_ties_{self.obj.name}",
        )
        link_renaming_map = seth.Bijection(
            dom=self.Links,
            cod=new_links,
            table=link_renaming,
            name=f"rename_links_{self.obj.name}",
        )
        new_node_map = seth.composition(tie_renaming_map.inverse, self.node_map)
        new_link_map = seth.composition_chaine(
            [tie_renaming_map.inverse, self.link_map, link_renaming_map]
        )
        return Hypergraph(
            Nodes=self.Nodes,
            Ties=new_ties,
            Links=new_links,
            node_map=new_node_map,
            link_map=new_link_map,
            name=self.name,
        )

    def canonical_elements_naming_isomorphism(self) -> HypergraphIsomorphism:
        """Rename the links and ties of H0 □ H1 with their canonical form"""
        renamed = self.canonical_elements_naming()
        node_map = self.Nodes.identity
        tie_map = seth.Bijection(
            dom=self.Ties,
            cod=renamed.Ties,
            table=lambda t: f"{t[1][0]} ⊗ {t[1][1]}",
            name=f"rename_ties_{self.name}",
        )
        link_map = seth.Bijection(
            dom=self.Links,
            cod=renamed.Links,
            table=lambda l: f"{l[1][0]} ⊗ {l[1][1]}",
            name=f"rename_links_{self.name}",
        )
        return HypergraphIsomorphism(
            dom=self,
            cod=renamed,
            map=(node_map, tie_map, link_map),
            name=f"canonical_elements_naming_iso_{self.name}",
        )


def funny_left_unitor(H: Representable) -> HypergraphIsomorphism:
    """
    Return the left unitor isomorphism

        λ : I □ H → H

    where I is the unit object for the funny tensor product."""
    left_prod = FunnyTensor(Unit_funny(), H)
    node_map = seth.unitor_cartesian_left(H.Nodes)
    tie_map = seth.composition(
        seth.unitor_coproduct_right(seth.Product(seth.Terminal(), H.Ties)),
        seth.unitor_cartesian_left(H.Ties),
    )
    link_map = seth.composition(
        seth.unitor_coproduct_right(seth.Product(seth.Terminal(), H.Links)),
        seth.unitor_cartesian_left(H.Links),
    )
    l = HypergraphIsomorphism(
        dom=left_prod, cod=H, map=(node_map, tie_map, link_map), name=f"λ_{H.name}"
    )
    return HypergraphIsomorphism.from_morphism(l)


def funny_right_unitor(H: Representable) -> HypergraphIsomorphism:
    """
    Return the right unitor isomorphism 

        ρ : H □ I → H

    where I is the unit object for the funny tensor product."""
    right_prod = FunnyTensor(H, Unit_funny())
    node_map = seth.unitor_cartesian_right(H.Nodes)
    tie_map = seth.composition(
        seth.unitor_coproduct_left(seth.Product(H.Ties, seth.Terminal())),
        seth.unitor_cartesian_right(H.Ties),
    )
    link_map = seth.composition(
        seth.unitor_coproduct_left(seth.Product(H.Links, seth.Terminal())),
        seth.unitor_cartesian_right(H.Links),
    )
    r = HypergraphIsomorphism(
        dom=right_prod, cod=H, map=(node_map, tie_map, link_map), name=f"ρ_{H.name}"
    )
    return HypergraphIsomorphism.from_morphism(r)


def funny_associator(
    H0: Representable, H1: Representable, H2: Representable
) -> HypergraphIsomorphism:
    """
    Return the associator isomorphism

        α : (H0 □ H1) □ H2 → H0 □ (H1 □ H2)
    """
    leftpair = FunnyTensor(H0, H1)
    left_prod = FunnyTensor(leftpair, H2)
    rightpair = FunnyTensor(H1, H2)
    right_prod = FunnyTensor(H0, rightpair)
    node_map = seth.associator_cartesian(H0.Nodes, H1.Nodes, H2.Nodes)
    chaine_tie = [
        seth.coproduct_maps(
            seth.Product(seth.Product(H0.Nodes, H1.Nodes).obj, H2.Ties).identity,
            seth.right_distributivity_isomorphism(
                ext=H2.Nodes,
                paire=(
                    seth.Product(H0.Nodes, H1.Ties),
                    seth.Product(H0.Ties, H1.Nodes),
                ),
            ),
        ),
        seth.coproduct_maps(
            seth.associator_cartesian(H0.Nodes, H1.Nodes, H2.Ties),
            seth.coproduct_maps(
                seth.associator_cartesian(H0.Nodes, H1.Ties, H2.Nodes),
                seth.associator_cartesian(H0.Ties, H1.Nodes, H2.Nodes),
            ),
        ),
        seth.associator_coproduct(
            seth.Product(H0.Nodes, seth.Product(H1.Nodes, H2.Ties)),
            seth.Product(H0.Nodes, seth.Product(H1.Ties, H2.Nodes)),
            seth.Product(H0.Ties, seth.Product(H1.Nodes, H2.Nodes)),
        ).inverse,
        seth.coproduct_maps(
            seth.left_distributivity_isomorphism(
                ext=H0.Nodes,
                paire=(
                    seth.Product(H1.Nodes, H2.Ties),
                    seth.Product(H1.Ties, H2.Nodes),
                ),
            ).inverse,
            seth.Product(H0.Ties, seth.Product(H1.Nodes, H2.Nodes)).identity,
        ),
    ]
    tie_map = seth.composition_chaine(chaine_tie)

    chaine_link = [
        seth.coproduct_maps(
            seth.Product(seth.Product(H0.Nodes, H1.Nodes), H2.Links).identity,
            seth.right_distributivity_isomorphism(
                ext=H2.Nodes,
                paire=(
                    seth.Product(H0.Nodes, H1.Links),
                    seth.Product(H0.Links, H1.Nodes),
                ),
            ),
        ),
        seth.coproduct_maps(
            seth.associator_cartesian(H0.Nodes, H1.Nodes, H2.Links),
            seth.coproduct_maps(
                seth.associator_cartesian(H0.Nodes, H1.Links, H2.Nodes),
                seth.associator_cartesian(H0.Links, H1.Nodes, H2.Nodes),
            ),
        ),
        seth.associator_coproduct(
            seth.Product(H0.Nodes, seth.Product(H1.Nodes, H2.Links)),
            seth.Product(H0.Nodes, seth.Product(H1.Links, H2.Nodes)),
            seth.Product(H0.Links, seth.Product(H1.Nodes, H2.Nodes)),
        ).inverse,
        seth.coproduct_maps(
            seth.left_distributivity_isomorphism(
                ext=H0.Nodes,
                paire=(
                    seth.Product(H1.Nodes, H2.Links),
                    seth.Product(H1.Links, H2.Nodes),
                ),
            ).inverse,
            seth.Product(H0.Links, seth.Product(H1.Nodes, H2.Nodes)).identity,
        ),
    ]
    link_map = seth.composition_chaine(chaine_link)

    a = HypergraphIsomorphism(
        dom=left_prod,
        cod=right_prod,
        map=(node_map, tie_map, link_map),
        name=f"α_{H0.name},{H1.name},{H2.name}",
    )
    return HypergraphIsomorphism.from_morphism(a)


def funny_product_maps(
    f0: HypergraphMorphism, f1: HypergraphMorphism
) -> HypergraphMorphism:
    """
    Return the morphism 
        
        f0 □ f1 : H0 □ H1 → G0 □ G1

    which has as components:

        - Sf0□f1 = Sf0 × Sf1 : SH0 × SH1 → SG0 × SG1
        - Tf0□f1 = Sf0 × Tf1 + Tf0 × Sf1 : SH0 × TH1 + TH0 × SH1 → SG0 × TG1 + TG0 × SG1
        - Lf0□f1 = Sf0 × Lf1 + Lf0 × Sf1 : SH0 × LH1 + LH0 × SH1 → SG0 × LG1 + LG0 × SG1
    """
    funny_prod_dom = FunnyTensor(f0.dom, f1.dom)
    funny_prod_cod = FunnyTensor(f0.cod, f1.cod)
    node_map = seth.product_maps(f0.node_map, f1.node_map)
    tie_map = seth.coproduct_maps(
        seth.product_maps(f0.node_map, f1.tie_map),
        seth.product_maps(f0.tie_map, f1.node_map),
    )
    link_map = seth.coproduct_maps(
        seth.product_maps(f0.node_map, f1.link_map),
        seth.product_maps(f0.link_map, f1.node_map),
    )
    return HypergraphMorphism(
        dom=funny_prod_dom,
        cod=funny_prod_cod,
        map=(node_map, tie_map, link_map),
        name=f"{f0.name} □ {f1.name}",
    )


def free_prod(H: Representable) -> FunnyTensor:
    """Return the free funny tensor product of a hypergraph H
    H □ 1, which creates a link l_x at each node x of H,
    and a tie t_x at each node x of H, such that t_x : x ∈ l_x."""
    return FunnyTensor(H, Terminal())


## -- Finite FunnyTensor --


class FiniteFunnyTensor(Construct):
    def __init__(self, hypergraph_list: list[Representable]):
        """
        Return the finite funny tensor product of a list of hypergraphs H0 □ H1 □ ... □ Hn:

            - S_{H0 □ H1 □ ... □ Hn} = S_{H0} × S_{H1} × ... × S_{Hn}
            - T_{H0 □ H1 □ ... □ Hn} = ∑_{i} (S_{H0} × ... × S_{H(i-1)} × T_{Hi} × S_{H(i+1)} × ... × S_{Hn})
            - L_{H0 □ H1 □ ... □ Hn} = ∑_{i} (S_{H0} × ... × S_{H(i-1)} × L_{Hi} × S_{H(i+1)} × ... × S_{Hn})
        """
        self.hypergraph_list = hypergraph_list
        self.arité = len(hypergraph_list)
        self._name = " □ ".join(f"{H.name}" for H in hypergraph_list)

        if not hypergraph_list:
            self._obj = Unit_funny()
        else:
            S = seth.FiniteProduct([H.Nodes for H in hypergraph_list])

            L_list = list()
            for i in range(self.arité):
                L_i = seth.FiniteProduct(
                    [
                        H.Links if j == i else H.Nodes
                        for j, H in enumerate(hypergraph_list)
                    ]
                )
                L_list.append(L_i)
            L = seth.FiniteCoproduct(L_list)

            T_list = list()
            for i in range(self.arité):
                T_i = seth.FiniteProduct(
                    [
                        H.Ties if j == i else H.Nodes
                        for j, H in enumerate(hypergraph_list)
                    ]
                )
                T_list.append(T_i)
            T = seth.FiniteCoproduct(T_list)

            node_maps = list()
            for i in range(self.arité):
                sigma_i = seth.finite_product_maps(
                    [
                        H.node_map if j == i else H.Nodes.identity
                        for j, H in enumerate(hypergraph_list)
                    ]
                )
                node_maps.append(sigma_i)
            self._node_map = T.universal_solution(list_f=node_maps)

            link_maps = list()
            for i in range(self.arité):
                lambda_i = seth.finite_product_maps(
                    [
                        H.link_map if j == i else H.Nodes.identity
                        for j, H in enumerate(hypergraph_list)
                    ]
                )
                link_maps.append(lambda_i)
            self._link_map = seth.finite_coproduct_maps(link_maps)

            self._obj = Hypergraph(
                Nodes=S,
                Ties=T,
                Links=L,
                node_map=self._node_map,
                link_map=self._link_map,
                name=self._name,
            )


def finite_funny_maps(morphism_list: list[HypergraphMorphism]) -> HypergraphMorphism:
    """
    Return the morphism 

     f0 □ f1 □ ... □ fn : H0 □ H1 □ ... □ Hn → G0 □ G1 □ ... □ Gn

    which has as components:

        - S(f0 □ f1 □ ... □ fn) = S(f0) × S(f1) × ... × S(fn) : S(H0) × S(H1) × ... × S(Hn) → S(G0) × S(G1) × ... × S(Gn)
        - T(f0 □ f1 □ ... □ fn) = ∑_{i} (S(f0) × ... × S(f(i-1)) × T(fi) × S(f(i+1)) × ... × S(fn)) : ∑_{i} (S(H0) × ... × S(H(i-1)) × T(Hi) × S(H(i+1)) × ... × S(Hn)) → ∑_{i} (S(G0) × ... × S(G(i-1)) × T(Gi) × S(G(i+1)) × ... × S(Gn))
        - L(f0 □ f1 □ ... □ fn) = ∑_{i} (S(f0) × ... × S(f(i-1)) × L(fi) × S(f(i+1)) × ... × S(fn)) : ∑_{i} (S(H0) × ... × S(H(i-1)) × L(Hi) × S(H(i+1)) × ... × S(Hn)) → ∑_{i} (S(G0) × ... × S(G(i-1)) × L(Gi) × S(G(i+1)) × ... × S(Gn))
    """
    if not morphism_list:
        return Unit_funny().identity
    else:
        funny_prod_dom = FiniteFunnyTensor([f.dom for f in morphism_list])
        funny_prod_cod = FiniteFunnyTensor([f.cod for f in morphism_list])
        node_map = seth.finite_product_maps([f.node_map for f in morphism_list])
        tie_map = seth.finite_coproduct_maps(
            [
                seth.finite_product_maps(
                    [
                        f.tie_map if j == i else f.node_map
                        for j, f in enumerate(morphism_list)
                    ]
                )
                for i in range(len(morphism_list))
            ]
        )
        link_map = seth.finite_coproduct_maps(
            [
                seth.finite_product_maps(
                    [
                        f.link_map if j == i else f.node_map
                        for j, f in enumerate(morphism_list)
                    ]
                )
                for i in range(len(morphism_list))
            ]
        )
        return HypergraphMorphism(
            dom=funny_prod_dom,
            cod=funny_prod_cod,
            map=(node_map, tie_map, link_map),
            name=" □ ".join(f"{f.name}" for f in morphism_list),
        )


# TODO : ajouter les unitors, associateurs, et vérifier les axiomes de la structure monoïdale.


## -- FunnyHomgraph and prenaturality


class PrenaturalTransformation:
    """Return a prenatural transformation alpha : H0 -> H1,
    where H0 and H1 are representable hypergraphs,
    and mapping is a function mapping : H0.Nodes -> H1.Links,
    such that for each node s in H0,
    the link alpha(s) in H1 has for source the image of s by the node map of H1,
    i.e. alpha(s) : H1.node_map(mapping(s)) ∈ alpha(s) in H1.
    """

    def __init__(
        self, H0: Representable, H1: Representable, mapping: NamedFunction, name=None
    ):
        self.H0 = H0
        self.H1 = H1
        self.mapping = mapping  # mapping : H0.Nodes -> H1.Links
        self.name = name

    def __repr__(self) -> str:
        return (
            f"Prenatural transformation {self.name} in Hyp[{self.H0.name},{self.H1.name}] \n"
            f"  mapping = {repr(self.mapping)} \n"
        )

    def __eq__(self, other) -> bool:
        return (
            self.H0 == other.H0
            and self.H1 == other.H1
            and self.mapping == other.mapping
        )

    def __hash__(self):
        return hash((hash(self.H0), hash(self.H1), hash(self.mapping)))

    def __lt__(self, other) -> bool:
        if self.H0 != other.H0 or self.H1 != other.H1:
            raise ValueError(
                "Les transformations prénaturelles ne sont pas comparables."
            )
        else:
            return self.name < other.name


class natural_tie:
    """Return a natural tie nw : f -> alpha,
    where f : H0 -> H1 is a hypergraph morphism,
    alpha : H0 -> H1 is a prenatural transformation,
    and map : H0.Nodes -> H1.Ties"""

    def __init__(
        self,
        f: HypergraphMorphism,
        alpha: PrenaturalTransformation,
        map: NamedFunction,
        name=None,
    ):
        self.H0 = f.dom
        self.H1 = f.cod
        self.f = f  # f : H0 -> H1
        self.alpha = alpha  # alpha : H0 -> H1
        self.map = map  # map : H0.Nodes -> H1.Ties
        self.name = name
        self.test = all(
            self.H1.node_map(self.map(s)) == self.f.node_map(s) for s in self.H0.Nodes
        ) and all(
            self.H1.link_map(self.map(s)) == self.alpha.mapping(s)
            for s in self.H0.Nodes
        )
        self.obstructions = set()
        if not self.test:
            for s in self.H0.Nodes:
                if self.H1.node_map(self.map(s)) != self.f.node_map(s):
                    self.obstructions.add((s, "node_map"))
                if self.H1.link_map(self.map(s)) != self.alpha.mapping(s):
                    self.obstructions.add((s, "link_map"))

    def __repr__(self) -> str:
        if self.test:
            return (
                f"Natural tie {self.name} for transformation {self.alpha.name} \n"
                f"  mapping = {repr(self.map)} \n"
            )
        else:
            return (
                f"Invalid natural tie {self.name}  \n"
                f"  obstructions = {repr(self.obstructions)} \n"
            )

    def __str__(self) -> str:
        return f"Natural tie {self.name} for transformation {self.alpha.name}"

    def __eq__(self, other) -> bool:
        return (
            self.H0 == other.H0
            and self.H1 == other.H1
            and self.f == other.f
            and self.alpha == other.alpha
            and self.map == other.map
        )

    def __hash__(self):
        return hash(
            (
                hash(self.H0),
                hash(self.H1),
                hash(self.f),
                hash(self.alpha),
                hash(self.map),
            )
        )

    def __lt__(self, other) -> bool:
        if self.H0 != other.H0 or self.H1 != other.H1:
            raise ValueError("Les témoins naturels ne sont pas comparables.")
        else:
            return self.name < other.name


class FunnyHomgraph(Construct):
    """Return the funny hom-graph Hom[H0, H1]_□,
    where H0 and H1 are representable hypergraphs.
     Nodes = Hom(H0, H1)
     Links = Prenatural transformations H0 -> H1
     Ties = Natural ties between morphisms and prenatural transformations, i.e.
    t : f -> alpha if for each node s in H0, the link alpha(s) in H1 has for source the image of s by the node map of H1,
    i.e. alpha(s) : H1.node_map(t(s)) ∈ alpha(s) in H1.
    """

    def __init__(self, H0: Representable, H1: Representable):
        self.H0 = H0
        self.H1 = H1
        self._name = f"Hom[{H0.name}, {H1.name}]_□"
        self._Nodes = CartesianHomSet(H0, H1)
        self.transformations = set()
        for mapping in HomSet(H0.Nodes, H1.Links):
            self.transformations.add(
                PrenaturalTransformation(
                    H0=H0,
                    H1=H1,
                    mapping=mapping,
                    name=f"alpha_{len(self.transformations)}",
                )
            )
        self._Links = NamedSet(
            self.transformations, name=f"L_Hom[{H0.name}, {H1.name})]_□"
        )
        self.Wit = set()
        for f in self._Nodes:
            for alpha in self._Links:
                # chercher les natural_tie possibles
                for map in HomSet(H0.Nodes, H1.Ties):
                    nw = natural_tie(f, alpha, map, name=f"nw_{len(self.Wit)}")
                    if nw.test:
                        self.Wit.add(nw)
        self._Ties = NamedSet(self.Wit, name=f"T_Hom[{H0.name}, {H1.name})]_□")
        self._node_map = NamedFunction(
            dom=self._Ties,
            cod=self._Nodes,
            table=lambda nw: nw.f,
            name=f"s_Hom[{H0.name}, {H1.name})]_□",
        )
        self._link_map = NamedFunction(
            dom=self._Ties,
            cod=self._Links,
            table=lambda nw: nw.alpha,
            name=f"t_Hom[{H0.name}, {H1.name})]_□",
        )
        self._obj = Hypergraph(
            Nodes=self._Nodes,
            Ties=self._Ties,
            Links=self._Links,
            node_map=self._node_map,
            link_map=self._link_map,
            name=self._name,
        )


def enriched_homset(X: NamedSet, Y: NamedSet) -> Hypergraph:
    """
    Return the simple enriched hom-set Hom[X, Y]_set, where X and Y are sets.

        - S_Hom[X, Y] = Hom(X, Y)
        - L_Hom[X, Y] = P(X) x P(Y))
        - and f : X -> Y is in l_(A, B) with A ⊆ X and B ⊆ Y,
        if for all x in A, s(x) is in B.
    """
    Nodes = HomSet(X, Y)
    Links = seth.Product(seth.powerset(X), seth.powerset(Y))
    Ties = set()
    for s in Nodes:
        for l in Links:
            if all(s(x) in l[1] for x in l[0]):
                Ties.add((s, l))
    Ties_setnom = NamedSet(Ties, name=f"T_Hom[{X.name}, {Y.name}]_set")
    node_map = NamedFunction(
        dom=Ties_setnom,
        cod=Nodes,
        table=lambda t: t[0],
        name=f"s_Hom[{X.name}, {Y.name}]_set",
    )
    link_map = NamedFunction(
        dom=Ties_setnom,
        cod=Links,
        table=lambda t: t[1],
        name=f"l_Hom[{X.name}, {Y.name}]_set",
    )
    return Hypergraph(
        Nodes=Nodes,
        Ties=Ties_setnom,
        Links=Links,
        node_map=node_map,
        link_map=link_map,
        name=f"Hom[{X.name}, {Y.name}]_set",
    )


###########################################################################
## Strong product


class StrongTensor(Construct):
    """
    Return the strong tensor product H0 ⊠ H1 of two hypergraphs H0 and H1

        - S_{H0 ⊠ H1} = S_{H0} × S_{H1}
        - T_{H0 ⊠ H1} = (S_{H0} × T_{H1}) + (T_{H0} × S_{H1}) + (T_{H0} × T_{H1})
        - L_{H0 ⊠ H1} = (S_{H0} × L_{H1}) + (L_{H0} × S_{H1}) + (L_{H0} × L_{H1})
    """

    def __init__(self, H0: Representable, H1: Representable):
        self.H0 = H0
        self.H1 = H1

        # Nodes = S0 × S1
        self._Nodes = seth.Product(H0.Nodes, H1.Nodes)

        # Ties = (S0×T1) + (T0×S1) + (T0×T1)
        S0xT1 = seth.Product(H0.Nodes, H1.Ties)
        T0xS1 = seth.Product(H0.Ties, H1.Nodes)
        T0xT1 = seth.Product(H0.Ties, H1.Ties)
        self._Ties = seth.FiniteCoproduct([S0xT1, T0xS1, T0xT1])

        # Links = (S0×L1) + (L0×S1) + (L0×L1)
        S0xL1 = seth.Product(H0.Nodes, H1.Links)
        L0xS1 = seth.Product(H0.Links, H1.Nodes)
        L0xL1 = seth.Product(H0.Links, H1.Links)
        self._Links = seth.FiniteCoproduct([S0xL1, L0xS1, L0xL1])

        sigma0 = seth.product_maps(H0.Nodes.identity, H1.node_map)
        sigma1 = seth.product_maps(H0.node_map, H1.Nodes.identity)
        sigma2 = seth.product_maps(H0.node_map, H1.node_map)
        self._node_map = self._Ties.universal_solution(list_f=[sigma0, sigma1, sigma2])
        lambda0 = seth.product_maps(H0.Nodes.identity, H1.link_map)
        lambda1 = seth.product_maps(H0.link_map, H1.Nodes.identity)
        lambda2 = seth.product_maps(H0.link_map, H1.link_map)
        self._link_map = seth.finite_coproduct_maps([lambda0, lambda1, lambda2])

        self._name = f"{H0.name} ⊠ {H1.name}"
        self._obj = Hypergraph(
            Nodes=self._Nodes,
            Ties=self._Ties,
            Links=self._Links,
            node_map=self._node_map,
            link_map=self._link_map,
            name=self._name,
        )


def strong_product_maps(
    f0: HypergraphMorphism, f1: HypergraphMorphism
) -> HypergraphMorphism:
    """
    For a pair of hypergraph morphisms f0 : H0 → G0 and f1 : H1 → G1, return the morphism 

         f0 ⊠ f1 : H0 ⊠ H1 → G0 ⊠ G1

    which has as components:

        - Sf0⊠f1 = Sf0 × Sf1 : SH0 × SH1 → SG0 × SG1
        - Tf0⊠f1 = Sf0 × Tf1 + Tf0 + Sf1 + Tf0 x Tf1 : SH0 × TH1 + TH0 + SH1 + TH0 x TH1 → SG0 × TG1 + TG0 + SG1 + TG0 x TG1
        - Lf0⊠f1 = Sf0 × Lf1 + Lf0 + Sf1 + Lf0 x Lf1 : SH0 × LH1 + LH0 + SH1 + LH0 x LH1 → SG0 × LG1 + LG0 + SG1 + LG0 x LG1
    """
    strong_prod_dom = StrongTensor(f0.dom, f1.dom)
    strong_prod_cod = StrongTensor(f0.cod, f1.cod)
    node_map = seth.product_maps(f0.node_map, f1.node_map)
    tie_map = seth.finite_coproduct_maps(
        [
            seth.product_maps(f0.node_map, f1.tie_map),
            seth.product_maps(f0.tie_map, f1.node_map),
            seth.product_maps(f0.tie_map, f1.tie_map),
        ]
    )
    link_map = seth.finite_coproduct_maps(
        [
            seth.product_maps(f0.node_map, f1.link_map),
            seth.product_maps(f0.link_map, f1.node_map),
            seth.product_maps(f0.link_map, f1.link_map),
        ]
    )
    return HypergraphMorphism(
        dom=strong_prod_dom,
        cod=strong_prod_cod,
        map=(node_map, tie_map, link_map),
        name=f"{f0.name} ⊠ {f1.name}",
    )


## -- Finite Strong Tensor --


class FiniteStrongTensor(Construct):
    """
    Return the finite strong tensor product H0 ⊠ H1 ⊠ ... ⊠ Hn of a list of hypergraphs H0 ⊠ H1 ⊠ ... ⊠ Hn:

        - S_(⊠_i H_i) = Π_i S_H_i
        - T_(⊠_i H_i) = ∑_(J⊆I, J≠∅) (Π_{j∈J } T_{H_j} × Π_{j ∉ J} × S_{H_i})
        - L_(⊠_i H_i) = ∑_(J⊆I, J≠∅) (Π_{j∈J } L_{H_j} × Π_{j ∉ J} × S_{H_i})
    """

    def __init__(self, hypergraph_list: list[Representable]):
        self.hypergraph_list = hypergraph_list
        self.arité = len(hypergraph_list)
        self._name = " ⊠ ".join(f"{H.name}" for H in hypergraph_list)

        if not hypergraph_list:
            self._obj = Unit_funny()
        else:
            S = seth.FiniteProduct([H.Nodes for H in hypergraph_list])

            # use bitmask to generate all subsets of I = {0, 1, ..., n-1} except the empty set
            L_list = list()
            for i in range(1, 2**self.arité):
                J = [j for j in range(self.arité) if (i >> j) & 1]
                L_J = seth.FiniteProduct(
                    [
                        H.Links if j in J else H.Nodes
                        for j, H in enumerate(hypergraph_list)
                    ]
                )
                L_list.append(L_J)
            L = seth.FiniteCoproduct(L_list)

            T_list = list()
            for i in range(1, 2**self.arité):
                J = [j for j in range(self.arité) if (i >> j) & 1]
                T_J = seth.FiniteProduct(
                    [
                        H.Ties if j in J else H.Nodes
                        for j, H in enumerate(hypergraph_list)
                    ]
                )
                T_list.append(T_J)
            T = seth.FiniteCoproduct(T_list)

            node_maps = list()
            for i in range(1, 2**self.arité):
                J = [j for j in range(self.arité) if (i >> j) & 1]
                sigma_i = seth.finite_product_maps(
                    [
                        H.node_map if j in J else H.Nodes.identity
                        for j, H in enumerate(hypergraph_list)
                    ]
                )
                node_maps.append(sigma_i)
            self._node_map = T.universal_solution(list_f=node_maps)

            link_maps = list()
            for i in range(1, 2**self.arité):
                J = [j for j in range(self.arité) if (i >> j) & 1]
                lambda_i = seth.finite_product_maps(
                    [
                        H.link_map if j in J else H.Nodes.identity
                        for j, H in enumerate(hypergraph_list)
                    ]
                )
                link_maps.append(lambda_i)
            self._link_map = seth.finite_coproduct_maps(link_maps)

            self._obj = Hypergraph(
                Nodes=S,
                Ties=T,
                Links=L,
                node_map=self._node_map,
                link_map=self._link_map,
                name=self._name,
            )


def finite_strong_maps(morphism_list: list[HypergraphMorphism]) -> HypergraphMorphism:
    """
    For a list of morphisms f_i : H_i → G_i, return the morphism

         f0 ⊠ f1 ⊠ ... ⊠ fn : H0 ⊠ H1 ⊠ ... ⊠ Hn → G0 ⊠ G1 ⊠ ... ⊠ Gn

    which has as components:

        - S_(⊠_i H_i) = Π_i S_f_i
        - T_(⊠_i H_i) = ∑_(J⊆I, J≠∅) (Π_{j∈J } T_{f_j} × Π_{j ∉ J} × S_{f_i})
        - L_(⊠_i H_i) = ∑_(J⊆I, J≠∅) (Π_{j∈J } L_{f_j} × Π_{j ∉ J} × S_{f_i})
    """
    if not morphism_list:
        return Unit_funny().identity
    else:
        strong_prod_dom = FiniteStrongTensor([f.dom for f in morphism_list])
        strong_prod_cod = FiniteStrongTensor([f.cod for f in morphism_list])
        node_map = seth.finite_product_maps([f.node_map for f in morphism_list])
        tie_components = []
        for mask in range(1, 2 ** len(morphism_list)):
            J = [j for j in range(len(morphism_list)) if (mask >> j) & 1]
            component = seth.finite_product_maps(
                [
                    f.tie_map if j in J else f.node_map
                    for j, f in enumerate(morphism_list)
                ]
            )
            tie_components.append(component)

        tie_map = seth.finite_coproduct_maps(tie_components)
        link_components = []
        for mask in range(1, 2 ** len(morphism_list)):
            J = [j for j in range(len(morphism_list)) if (mask >> j) & 1]
            component = seth.finite_product_maps(
                [
                    f.link_map if j in J else f.node_map
                    for j, f in enumerate(morphism_list)
                ]
            )
            link_components.append(component)

        link_map = seth.finite_coproduct_maps(link_components)
        return HypergraphMorphism(
            dom=strong_prod_dom,
            cod=strong_prod_cod,
            map=(node_map, tie_map, link_map),
            name=" ⊠ ".join(f"{f.name}" for f in morphism_list),
        )


## -- Straight tensor product --


class StraightTensor(Construct):
    def __init__(self, H0: Representable, H1: Representable):
        """Return the straight tensor product of two hypergraphs H0 and H1
        defined as the hypergraph

            H0 ⬦ H1 = (H0* □ H1*)*
        """
        self.H0 = H0
        self.H1 = H1
        self._obj = FunnyTensor(H0.dual, H1.dual).dual
        self._name = f"{H0.name} ⬦ {H1.name}"


Binary_Type: TypeAlias = (
    Product
    | Coproduct
    | FunnyTensor
    | StrongTensor
    | StraightTensor
    | FiniteProduct
    | FiniteCoproduct
    | FiniteFunnyTensor
    | FiniteStrongTensor
)

##################################################

# The garden of finite hypergraphs

"""This section defines some specific hypergraphs 
that are useful for testing and examples
and generating finite hypergraphs."""


def finset(n: int) -> NamedSet:
    """Return the finite set {0, 1, ..., n-1} with name [n]."""
    return NamedSet(set(range(n)), f"[{n}]")


def walking_link(n: int) -> Hypergraph:
    """Return the finite hypergraph [n]
    with n nodes and one link"""
    return hypergraph_from_set(set_input=finset(n), name=f"[{n}]")


def walking_links_product(n: int, m: int) -> FunnyTensor:
    """Return the funny tensor product of two walking links [n] and [m]."""
    return FunnyTensor(walking_link(n), walking_link(m))


def walking_links_finite_product(list_nat: list) -> FiniteFunnyTensor:
    """Return the finite funny tensor product of a list of walking links."""
    list_hyp = [walking_link(n) for n in list_nat]
    return FiniteFunnyTensor(list_hyp)


def walking_loop(n: int) -> Hypergraph:
    """Return the hypergraph with a single node
    a single link, and n ties from the node to the link."""
    dict_boucle = {f"t_{i}": ("x", "l") for i in range(n)}
    return hypergraph_from_dict(
        dic=dict_boucle,
        Nodes=NamedSet({"x"}, "S"),
        Links=NamedSet({"l"}, "L"),
        name=f"boucle_{n}",
    )


def walking_links_exponential(n: int, m: int) -> FunnyHomgraph:
    """Return the funny hom-graph Hom[[n], [m]]_□,
    where [n] and [m] are walking links."""
    return FunnyHomgraph(walking_link(n), walking_link(m))


def discret(n: int) -> Hypergraph:
    """Return the discrete hypergraph with n nodes and n links
    with each node isolated in a link."""
    S = finset(n)
    L = finset(n)
    dict_discret = {f"t_{i}": (i, i) for i in range(n)}
    return hypergraph_from_dict(dic=dict_discret, Nodes=S, Links=L, name=f"discret_{n}")


def hyp_from_matrice(matrice: np.ndarray) -> Hypergraph:
    """Return the hypergraph associated to a matrix of natural numbers,
    with n rows and m columns, where each entry (i, j) of the matrix
    gives the number of ties from node i to link j."""
    n, m = matrice.shape
    S = NamedSet(elements={f"x{i}" for i in range(n)}, name="S")
    L = NamedSet(elements={f"l{j}" for j in range(m)}, name="L")
    dict_hyp = {
        f"t_{i}_{j}_{k}": (f"x{i}", f"l{j}")
        for i in range(n)
        for j in range(m)
        for k in range(int(matrice[i, j]))
        if matrice[i, j] > 0
    }
    return hypergraph_from_dict(
        dic=dict_hyp, Nodes=S, Links=L, name=f"hyp_from_matrice_{n}_{m}"
    )


def hyp_to_matrice(
    hyp: Representable, sorting_nodes: list, sorting_links: list
) -> np.ndarray:
    """For H a hypergraph, given a sorting of the nodes and links of H,
    return the M(H) with M(H)[i, j]  = <xi, l_j> the number of ties from node x_i to link l_j in H.
    """
    n = len(sorted(hyp.Nodes, key=lambda x: sorting_nodes.index(x)))
    m = len(sorted(hyp.Links, key=lambda x: sorting_links.index(x)))
    matrice = np.zeros((n, m), dtype=int)
    for t in hyp.obj.dictionnaire():
        i = sorting_nodes.index(hyp.obj.dictionnaire()[t][0])
        j = sorting_links.index(hyp.obj.dictionnaire()[t][1])
        matrice[i, j] += 1
    return matrice


def sorting_defaut(s: set) -> list:
    return sorted(s, key=lambda x: x)


def preim_partition(f: NamedFunction) -> Hypergraph:
    """Return the hypergraph associated to the partition of the domain of f
    by the preimages of the elements of the codomain of f, i.e.
    S = dom(f), L = cod(f), and T = {(s, f(s)) | s in S}."""
    S = f.dom
    L = f.cod
    T = set()
    dict_preim = {}
    for s in S:
        T.add(f"t_({s}, {f(s)}))")
        dict_preim[f"t_({s}, {f(s)}))"] = (s, f(s))
    return hypergraph_from_dict(
        dic=dict_preim, Nodes=S, Links=L, name=f"preim_partition_{f.__str__()}"
    )


def homsetfin(n: int, m: int) -> HomSet:
    """Return the hom-set Hom[[n], [m]],
    where [n] and [m] are walking links."""
    return HomSet(finset(n), finset(m))


def sum_lien_libres(n: int, m: int) -> Coproduct:
    """Return the coproduct of two walking links [n] and [m]."""
    return Coproduct(walking_link(n), walking_link(m))


def sum_fin_lien_libres(list_nat: Sequence) -> FiniteCoproduct:
    """Return the coproduct of a list of walking links."""
    list_hyp = [walking_link(n) for n in list_nat]
    return FiniteCoproduct(list_hyp)


def list_hyp(n: int) -> Hypergraph:
    """Return the hypergraph of lists of length n,
    with n nodes and n links [i] containing the j < i."""
    S = finset(n)
    L = NamedSet({f"[{i}]" for i in finset(n)}, f"L_{n}")
    dict_hyp = {f"t_{j,i}": (j, f"[{i}]") for i in finset(n) for j in range(i)}
    return hypergraph_from_dict(dic=dict_hyp, Nodes=S, Links=L, name=f"List_{n}")


def list_ordered(n: int) -> Hypergraph:
    """Return the hypergraph of ordered lists of length n,
    with n nodes and n(n+1)/2 links [i, j]
    containing the k with i <= k <= j."""
    S = finset(n)
    L = NamedSet(
        {f"[{i}, {j}]" for i in finset(n) for j in finset(n) if i <= j}, f"L_{n}"
    )
    dict_hyp = {
        f"t_{i}_{j}": (i, f"[{i}, {j}]") for i in finset(n) for j in finset(n) if i <= j
    }
    return hypergraph_from_dict(
        dic=dict_hyp, Nodes=S, Links=L, name=f"Ordered_list_{n}"
    )


def list_2_by_2(n: int) -> Hypergraph:
    """Return the hypergraph of lists of length n,
    with n nodes and n-1 links [i, i+1] containing the nodes i and i+1."""
    S = finset(n)
    L = NamedSet({f"[{i}, {i+1}]" for i in finset(n) if i < n - 1}, f"L_{n}")
    dict_hyp = {f"t_{i}_{0}": (i, f"[{i}, {i+1}]") for i in finset(n) if i < n - 1} | {
        f"t_{i}_{1}": (i + 1, f"[{i}, {i+1}]") for i in finset(n) if i < n - 1
    }
    return hypergraph_from_dict(dic=dict_hyp, Nodes=S, Links=L, name=f"list_2_by_2_{n}")


def reticulation(n: int, m: int) -> FunnyTensor:
    """Return the funny tensor of two hypergraphs of lists of length n and m."""
    return FunnyTensor(list_2_by_2(n), list_2_by_2(m))


## container pour les instances de classe de type opération (limites, colimite, tenseur, homgraph)


def obj(input) -> Hypergraph:
    if isinstance(input, Hypergraph):
        return input
    else:
        return input.obj
