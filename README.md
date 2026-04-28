# Hypypyp



A small python library to work in the categories of (finite) sets and (finite) hypergraphs.



## Overview



Hypypyp contains two modules:

* Seth: a module to work in the category of (finite) sets
* Hyp : a module depending on Seth to work in the category of (finite) hypergraphs

## Documentation

The documentation is available here:

https://axel-osmond.github.io/hypypyp/


## Seth



Seth allows to represent (finite) sets and maps between them.



Sets are represented by the class NamedSet, whose instances consist of a frozenset and a name.

As frozenset are immutable, so are NamedSet, which can hence be elements of other NamedSets.



Maps between sets are represented by the class NamedFunction, which take as parameters a domain, a codomain, a table, and a name. The table can be either a dictionnary for extensive definition, or a callable for generic description.



The usual universal constructions are represented each by a class:

* products, coproducts,
* finite product, finite coproducts
* terminal object, initial object,
* pullback, pushout
* equalizers, coequalizers



Each of those classes comes with methods corresponding to the natural constructions, as the canonical projections/inclusions, the solutions provided by universal property, the structure maps for the underlying monoidal structure for product and coproduct etc...


Pushout and coequalizers rely on the methods of a class Setoid, which may also be used to implement equality based algebraic structures.



In order to preserve the structure attached to objects, universal constructions and named sets are subsumed by two layer of superclasses from which they inherit basic methods. The hierarchy is:



```text

Representable
  |--- NamedSets
  |--- Constructs
         |--- Products, Coproducts
         |--- Terminal, Initial
         |--- other limits and colimits
         |--- Setoids, homsets…
```

Universal constructions, as well as the domain and codomain arguments of NamedFunctions, take Representable as input: hence the structure attached to object can be composed without loss. One accesses to the underlying object of a representable by the attribute .obj.

Finally topos aspects of sets are implemented via

* a class to represent and calculate homsets
* set theoretic constructions as covariant and contravariant powerset
* subobject calculus



## Hyp



Hyp allows to represent hypergraphs and hypergraphs morphisms between them.



Here we define hypergraphs as mere span of sets: H = (S,T,L) with maps σ : T → S, λ : T → L. Elements of S are called nodes, elements of T are called ties, and elements of L are called links; if σ(t) = s and λ(t) = l, one can write t : s ∈ l to means that t witnesses that the node s lies in the link l. This allows for multiple occurrences of a same node in a same link (hence for loops). Links may have arbitrary size and shape.



Hypergraphs are represented by the class Hypergraph, which take as S,T,L three NamedSets and as σ,λ two NamedFunction: relying on the classes NamedSet and NamedFunction will allow to invoke the methods of seth. The resulting objects are immutable and eligible to different methods returning as dictionaries the relations between nodes, links and ties, or also displaying supports of links, occurrences of nodes etc...



A second class, MutableHypergraphs, simplifies their presentation via native sets to allow for operation on them as adding, removing, renaming elements. One can go from mutable to immutable and vice-versa.



Hypergraph morphisms f : H → G are triples f = (Sf : SH → SG, Tf : TH → TG, Lf : LH → LG) satisfying that σG Tf = Sf σH and λG Tf = Lf λH. They are represented by triple of NamedFunctions.



The category of hypergraphs is a presheaf topos, hence all limits and colimits are easily constructed from those in Seth. Again a hierarchy of Representable, Construct, Hypergraph and corresponding classes for universal construction allow to stack structure without loss.



Also available are the different closed monoidal structures on Hyp, namely

* the funny tensor product
* the straight tensor product
* the strong tensor product

Funny product comes with its structure maps and homgraphs (v1)



Some topos constructions are also available, as the subobject classifier.



## Status



This project is experimental and under active development.



The public API may change. The current version is intended primarily for mathematical experimentation, examples, and future visualization tools.



## Installation



From the repository root:



pip install -e .



For development tools:



pip install -e ".\[dev]"



## Quick examples



### To initialize a NamedSet



X = seth.NamedSet(name = "X", elements = {1, 2, 3})



X.content() to display its elements inline, X.display() to spread each elements on distincts lines



### To initialize a NamedFunction



```python

Y = seth.NamedSet(name = "Y", elements = {"a", "b"})



f_table = {
        1: "a",
        2: "a",
        3: "b"
    }
    f = seth.NamedFunction(name = "f", dom = X, cod = Y, table = f_table)

f.display(), f.image\_factorization, f.fibers\_decomposition

```



Beware that composition(f,g) is g ∘ f



## To initialize a Hypergraph



```python

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


HX.identity, HX.dual, HX.support\_ties('x0'), HX.hypergraph\_to\_mutable()…

```



Also are provided a list of simple finite hypergraphs:



walking\_link(n), walking\_links\_product(n,m), walking\_loop(n), discret(n), reticulation(n,m)...



## To initialize a HypergraphMorphism



```python

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

```



## To initialize constructions



Product(X,Y), Product(X,Y).proj\_0...

FunnyTensor(HX,HY)...



Beware that universal constructions in Set and Hyp as (co)product, initial/terminal, pullback/pushout etc have the same name, so beware to import them separately as seth.construction… and hyp.construction… or to create alias.



## Acknowledgements



This project was developed with assistance from ChatGPT and GitHub Copilot for code review, documentation, refactoring suggestions, and development workflow support.



All mathematical definitions, design decisions, and implementation choices remain the responsibility of the author.

