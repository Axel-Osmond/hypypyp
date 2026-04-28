# Hypypyp



A small python Library to work in the categories of (finite) sets and (finite) hypergraphs.



## Overview



Hypypyp contains two librarires:

* Seth: a module to work in the category of (finite) sets
* Hyp : a module depending on Seth to work in the category of (finite) hypergraphs



## Seth



Seth allows to represent (finite) sets and map between them.



Sets are represented by the class NamedSet, whose instances consist of a frozenset and a name.

As frozenset are immutables, so are NamedSet, which can hence be elements of other NamedSets.



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



Representable

&#x09;|--- NamedSets

&#x09;|--- Constructs

&#x09;	|--- Products, Coproducts

&#x09;	|--- Terminal, Initial

&#x09;	|--- other limits and colimits

&#x09;	|--- Setoids, homsets...



Universal constructions, as well as the domain and codomain arguments of NamedFunctions, take Representable as input: hence the structure attached to object can be composed without loss. One access to the underlying object of a representable by the attribute .obj.



Finally topos aspects of sets are implemented via

* a class to represent and calculate homsets
* set theoretic constructions as covariant and contravariant powerset
* subobject calculus



## Hyp



Hyp allows to represent hypergraphs and hypergraphs morphisms between them.



Here we define hypergraphs as mere span of sets: H = (S,T,L) with maps σ : T → S, λ : T → L. Elements of S are called nodes, elements of T are called ties, and elements of L are called links; if σ(t) = s and λ(t) = l, one can write t : s ∈ l to means that t witnesses that the node s lies in the link l. This allows for multiple occurrences of a same node in a same link (hence for loops). Links may have arbitrary size and shape.



Hypergraphs are represented by the class Hypergraph, which take as S,T,L three NamedSets and as σ,λ two NamedFunction: relying on the classes NamedSet and NamedFunction will allow to invoque the methods of seth. The resulting objects are immutable and eligible to different methods returning as dictionnaries the relations between nodes, links and ties, or also displaying supports of links, occurrences of nodes etc...



A second class or MutableHypergraphs simplifies their presentation via native sets to allow for operation on them as adding, removing, renaming elements. One can go from mutable to immutable and vice-versa.



Hypergraph morphisms f : H → G are triples f = (Sf : SH → SG, Tf : TH → TG, Lf : LH → LG) satisfying that σG Tf = Sf σH and λG Tf = Sf λH. They are represented by triple of NamedFunctions. 



The category of hypergraphs is a presheaf topos, hence all limits and colimits are easily constructed from those in Seth. Again a hierarchie of Representable, Construct, Hypergraph and corresponding classes for universal construction allow to stack structure without loss. 



Also available are the different closed monoidal structures on Hyp, namely 

* the funny tensor product
* the straight tensor product
* the strong tensor product

Funny product comes with its structures maps and homgraphs (v1)



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



Y = seth.NamedSet(name = "Y", elements = {"a", "b"})



f\_table = {

&#x20;       1: "a",

&#x20;       2: "a",

&#x20;       3: "b"

&#x20;   }

f = seth.NamedFunction(name = "f", dom = X, cod = Y, table = f\_table)



f.display(), f.image\_factorization, f.fibers\_decomposition



Beware that composition(f,g) is g ∘ f



## To initialize a Hypergraph



SX = seth.NamedSet(elements = {'x0', 'x1'}, name = 'SX')

&#x20;   TX = seth.NamedSet(elements = {'t0', 't1'}, name = 'TX')

&#x20;   LX = seth.NamedSet(elements = {'lx'}, name = 'LX')



&#x20;   HX = hyp.Hypergraph(

&#x20;   Nodes = SX,

&#x20;   Ties = TX,

&#x20;   Links = LX,

&#x20;   node\_map = seth.NamedFunction(

&#x20;       dom = TX,

&#x20;       cod = SX,

&#x20;       table = {'t0': 'x0', 't1': 'x1'},

&#x20;       name = 'node\_map'),

&#x20;   link\_map = seth.NamedFunction(

&#x20;       dom = TX,

&#x20;       cod = LX,

&#x20;       table = {'t0': 'lx', 't1': 'lx'},

&#x20;       name = 'link\_map'),

&#x20;   name = 'HX'

&#x20;   )



HX.identity, HX.dual, HX.support\_ties('x0'), HX.hypergraph\_to\_mutable()…



Also are provided a list of simple finite hypergraphs:



walking\_link(n), walking\_links\_product(n,m), walking\_loop(n), discret(n), reticulation(n,m)...



## To initialize a HypergraphMorphism



Sf = seth.NamedFunction(

&#x20;   dom = SX,

&#x20;   cod = SY,

&#x20;   table = {'x0': 'y0', 'x1': 'y1'},

&#x20;   name = 'Sf'

&#x20;   )

&#x20;   Tf = seth.NamedFunction(

&#x20;       dom = TX,

&#x20;       cod = TY,

&#x20;       table = {'t0': 'r0', 't1': 'r1'},

&#x20;       name = 'Tf'

&#x20;   )

&#x20;   Lf = seth.NamedFunction(

&#x20;       dom = LX,

&#x20;       cod = LY,

&#x20;       table = {'lx': 'ly'},

&#x20;       name = 'Lf'

&#x20;   )



&#x20;   f = hyp.HypergraphMorphism(

&#x20;   dom = HX,

&#x20;   cod = HY,

&#x20;   map = (Sf, Tf, Lf),

&#x20;   name = 'f'

&#x20;   )





## To initialize constructions



Product(X,Y), Product(X,Y).proj\_0...

FunnyTensor(HX,HY)...



Beware that universal constructions in Set and Hyp as (co)product, initial/terminal, pullback/pushout etc have the same name, so beware to import them separately as seth.construction… and hyp.construction… or to create alias. 



