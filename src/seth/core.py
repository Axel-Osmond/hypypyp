# Seth: a library for the category of (finite) sets

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping, Callable, Sequence
import itertools as it
from types import MappingProxyType
from . import _native as rustic

#########################################################################################

# Base types

## -- Representables --


class Representable(ABC):
    """A representable is an object that can be represented by an underlying set.
    It will be the parent class of:

    - NamedSet, which is a set with a name and a custom display method;
    - Classes for universal constructions like limits and colimits, which will have an underlying NamedSet together with structure
    - homsets and some distributors

    This superclass promises:

    - that the underlying set is accessible via the obj property
    - that this underlying set is a NamedSet
    - that the representable has a name, which is the name of the underlying set"""

    @property
    @abstractmethod
    def obj(self) -> NamedSet:
        """Underlying representing object."""
        pass

    @property
    def identity(self) -> Bijection:
        """Return the identity function on the named set."""
        regle_id = lambda x: x
        return Bijection(dom=self, cod=self, table=regle_id, name=f"id_{self.name}")

    @property
    def name(self) -> str:
        return self.obj.name

    def __eq__(self, other: object) -> bool:
        """Equality based on the underlying set, independently of the name."""
        if not isinstance(other, Representable):
            return NotImplemented
        return self.obj == other.obj

    def __hash__(self):
        """Hash based on the underlying set."""
        return hash(self.obj)

    def __repr__(self) -> str:
        return self.obj.__repr__()
    
    def content(self) -> str:
        return self.obj.content()

    def __str__(self) -> str:
        return self.obj.__str__()

    def display(self) -> str:
        return self.obj.display()

    def rename(self, new_name: str) -> Representable:
        return self.obj.rename(new_name)

    def rename_elements(self, rename_func: Callable | dict) -> Representable:
        return self.obj.rename_elements(rename_func)

    @abstractmethod
    def __iter__(self):
        """Iterate over the elements of the underlying set."""
        pass

    def __len__(self) -> int:
        return len(self.obj)


    ### Enumerative methods
    def __contains__(self, x) -> bool:
        """Check if an element is in the underlying set."""
        return x in self.obj

    @property
    def enumeration(self) -> tuple:
        """Return the enumeration of the underlying set."""
        return self.obj.enumeration

    def rank(self, x) -> int:
        """Return the rank of an element in the underlying set."""
        return self.obj.rank(x)

    def unrank(self, n: int):
        """Return the element of rank n in the underlying set."""
        return self.obj.unrank(n)
    


## -- Constructs --


class Construct(Representable):
    """A construct represents the result of a universal construction
    It is a representable, hence has a name, .obj returns an underlying object which is always a NamedSet.
    It is eligible as a domain or codomain of functions, and can be used in further constructions.
    It factorizes the method for iter, repr and len"""

    _obj: NamedSet
    _name: str

    @property
    def obj(self) -> NamedSet:
        return self._obj

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return self.obj.display()

    def __iter__(self):
        return iter(self.obj)

    #########
    # Enumerative methods

    def __contains__(self, x) -> bool:
        """Check if an element is in the underlying set."""
        return x in self.obj

    @property
    def enumeration(self) -> tuple:
        """Return the enumeration of the underlying set."""
        return self.obj.enumeration

    def rank(self, x) -> int:
        """Return the rank of an element in the underlying set."""
        return self.obj.rank(x)

    def unrank(self, n: int):
        """Return the element of rank n in the underlying set."""
        return self.obj.unrank(n)


# TODO : enumerate will have to be removed when implementing infinite sets

## -- Frozen sets with custom display --


class FrozenSetAffiche(frozenset):
    """A frozenset with a custom display method."""

    def __repr__(self):
        """Display the elements of the frozenset in a sorted order.
         Example: {a, b, c} instead of {c, a, b} or {b, c, a}."""
        parts = [
            repr(x) for x in sorted(self, key=lambda x: (type(x).__name__, repr(x)))
        ]
        if any("\n" in part for part in parts):
            indented_parts = [part.replace("\n", "\n\t") for part in parts]
            return "{\n\t" + ",\n\t".join(indented_parts) + "\n}"
        return "{" + ", ".join(parts) + "}"

    def __str__(self):
        return self.__repr__()

    def display(self):
        """Display the elements of the frozenset in a sorted order with indentation.
        Example:
            {a, b, c} will be displayed as:
                a
                b
                c
        """
        listloc = sorted(self, key=lambda x: (type(x).__name__, repr(x)))
        disp = str()
        for x in listloc:
            disp = disp + f" \t {repr(x)} \n"
        return disp


## -- Named sets --


class NamedSet(Representable):
    """A named set is a FrozensetAffiche equipped with a name.
    Named sets are immutable, hence hashable; they have an extensional equality
    and can be themselves elements of other sets, or be keys in dictionnaries.
    Named sets will also be provided with enumeration of their elements, 
    which will be used to allow for light representations of functions and other constructions."""

    def __init__(self, elements: set | frozenset | FrozenSetAffiche, name: str, enum: tuple | None = None):
        """Initialize a NamedSet with a given set and name:

        Args:
            elements: a set, frozenset, or FrozenSetAffiche representing the elements of the set.
            name: a string representing the name of the set.

        The constructor ensures that the underlying set is stored as a FrozenSetAffiche
        """

        if isinstance(elements, FrozenSetAffiche):
            self.set = elements
        else:
            self.set = FrozenSetAffiche(elements)
        self._name = name


    ################################################################
        # New code added to support enumeration of the elements of the set

        if enum is not None:
            self._enum = tuple(enum)
        elif all(hasattr(x, "name") for x in self.set):
            self._enum = tuple(
                sorted(self.set, key=lambda x: (x.name, type(x).__name__, repr(x)))
            )
        else:
            self._enum = tuple(
                sorted(self.set, key=lambda x: (type(x).__name__, repr(x)))
            )

        if len(self._enum) != len(self.set) or frozenset(self._enum) != self.set:
            raise ValueError("The enumeration must contain all elements of the set exactly once.")
        
        self._enum_dict = {x: i for i, x in enumerate(self._enum)}

    @property
    def enumeration(self):
        """Return the enumeration of the underlying set."""
        return self._enum

    def rank(self, x):
        """Return the rank of an element in the underlying set."""
        return self._enum_dict[x]

    def unrank(self, i):
        """Return the element of the underlying set with the given rank."""
        return self._enum[i]

    def restrict_enum(self, subset: set | frozenset | FrozenSetAffiche) -> tuple:
        """Return the enumeration of the subset, preserving the order of the original enumeration."""
        return tuple(x for x in self._enum if x in subset)

    ############################################################

    @property
    def obj(self) -> NamedSet:
        return self  # The underlying representing object is itself

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        """Equality based on the underlying set, independently of the name."""
        if not isinstance(other, NamedSet):
            return NotImplemented
        return self.set == other.set

    def __hash__(self):
        """Hash based on the frozen underlying set."""
        return hash(self.set)

##############################################################

    def __iter__(self):
        """Iterate over the elements of the underlying set.
        Hence one can write "for x in A" where A is a NamedSet"
        and it will iterate over the elements of A.set.
        Iter is based on the enumeration."""
        return iter(self._enum)


##############################################################

    def __len__(self) -> int:
        """Return the cardinality of the underlying set."""
        return len(self.set)

    def __contains__(self, x) -> bool:
        """Check if x is in the underlying set."""
        return x in self.set

    def __repr__(self) -> str:
        """Display the name and the elements of the set in a sorted order."""
        """Example: {a, b, c} instead of {c, a, b} or {b, c, a}."""
        parts = [
            repr(x) for x in self._enum
        ]
        if any("\n" in part for part in parts):
            indented_parts = [part.replace("\n", "\n\t") for part in parts]
            return "{\n\t" + ",\n\t".join(indented_parts) + "\n}"
        return "{" + ", ".join(parts) + "}"
    
    def content(self) -> str:
        """Display the name and the elements of the set in a sorted order."""
        return f"{self.name} = {FrozenSetAffiche(self.set).__repr__()}"

    def display(self):
        """Display the elements of the frozenset in a sorted order with indentation.
        Example:
            {a, b, c} will be displayed as:
                a
                b
                c
        """
        disp = str()
        for x in self._enum:
            disp = disp + f" \t {repr(x)} \n"
        return disp

    def __str__(self) -> str:
        """Display only the name of the set."""
        return f"{self.name}"

    def __lt__(self, other) -> bool:
        """Order based on the name, independently of the underlying set."""
        if not isinstance(other, NamedSet):
            return NotImplemented
        return self.name < other.name

    def unfreeze(self) -> set:
        """Return the underlying set as a regular set (not frozenset)."""
        return set(self.set)

##########################################################
    # with enumeration process

    def add(self, x) -> "NamedSet":
        """Return a new NamedSet with the same name and the element x added to the underlying set."""
        if x in self.set:
            return self

        return NamedSet(self.set | {x}, self.name, enum=(*self._enum, x))


    def remove(self, x) -> "NamedSet":
        """Return a new NamedSet with the same name and the element x removed from the underlying set."""
        if x not in self.set:
            raise KeyError(x)  # ou conserver le ValueError actuel de set.remove

        return NamedSet( self.set - {x}, self.name, enum=tuple(y for y in self._enum if y != x))


    def rename(self, new_name: str) -> "NamedSet":
        """Return a new NamedSet with the same underlying set
        but a new name."""
        return NamedSet(self.set, new_name, enum = self._enum)

   

    def rename_elements(self, rename_func: Callable | dict) -> "NamedSet":
        """Return a new NamedSet with the same name
        but with the elements renamed according to rename_func."""
        if isinstance(rename_func, Mapping):
            transform = lambda x: rename_func.get(x, x)
        else:
            transform = rename_func
        new_enum = tuple(transform(x) for x in self._enum)
        new_set = frozenset(new_enum)
        if len(new_set) != len(self.set):
            raise ValueError("The renaming function must be injective (one-to-one).")
        return NamedSet(new_set, self.name, enum=new_enum)

    
    @classmethod
    def from_list(cls, elements: list, name: str) -> NamedSet:
        """Create a NamedSet from a list of elements, preserving the order of the list."""
        unique_elements = []
        seen = set()
        for x in elements:
            if x not in seen:
                unique_elements.append(x)
                seen.add(x)
        return cls(set(unique_elements), name, enum=tuple(unique_elements))

def name_by_content(elements: set | frozenset | FrozenSetAffiche) -> NamedSet:
    """Generate a name for a set based on its content.
    The name is generated by sorting the elements
    and joining their string representations with commas."""
    sorted_elements = sorted(elements, key=lambda x: (type(x).__name__, repr(x)))
    name = "{" + ", ".join(repr(x) for x in sorted_elements) + "}"
    return NamedSet(elements, name)


def finset(n: int) -> NamedSet:
    """Return the finite set {0, 1, ..., n-1} as a NamedSet."""
    return NamedSet(set(range(n)), name=f"[{n}]")


########################################################################################

# Setoids


class Setoid(Construct):
    """A setoid is a set equipped with an equivalence relation,
    represented by a set of generating equalities.
    The quotient is represented by the set of equivalence classes,
    which are frozensets of elements of X.
    The projection is the function that sends each element of X to its equivalence class.
    Uses a union-find data structure to compute the equivalence classes efficiently.
    Used to compute colimits of sets or present structures by generators and relations.
        
    Args:
        X: a Representable representing the underlying set of the setoid.
        eq: a set of pairs (x, y) of elements of X
                representing the generating equalities x = y.
        name: an optional name for the setoid.

        The underlying object is the quotient set.
        The enumeration of the quotient is computed by sorting the equivalence classes 
        by the rank of their minimal element in the enumeration of X.
    """

    def __init__(self, X: Representable, eq: set[tuple]):
        self.generator = X
        self.X = X.obj
        self._name = f"({X.name}/{eq})"
        self.eq = set(eq)

        hors_de_X = {(x, y) for (x, y) in self.eq if x not in X or y not in X}
        if hors_de_X:
            raise ValueError(
                f"Some generating equalities are out of {X.name}: {hors_de_X}"
            )

        # Union-find
        self.parent = {x: x for x in X}
        self.rank = {x: 0 for x in X}

        for x, y in self.eq:
            self._union(x, y)

        # Equivalence classes
        classes: dict = {}
        for x in X:
            r = self.find(x)
            classes.setdefault(r, set()).add(x)

        self.classes = FrozenSetAffiche(FrozenSetAffiche(c) for c in classes.values())
        self._obj = (
            self.quotient()
        )  # The underlying representing object is the quotient set

        # Cache: element -> class
        self._class_of = {}
        for c in self.classes:
            for x in c:
                self._class_of[x] = c

        # Lazy cache for the explicit closure
        self._closure = None

    # ---------- Union-find kernel ----------

    def find(self, x):
        """Find the representative of the class of x, with path compression."""
        if x not in self.parent:
            raise KeyError(f"{x} n'appartient pas à {self.X.name}")
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # compression de chemin
        return self.parent[x]

    def _union(self, x, y):
        """Union the classes of x and y, using union by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return

        # union par rang
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

    # ---------- Basic access ----------

    def class_of(self, x) -> FrozenSetAffiche:
        """Return the equivalence class of x as a frozenset."""
        if x not in self._class_of:
            raise KeyError(f"{x} n'appartient pas à {self.X.name}")
        return self._class_of[x]

    def sorted_class(self, x) -> tuple:
        """Return the equivalence class of x as a sorted tuple by restricting the enumeration of X to the class of x."""
        return self.X.restrict_enum(self.class_of(x))

    def created_equality(self, x, y) -> bool:
        """Return True if x and y are in the same equivalence class,
        i.e., if they are created equalities."""
        return self.find(x) == self.find(y)

    def quotient(self) -> NamedSet:
        """Return the quotient set as a NamedSet,
        where the elements are the equivalence classes."""
        quotient_enum = tuple(sorted(self.classes, key=lambda c: self.X.rank(self.X.restrict_enum(c)[0])))
        return NamedSet(set(quotient_enum), f"{self.X.name}/eq", enum=quotient_enum)

    def projection(self) -> NamedFunction:
        """
        Return the projection function from X to the quotient,
        sending each element to its class

            q : X -> X/eq

        defined as

            q(x) = [x] 

        """
        Q = self.quotient()
        return NamedFunction(
            dom=self.X,
            cod=Q,
            table=lambda x: self.class_of(x),
            name=f"q_({self.name}, {self.eq})",
        )

    # ---------- Explicit closure of eq ----------

    def closure(self):
        """Return the explicit closure of eq,
        i.e., the set of all pairs (x, y)
        such that x and y are in the same equivalence class."""
        if self._closure is None:
            self._closure = {(x, y) for c in self.classes for x in c for y in c}
        return self._closure

    # ---------- Displays ----------

    def representation(self) -> set[str]:
        """Return a set of string representations of the equivalence classes.
        Represent each class by its minimal element (in sorted order) between brackets.
        """
        repres = set()
        for c in self.classes:
            lst = sorted(c, key=str)
            repres.add(f"[{lst[0]}]")
        return repres

    def card(self) -> int:
        """Return the cardinality of the quotient,
        i.e., the number of equivalence classes."""
        return len(self.classes)

    def created_equalities(self, x) -> str:
        """Return a string representation of the equivalence class of x."""
        return "=".join(str(y) for y in sorted(self.class_of(x), key=str))

    def equalities(self) -> str:
        """Return a string representation of all the equivalence classes,
        sorted by their minimal element."""
        classes_tries = sorted(self.classes, key=lambda c: sorted(c, key=str)[0])
        return ", ".join(
            "=".join(str(y) for y in sorted(c, key=str)) for c in classes_tries
        )

    # ---------- Python methods ----------

    def __iter__(self):
        """Iterate over the equivalence classes of the quotient."""
        return iter(self.quotient())

    def __len__(self) -> int:
        """Return the number of equivalence classes in the quotient."""
        return len(self.quotient())

    def __contains__(self, x) -> bool:
        """Return True if x is an equivalence class in the quotient,
        i.e., if x is in the set of classes."""
        return x in self.quotient()

    def __repr__(self) -> str:
        """Display the name, the generators, and the generating equalities."""
        return f"Setoid(X={self.X!r}, eq={self.eq!r}, name={self.name!r})"

    def __str__(self) -> str:
        """Display the name and the quotient set."""
        name = f", name={self.name}" if self.name else ""
        return f"Setoid({self.X.name}{name})"


########################################################################################

# Functions between named sets


class NamedFunction:
    """
    A function f : X -> Y is represented by a rule or a table
    that assigns to each element x of X an element f(x) of Y.

    Args: 

        dom: the domain as a Representable
        cod: the codomain as a Representable
        table: a mapping defining the function; be given as :

            - a mapping (e.g., dict)
            - or as a callable (e.g., lambda).

        name: an optional name for the function


    V1 : all sets are finite,
    so we can always define the function by enumerating the domain and codomain.

    _raw is the tuple of ranks of the images of the elements of the domain,
    and rustic_code is the integer encoding of the function as a base-|Y| number.
    """

    def __init__(
        self,
        dom: Representable,
        cod: Representable,
        table: Mapping | Callable,
        name: str = "",
    ):
        """Initialize a named function:

    Args:

        dom and cod: the domain and codomain as Representable instances
        table: a mapping or a callable defining the function
        name: an optional name for the function

    Examples: 

        If dom.set = {a, b} and cod.set = {0, 1},
        one can define f by a dict: table = {a: 0, b: 1}
        If dom.set = {0, 1} and cod.set = {'x0', 'x1'},
        one can define f by a lambda: table = lambda i: f"x{i}"
        """
        self.dom = dom
        self.cod = cod
        self.table = table
        self.name = name if name else "f"
        if isinstance(table, Mapping):
            self.entry_mode = "mapping"
            values = dict(table)
        elif callable(table):
            self.entry_mode = "callable"
            values = {x: table(x) for x in self.dom}
        else:
            raise TypeError("Table must be a mapping or a callable.")
        self.values = MappingProxyType(values)
        self._verify_values()

        self._dom_enum = self.dom.enumeration
        self._cod_enum = self.cod.enumeration
        self._raw = tuple(self.cod.rank(self.values[x]) for x in self._dom_enum)
        try:
            self.rustic_code = rustic.encode_function(len(self.cod), self._raw)
        except OverflowError:
            self.rustic_code = sum(
                image * len(self.cod) ** position
                for position, image in enumerate(self._raw)
            )


    def _verify_values(self) -> None:
        """Verify that the values are well-defined, i.e.:

        - that the function is total
        - and that the images are in the codomain."""
        dom_keys = set(self.dom)
        val_keys = set(self.values)
        if val_keys != dom_keys:
            missing = dom_keys - val_keys
            extra = val_keys - dom_keys
            raise ValueError(
                f"{self.dom.name} is not total " f"Missing: {missing}; Extra: {extra}"
            )
        images_hors_cod = {y for y in self.values.values() if y not in self.cod}
        if images_hors_cod:
            raise ValueError(
                f"Some images are not in the codomain {self.cod.name}: {images_hors_cod}"
            )

    def __call__(self, x) -> object:
        """Make the function callable: f(x) returns the image of x under f."""
        if x not in self.dom:
            raise KeyError(f"{x} does not belong to the domain {self.dom.name}")
        return self.values[x]

    def __eq__(self, other: object) -> bool:
        """Equality based on the rule or the values, independently of the name."""
        if not isinstance(other, NamedFunction):
            return NotImplemented
        # Based on rule/value equality independently of the name
        if self.dom is other.dom and self.cod is other.cod:
            return self._raw == other._raw
        return (
            self.dom == other.dom
            and self.cod == other.cod
            and self.values == other.values
        )

    def __hash__(self):
        return hash((self.dom, self.cod, frozenset(self.values.items())))

    ## Function analysis methods: fibers, graphs, images, etc.

    def fiber(self, y) -> NamedSet:
        """Return the pre-image of y."""
        return NamedSet(
            {x for x in self.dom if self.values[x] == y}, name=f"{self.name}⁻¹({y})"
        )

    def fibers(self) -> dict[object, NamedSet]:
        """Return the dictionary of fibers,
        i.e., the pre-images of all elements of the codomain."""
        return {y: self.fiber(y) for y in self.cod}

    def fibers_decomposition(self) -> FiniteCoproduct:
        """Fiber decomposition of the domain
        indexed over the codomain
        For f : X -> Y, return the coproduct of f⁻¹(y) for y in Y,
        which is isomorphic to X."""
        index = sorted(self.cod, key=str)
        return FiniteCoproduct([self.fiber(y) for y in index])

    def fibers_decomposition_isomorphism(self) -> Bijection:
        """Return the isomorphism between the domain and the fibers decomposition."""
        index = sorted(self.cod, key=str)
        fiber_decomp = self.fibers_decomposition()
        values_iso = {}
        for i,y in enumerate(index):
            for x in self.fiber(y):
                values_iso[x] = (i, x)
        iso = Bijection(
            dom=self.dom,
            cod=fiber_decomp,
            table=values_iso,
            name=f"iso_{self.name}_fiber_decomp",
        )
        return iso

    def image(self) -> NamedSet:
        """Return the image as a named set."""
        im = {self.values[x] for x in self.dom}
        indices = set(self._raw)
        image_enum = tuple(
            y for j, y in enumerate(self.cod.obj._enum)
            if j in indices)
        return NamedSet(im, name=f"Im({self.name})", enum=image_enum)

    def injective_part(self) -> NamedFunction:
        """
        Return the injective part of the function

            Im(f) → Y,
        
        defined by the same rule as f
        but with domain restricted to the image."""
        Im = self.image()
        values_image = {x: x for x in Im}
        mono = Injection(Im, self.cod, values_image, name=f"m_{self.name}")
        return mono

    def kernel(self) -> Pullback:
        """Return the kernel pair of the function, i.e., the pullback of f along itself.
        for f : X → Y, this is the set of pairs (x1, x2) in X x X such that f(x1) = f(x2).
        """
        return Pullback(self, self)

    def kernel_pair(self) -> tuple[NamedFunction, NamedFunction]:
        """Return the kernel pair of the function, i.e., the two projections from the pullback of f along itself."""
        ker = self.kernel()
        return (ker.proj_0(), ker.proj_1())

    def image_factorization(self) -> tuple[NamedFunction, NamedFunction]:
        ker_pair = self.kernel_pair()
        epi = Coequalizer(ker_pair[0], ker_pair[1]).projection()
        mono = Coequalizer(ker_pair[0], ker_pair[1]).universal_solution(f=self)
        return (epi.rename(f"q_{self.name}"), mono.rename(f"m_{self.name}"))

    ## Displays and comparisons

    def symbolic_repr(self) -> str:
        """
        Return a symbolic representation of the function, e.g., 

            f : X → Y 
            """
        return f"{self.name} : {self.dom.name} → {self.cod.name} "

    def graph_repr(self) -> str:
        """Return a representation of the graph of the function
        where each line is of the form "x ↦ f(x)"."""
        return "\n".join(f"\t\t{self.dom.enumeration[i]} ↦ {self.cod.enumeration[y]}" for i, y in enumerate(self._raw))

    def __repr__(self) -> str:
        """Return a representation of the function
        that includes its name, domain, codomain, and graph."""
        return self.symbolic_repr() + "\n" + self.graph_repr()

    def display(self) -> str:
        """Return a display of the function that includes its name, domain, codomain, and graph."""
        return self.__repr__()

    def __str__(self) -> str:
        return self.name

    def __lt__(self, other: NamedFunction) -> bool:
        """Order based on the name,
        independently of the rule/values, domain and codomain."""
        return self.name < other.name

    def rename(self, new_name: str) -> NamedFunction:
        """Return a new NamedFunction with the same domain, codomain and values
        but a new name."""
        return NamedFunction(
            dom=self.dom, cod=self.cod, table=self.table, name=new_name
        )

    ## Tests

    def injectivity_test(self) -> bool:
        """Test injectivity
        checking that no two distinct elements of the domain have the same image."""
        images = set()
        for x in self.dom:
            y = self.values[x]
            if y in images:
                return False
            images.add(y)
        return True

    def injectivity_test_fast(self) -> bool:
        """Fast test for injectivity
        trim redundancy in the raw list and compare."""
        return len(set(self._raw)) == len(self._raw)

    def surjectivity_test(self) -> bool:
        """Test surjectivity
        checking that every element of the codomain has a pre-image in the domain."""
        images = set(self.values.values())
        return images == self.cod.set

    def surjectivity_test_fast(self) -> bool:
        """Fast test for surjectivity
        based on the sizes of the domain and codomain."""
        return len(set(self._raw)) == len(self.cod)

    def bijectivity_test(self) -> bool:
        """Test bijectivity by checking
        both injectivity and surjectivity."""
        return self.injectivity_test() and self.surjectivity_test()

    def bijectivity_test_fast(self) -> bool:
        """Fast test for bijectivity
        based on the sizes of the domain and codomain."""
        return self.injectivity_test_fast() and self.surjectivity_test_fast()

    @classmethod
    def from_raw(cls, dom : Representable, cod : Representable, raw_liste : Sequence[int], name : str) -> NamedFunction:
        """Factory method to create a NamedFunction from a raw list of indices
        corresponding to the images of the elements of the domain in the codomain.
        The raw list must have the same length as the domain and contain indices
        that are valid for the codomain."""
        if len(raw_liste) != len(dom):
            raise ValueError("The raw list must have the same length as the domain.")
        if any(type(j) is not int or not 0 <= j < len(cod) for j in raw_liste):
            raise ValueError("Invalid codomain index")
        values = {x: cod.unrank(i) for x, i in zip(dom.enumeration, raw_liste)}
        return cls(dom=dom, cod=cod, table=values, name=name)

    @classmethod
    def from_index(cls, dom : Representable, cod : Representable, index : int, name : str) -> NamedFunction:
        """Factory method to create a NamedFunction from an index
        corresponding to the images of the elements of the domain in the codomain.
        The index must be valid for the number of functions from dom to cod."""
        return cls.from_raw(dom=dom, 
                            cod=cod, 
                            raw_liste=rustic.decode_function(len(dom), len(cod), index), 
                            name=name or f"f_{{{index}}}")


def name_by_rule(
    dom: Representable, cod: Representable, table: Mapping | Callable
) -> NamedFunction:
    """Generate a name for a function based on its rule or values."""
    if isinstance(table, Mapping):
        values = dict(table)
    elif callable(table):
        values = {x: table(x) for x in dom}
    rule_repr = ", ".join(f"{x} ↦ {y}" for x, y in values.items())
    name = f"f_{{{rule_repr}}}"
    return NamedFunction(dom=dom, cod=cod, table=values, name=name)


## -- Compositionality --


def chaines(D: set, n: int) -> set[tuple[NamedFunction, ...]]:
    """D: set of named functions
    set of composable chains of length n,
    i.e., tuples (f_0, ..., f_{n-1})
    such that for all i

        cod(f_i) = dom(f_{i+1}) 
    """
    chaines_brutes = it.product(D, repeat=n)
    chaines = set()
    for chaine in chaines_brutes:
        if all(chaine[i].cod == chaine[i + 1].dom for i in range(len(chaine) - 1)):
            chaines.add(chaine)
    return chaines


def composition(f: NamedFunction, g: NamedFunction) -> NamedFunction:
    """Return the composition g ∘ f, defined by (g ∘ f)(x) = g(f(x)).
    Beware of the order of composition:
    composition(f,g) means "apply f first, then g".
    """
    if f.cod != g.dom:
        raise ValueError("Not composable functions.")
    if f.cod is g.dom or f.cod.enumeration == g.dom.enumeration:
        composite = tuple(g._raw[j] for j in f._raw)
        return NamedFunction.from_raw(dom=f.dom, cod=g.cod, raw_liste=composite, name=f"{g.name} ∘ {f.name}")
    return NamedFunction(
        dom=f.dom,
        cod=g.cod,
        table={x: g(f(x)) for x in f.dom},
        name=f"{g.name} ∘ {f.name}",
    )


def composition_chaine(chaine: Sequence[NamedFunction]) -> NamedFunction:
    """
    Return the composition of a chain of composable functions, defined by 

        (f_{n-1} ∘ ... ∘ f_0)(x) = f_{n-1}(...(f_0(x))...)
    
    """
    chaine_tuple: tuple[NamedFunction, ...] = tuple(chaine)
    if len(chaine_tuple) == 0:
        raise ValueError("The chain is empty.")
    result = chaine_tuple[0]
    for f in chaine_tuple[1:]:
        result = composition(result, f)
    return result


def parallel_chains(D: set, n: int) -> set[tuple[NamedFunction, NamedFunction]]:
    """Return the set of pairs of distinct composable chains of length n
    that have the same domain and codomain."""
    ch = chaines(D, n)
    composees = [(chaine, composition_chaine(chaine)) for chaine in ch]
    paires = set()
    for i, (chaine_0, comp_0) in enumerate(composees):
        for chaine_1, comp_1 in composees[i + 1 :]:
            if (
                chaine_0 != chaine_1
                and comp_0.dom == comp_1.dom
                and comp_0.cod == comp_1.cod
            ):
                paires.add((comp_0, comp_1))
    return paires


def equalities(D: set, n: int) -> set[str]:
    """Return the set of equalities between distinct composable chains of length n."""
    equalities = set()
    for paire in parallel_chains(D, n):
        f0 = paire[0]
        f1 = paire[1]
        if f0 == f1:  # equality of functions (based on rules or values)
            equalities.add(f"{f0.symbolic_repr()} = {f1.symbolic_repr()}")
    return equalities

## -- Auxiliary use of rustic decoding --

def decode_function_raw(dom: Representable, cod: Representable, index: int) -> tuple[int]:
    """Decode a function from its rustic index to a raw list of codomain indices."""
    return tuple(rustic.decode_function(len(dom), len(cod), index))

def decode_function_to_dict(dom: Representable, cod: Representable, index: int) -> dict:
    """Decode a function from its rustic index to a dictionary representation."""
    raw_liste = rustic.decode_function(len(dom), len(cod), index)
    return {x: cod.unrank(i) for x, i in zip(dom.enumeration, raw_liste)}

## -- Injections --


class Injection(NamedFunction):
    """An injection is a function that is injective,
    hence the class of injections inherits from NamedFunction
    and adds a test for injectivity in the constructor."""

    def __init__(
        self,
        dom: Representable,
        cod: Representable,
        table: Mapping | Callable,
        name=None,
    ):
        super().__init__(dom, cod, table=table, name=name)
        if not self.injectivity_test_fast():
            paires = {
                (x, x2)
                for x in self.dom
                for x2 in self.dom
                if x != x2 and self.values[x] == self.values[x2]
            }
            raise ValueError(
                f"{self.name} is not injective: the elements {paires} have the same image."
            )
        try:
            self.rustic_code_injection = rustic.encode_injection(len(self.cod), self._raw)
        except OverflowError:
            self.rustic_code_injection = None

    def symbolic_repr(self) -> str:
        """
        Return a symbolic representation of the injection, e.g., 
            
            f : X ↣ Y

        """
        return f"{self.name} : {self.dom.name} ↣ {self.cod.name} "

    @classmethod
    def from_function(cls, f: NamedFunction) -> "Injection":
        """Factory method to create an injection from a function that has tested injective"""
        return cls(f.dom, f.cod, f.values, name=f"m_{f.name}")

    @classmethod
    def from_raw(cls, dom: Representable, cod: Representable, raw_liste: Sequence[int], name: str) -> "Injection":
        f = NamedFunction.from_raw(dom, cod, raw_liste, name)
        return cls(f.dom, f.cod, f.values, name=name)

    @classmethod
    def from_injection_index(cls, dom: Representable, cod: Representable, index: int, name: str = None) -> "Injection":
        return cls.from_raw(dom=dom, 
                            cod=cod, 
                            raw_liste=rustic.decode_injection(len(dom), len(cod), index), 
                            name=name or f"m_{{{index}}}")

def decode_injection_raw(dom: Representable, cod: Representable, index: int) -> tuple[int]:
    """Decode an injection from its rustic index to a raw list of codomain indices."""
    return tuple(rustic.decode_injection(len(dom), len(cod), index))

def decode_injection_to_dict(dom: Representable, cod: Representable, index: int) -> dict:
    """Decode an injection from its rustic index to a dictionary representation."""
    raw_liste = rustic.decode_injection(len(dom), len(cod), index)
    return {x: cod.unrank(i) for x, i in zip(dom.enumeration, raw_liste)}


## -- Surjections --


class Surjection(NamedFunction):
    """A surjection is a function that is surjective,
    hence the class of surjections inherits from NamedFunction
    and adds a test for surjectivity in the constructor."""

    def __init__(
        self,
        dom: Representable,
        cod: Representable,
        table: Mapping | Callable,
        name=None,
    ):
        super().__init__(dom, cod, table=table, name=name)
        if not self.surjectivity_test_fast():
            Complement_image = self.cod.obj.set - self.image().set
            raise ValueError(
                f"{self.name} is not surjective: the image does not cover {Complement_image}"
            )
        try:
            self.rustic_code_surjection = rustic.encode_surjection(
                len(self.dom), len(self.cod), self._raw
            )
        except OverflowError:
            self.rustic_code_surjection = None

    def fibers_raw(self) -> list[list[int]]:
        """Return the fibers as domain ranks, ordered by codomain rank."""
        fibers = [[] for _ in self.cod.enumeration]
        for domain_rank, codomain_rank in enumerate(self._raw):
            fibers[codomain_rank].append(domain_rank)
        return fibers

    def symbolic_repr(self) -> str:
        """
        Return a symbolic representation of the surjection, e.g.,
          
            f : X ↠ Y

        """
        return f"{self.name} : {self.dom.name} ↠ {self.cod.name} "

    @classmethod
    def from_function(cls, f: NamedFunction) -> "Surjection":
        """Factory method to create a surjection from a function that has tested surjective"""
        return cls(f.dom, f.cod, f.values, name=f.name)


    @classmethod
    def from_raw(cls, dom: Representable, cod: Representable, raw_liste: Sequence[int], name: str) -> "Surjection":
        f = NamedFunction.from_raw(dom, cod, raw_liste, name)
        return cls(f.dom, f.cod, f.values, name=name)

    @classmethod
    def from_surjection_index(cls, dom: Representable, cod: Representable, index: int, name: str = None) -> "Surjection":
        return cls.from_raw(dom=dom, 
                            cod=cod, 
                            raw_liste=rustic.decode_surjection(len(dom), len(cod), index), 
                            name=name or f"m_{{{index}}}")
    

    def sections(self) -> NamedSet:
        """
        Return the set of sections of the surjection,
        i.e., the functions  

            s : cod -> dom 

        such that 

            f ∘ s = id_cod
            
        """
        sections = {
            self.section_from_index(index)
            for index in range(self.number_of_sections())
        }
        return NamedSet(sections, f"Γ({self.name})")

    def number_of_sections(self) -> int:
        """Return the cardinality of the set of sections of this surjection."""
        return rustic.number_of_sections(self.fibers_raw())

    def section_from_index(self, index: int) -> NamedFunction:
        """Decode the locally indexed section with code ``index``."""
        raw_section = rustic.decode_section(self.fibers_raw(), index)
        return NamedFunction.from_raw(
            dom=self.cod,
            cod=self.dom,
            raw_liste=raw_section,
            name=f"s_{index}",
        )

    def section_index(self, section: NamedFunction) -> int:
        """Encode one of this surjection's sections by its local index."""
        if section.dom != self.cod or section.cod != self.dom:
            raise ValueError("The function has incompatible domain or codomain.")
        return rustic.encode_section(self.fibers_raw(), section._raw)

    def section_code_to_injection_code(self, section_code: int) -> int:
        """Convert a local section code to its global injection code."""
        return rustic.section_code_to_injection_code(
            self.fibers_raw(), section_code
        )

def decode_surjection_raw(dom: Representable, cod: Representable, index: int) -> tuple[int]:
    """Decode a surjection from its rustic index to a raw list of codomain indices."""
    return tuple(rustic.decode_surjection(len(dom), len(cod), index))

def decode_surjection_to_dict(dom: Representable, cod: Representable, index: int) -> dict:
    """Decode a surjection from its rustic index to a dictionary representation."""
    raw_liste = rustic.decode_surjection(len(dom), len(cod), index)
    return {x: cod.unrank(i) for x, i in zip(dom.enumeration, raw_liste)}


## -- Bijections --


class Bijection(NamedFunction):
    """A bijection is a function that is bijective,
    hence the class of bijections inherits from NamedFunction
    and adds a test for bijectivity in the constructor."""

    def __init__(
        self,
        dom: Representable,
        cod: Representable,
        table: Mapping | Callable,
        name: str,
    ):
        if len(dom) != len(cod):
            raise ValueError("dom et cod doivent être de même taille.")
        super().__init__(dom=dom, cod=cod, table=table, name=name)
        if not self.bijectivity_test_fast():
            raise ValueError(f"{self.name} n'est pas bijective")
        self.backward = {y: x for x, y in self.values.items()}
        self.inverse = NamedFunction(
            dom=self.cod, cod=self.dom, table=self.backward, name=f"{self.name}⁻¹"
        )
        try:
            self.rustic_code_bijection = rustic.encode_bijection(
                len(self.dom), len(self.cod), self._raw
            )
        except OverflowError:
            self.rustic_code_bijection = None

    def symbolic_repr(self) -> str:
        """
        Return a symbolic representation of the bijection, e.g., 

            f : X ≃ Y

        """
        return f"{self.name} : {self.dom.name} ≃ {self.cod.name} "

    @classmethod
    def from_function(cls, f: NamedFunction) -> "Bijection":
        """Factory method to create a bijection from a function that has tested bijective"""
        return cls(f.dom, f.cod, f.values, name=f.name)

    @classmethod
    def from_raw(cls, dom: Representable, 
                 cod: Representable, 
                 raw_liste: Sequence[int], 
                 name: str) -> "Bijection":
        """Factory method to create a bijection from a raw list of image ranks."""
        f = NamedFunction.from_raw(dom, cod, raw_liste, name)
        return cls(f.dom, f.cod, f.values, name=name)

    @classmethod
    def from_bijection_index(cls, 
                             dom: Representable, 
                             cod: Representable, 
                             index: int, 
                             name: str = None) -> "Bijection":
        """Factory method to create a bijection from an index with the rustic encoding."""
        return cls.from_raw(dom=dom, 
                            cod=cod, 
                            raw_liste=rustic.decode_bijection(len(dom), len(cod), index), 
                            name=name or f"m_{{{index}}}")


    def inverse_test(self) -> bool:
        """Test that the backward function is indeed the inverse of the forward function."""
        for a in self.dom:
            if self.backward[self(a)] != a:
                return False
        for b in self.cod:
            if self(self.backward[b]) != b:
                return False
        return True


## -- Homsets --


class HomSet(Construct):
    """HomSet(A, B) is the NamedSet of all functions from A to B.
    The object is only produced by a property method called on demand,
    which generates all the functions from A to B"""

    def __init__(self, A: Representable, B: Representable):
        self.A = A
        self.B = B
        self._name = f"Hom({A.name},{B.name})"
        self._obj = None

    def generate_list_tables(self) -> list[dict]:
        """Generate the table from indexes by rustic decoding"""
        return [rustic.decode_function(len(self.A), len(self.B), i) for i in range(self.card())]

    def generate(self):
        """Generate all functions from A to B as a NamedSet of NamedFunctions.
        use the rustic decoding for a quick enumeration."""
        return NamedSet.from_list(
            [NamedFunction.from_index(self.A, self.B, i, name=f"f_{i}") for i in range(self.card())],
            name=self.name,
        )

    def access(self, i: int) -> NamedFunction:
        """Access the i-th function in the homset, based on the rustic encoding."""
        if not (0 <= i < self.card()):
            raise IndexError(f"Index {i} out of range for HomSet of size {self.card()}.")
        return NamedFunction.from_index(self.A, self.B, i, name=f"f_{i}")

    def card(self) -> int:
        return len(self.B) ** len(self.A)


    @property
    def obj(self) -> NamedSet:
        """Realizes Hom[A,B] as a NamedSet."""
        if self._obj is None:
            self._obj = self.generate()
        return self._obj

    @property
    def name(self) -> str:
        """Return the name of the homset."""
        return self._name


def currying(A: Representable, B: Representable, C: Representable) -> Bijection:
    """
    Return the currying isomorphism 

     Hom(A x B, C) ≃ Hom(A, Hom(B, C))
    
    """
    hom_AB_C = HomSet(Product(A, B), C)
    hom_A_hom_BC = HomSet(A, HomSet(B, C))
    hom_B_C = HomSet(B, C)

    def to_curried(f: NamedFunction) -> NamedFunction:
        """Given f : A x B → C, 
        return the curried function 

            g : A → Hom(B, C)
         
        defined by 

            g(a)(b) = f(a, b)."""

        def curr(a):
            table = {b: f((a, b)) for b in B}
            return name_by_rule(B, C, table=table)

        table = {a: curr(a) for a in A}
        return name_by_rule(A, hom_B_C, table=table)

    return Bijection(
        dom=hom_AB_C,
        cod=hom_A_hom_BC,
        table={f: to_curried(f) for f in hom_AB_C},
        name="Currying",
    )


## -- Functoriality of homsets --

def Hom_left(f: NamedFunction, B: Representable) -> NamedFunction:
    """Given f : A → A' and B, return the function

        Hom(f, B) : Hom(A', B) → Hom(A, B)

    defined by 

        Hom(f, B)(g) = g ∘ f
    """
    A = f.dom
    A_prime = f.cod
    Hom_Ap_B = HomSet(A_prime, B)
    Hom_A_B = HomSet(A, B)
    table = {g: composition(f, g) for g in Hom_Ap_B}
    return NamedFunction(dom=Hom_Ap_B, cod=Hom_A_B, table=table, name=f"Hom({f.name}, {B.name})")

def Hom_right(A: Representable, g: NamedFunction) -> NamedFunction:
    """Given A and g : B → B', return the function

        Hom(A, g) : Hom(A, B) → Hom(A, B')

    defined by 

        Hom(A, g)(f) = g ∘ f
    """
    B = g.dom
    B_prime = g.cod
    Hom_A_B = HomSet(A, B)
    Hom_A_Bp = HomSet(A, B_prime)
    table = {f: composition(f, g) for f in Hom_A_B}
    return NamedFunction(dom=Hom_A_B, cod=Hom_A_Bp, table=table, name=f"Hom({A.name}, {g.name})")

## -- Subset of the homset set --

def Inj(A: Representable, B: Representable) -> NamedSet:
    """Return the set of injections from A to B as a NamedSet."""
    number = rustic.count_injections(len(A), len(B))
    inj = [Injection.from_injection_index(A, B, i, name=f"inj_{i}") for i in range(number)]
    return NamedSet.from_list(inj, f"Inj({A.name}, {B.name})")

def Inj_inclusion(A: Representable, B: Representable) -> Injection:
    """Return the inclusion of injections into the homset as an Injection."""
    Hom = HomSet(A, B)
    Inj_set = Inj(A, B)
    inclusion_table = {f: Hom.access(f.rustic_code) for f in Inj_set}
    return Injection(dom=Inj_set, 
                     cod=Hom, 
                     table=inclusion_table, 
                     name=f"Inj_inclusion({A.name}, {B.name})")

def Surj(A: Representable, B: Representable) -> NamedSet:
    """Return the set of surjections from A to B as a NamedSet."""
    number = rustic.count_surjections(len(A), len(B))
    surj = [Surjection.from_surjection_index(A, B, i, name=f"surj_{i}") for i in range(number)]
    return NamedSet.from_list(surj, f"Surj({A.name}, {B.name})")

def Surj_inclusion(A: Representable, B: Representable) -> Injection:
    """Return the inclusion of surjections into the homset as an Injection."""
    Hom = HomSet(A, B)
    Surj_set = Surj(A, B)
    inclusion_table = {f: Hom.access(f.rustic_code) for f in Surj_set}
    return Injection(dom=Surj_set, 
                     cod=Hom, 
                     table=inclusion_table, 
                     name=f"Surj_inclusion({A.name}, {B.name})")

def Bij(A: Representable, B: Representable) -> NamedSet:
    """Return the set of bijections from A to B as a NamedSet."""
    number = rustic.bijection_number(len(A), len(B))
    bij = [Bijection.from_bijection_index(A, B, i, name=f"bij_{i}") for i in range(number)]
    return NamedSet.from_list(bij, f"Bij({A.name}, {B.name})")

def Bij_inclusion(A: Representable, B: Representable) -> Injection:
    """Return the inclusion of bijections into the homset as an Injection."""
    Hom = HomSet(A, B)
    Bij_set = Bij(A, B)
    inclusion_table = {f: Hom.access(f.rustic_code) for f in Bij_set}
    return Injection(dom=Bij_set, 
                     cod=Hom,   
                     table=inclusion_table,
                     name=f"Bij_inclusion({A.name}, {B.name})")


##########################################################################################

# Limits and colimits of sets

"""Each limit or colimit is represented by a class
with the diagrammatic arities represented by attributes,
(co)projections and universal solutions as methods,
as well as detection and isomorphism methods for the special cases of terminal and initial objects.
The instance of the limit or colimit has an underlying NamedSet in the attribute obj"""


## -- Terminal object --


class Terminal(Construct):
    """The terminal object in the category of sets is the singleton set, which we denote by 1."""

    def __init__(self):
        terminal_enum = ("*",)
        self.elements = set(terminal_enum)
        self._name = "1"
        self._obj = NamedSet(self.elements, self.name, enum=terminal_enum)

    def unique_map(self, A: Representable) -> NamedFunction:
        """
        Return the unique function 

            !_A : A → 1

        defined by f(a) = *."""
        regle_unique = lambda x: "*"
        unique = NamedFunction(dom=A, cod=self, table=regle_unique, name=f"!_{A.name}")
        if A.obj.set == set():
            return unique
        elif A.obj.set == {"*"}:
            return self.identity
        else:
            return Surjection.from_function(unique)


def terminal_detection(X: Representable) -> bool:
    """Return True if X is a terminal object,
    i.e., if there is a unique function from any set to X."""
    return len(X.obj) == 1


def terminal_isomorphism(X: Representable) -> Bijection:
    """Return the unique isomorphism from X ≃ 1
    if X has exactly one element, otherwise raise an error."""
    if not terminal_detection(X):
        raise ValueError(f"{X.name} n'est pas un objet terminal.")
    regle_iso = lambda x: "*"
    return Bijection(dom=X, cod=Terminal(), table=regle_iso, name=f"iso_{X.name}_to_1")


## -- Binary product --


class Product(Construct):
    """Return the binary product X0 x X1 of two named sets X0 and X1, that is the set of pairs
    (a, b) with a in X0 and b in X1, 
    endowed with the two projections p0 and p1 and the universal solution property.
    The enumeration of the product is the lexicographic order of the enumerations of X0 and X1."""

    def __init__(self, X0: Representable, X1: Representable):
        self.X0 = X0
        self.X1 = X1
        product_enum = tuple(
            (a, b)
            for a in X0.enumeration
            for b in X1.enumeration
        )

        self.elements = set(product_enum)

        if isinstance(self.X0, Product | Coproduct) and not isinstance(
            self.X1, Product | Coproduct
        ):
            self._name = f"({X0.name}) x {X1.name}"
        elif isinstance(self.X1, Product | Coproduct) and not isinstance(
            self.X0, Product | Coproduct
        ):
            self._name = f"{X0.name} x ({X1.name})"
        elif isinstance(self.X0, Product | Coproduct) and isinstance(
            self.X1, Product | Coproduct
        ):
            self._name = f"({X0.name}) x ({X1.name})"
        else:
            self._name = f"{X0.name} x {X1.name}"

        self._obj = NamedSet(
            self.elements,
            self.name,
            enum=product_enum,
        )    

    def proj_0(self) -> Surjection:
        """
        Return the first projection 

            proj_0 : X0 x X1 → X0

        defined by 

            p0(a, b) = a
        """
        proj_0 = lambda pair: pair[0]
        return Surjection(dom=self, cod=self.X0, table=proj_0, name=f"p0_{self.name}")

    def proj_1(self) -> Surjection:
        """
        Return the second projection 

            proj_1 : X0 x X1 → X1 

        defined by 

            p1(a, b) = b
        """
        proj_1 = lambda pair: pair[1]
        return Surjection(dom=self, cod=self.X1, table=proj_1, name=f"p1_{self.name}")

    def universal_solution(self, f0: NamedFunction, f1: NamedFunction) -> NamedFunction:
        """Suppose f0 : C → X0 and f1 : C → X1 are two functions with the same domain C.
        Returns the unique function 

            u : C → X0 x X1 

        such that 

            p0 ∘ u = f0 and p1 ∘ u = f1
        """
        if f0.cod != self.X0 or f1.cod != self.X1:
            raise ValueError(
                "The codomains of f0 and f1 must be X0 and X1 respectively."
            )
        if f0.dom != f1.dom:
            raise ValueError("The domains of f0 and f1 must be the same.")
        C = f0.dom
        regle_u = lambda x: (f0(x), f1(x))
        return NamedFunction(
            dom=C, cod=self, table=regle_u, name=f"({f0.name}, {f1.name})"
        )

    def braiding(self) -> Bijection:
        """
        Return the braiding isomorphism 

            β : X0 x X1 ≃ X1 x X0

        defined by

            β(a, b) = (b, a)
        """
        braid = Product(self.X1, self.X0)
        regle_braid = lambda pair: (pair[1], pair[0])
        return Bijection(
            dom=self,
            cod=braid,
            table=regle_braid,
            name=f"β_({self.X0.name}, {self.X1.name})",
        )


def unitor_cartesian_left(X: Representable) -> Bijection:
    """
    Return the left unitor isomorphism 

        λ : 1 x X ≃ X

    defined by 

        λ(*, x) = x
    """
    unitor = Product(Terminal(), X)
    regle_unitor = lambda pair: pair[1]
    return Bijection(dom=unitor, cod=X, table=regle_unitor, name=f"λ_{X.name}")


def unitor_cartesian_right(X: Representable) -> Bijection:
    """
    Return the right unitor isomorphism 

        ρ : X x 1 ≃ X

    defined by 

        ρ(x, *) = x
    """
    unitor = Product(X, Terminal())
    regle_unitor = lambda pair: pair[0]
    return Bijection(dom=unitor, cod=X, table=regle_unitor, name=f"ρ_{X.name}")


def associator_cartesian(
    X: Representable, Y: Representable, Z: Representable
) -> Bijection:
    """
    Return the associator isomorphism

        α : (X x Y) x Z ≃ X x (Y x Z)

    defined by 

        α((x, y), z) = (x, (y, z))
    """
    leftpair = Product(X, Y)
    prod_gauche = Product(leftpair, Z)
    rightpair = Product(Y, Z)
    prod_droite = Product(X, rightpair)
    regle_associator = lambda triple: (triple[0][0], (triple[0][1], triple[1]))
    return Bijection(
        dom=prod_gauche,
        cod=prod_droite,
        table=regle_associator,
        name=f"α_({X.name}, {Y.name}, {Z.name})",
    )

def diagonal(X: Representable) -> NamedFunction:
    """
    Return the diagonal function 

    Δ : X → X x X

    defined by 

        Δ(x) = (x, x)
    """
    product = Product(X, X)
    regle_diag = lambda x: (x, x)
    return Injection(dom=X, cod=product, table=regle_diag, name=f"Δ_{X.name}")

def product_maps(f: NamedFunction, g: NamedFunction) -> NamedFunction:
    """
    Return the product of two functions f : A0 → B0 and g : A1 → B1, defined by 

        (f x g)(a0, a1) = (f(a0), g(a1))
    """
    product_dom = Product(f.dom, g.dom)
    product_cod = Product(f.cod, g.cod)
    regle_prod = lambda pair: (f(pair[0]), g(pair[1]))
    return NamedFunction(
        dom=product_dom, cod=product_cod, table=regle_prod, name=f"{f.name} x {g.name}"
    )


def detection_binary_product(candidate: Representable) -> bool:
    """Return True if obj is isomorphic to a binary product,
    i.e., if it is a set of pairs (a, b)
    such that for any a in the first component
    and b in the second component,
    (a, b) is in the set."""
    if not isinstance(candidate, Representable):
        return False
    if isinstance(candidate, Product):
        return True
    if len(candidate) == 0:
        return True  # for any A, A x ∅ = ∅, so ∅ is a product
    if len(candidate) == 1:
        return True  # Empty product is 1

    if not all(isinstance(a, tuple) and len(a) == 2 for a in candidate):
        return False

    A0 = {a[0] for a in candidate}
    A1 = {a[1] for a in candidate}
    return all((x, y) in candidate for x in A0 for y in A1)


def decomposition_binary_product(candidate: Representable) -> Product:
    """Return the decomposition of candidate as a binary product."""
    if not detection_binary_product(candidate):
        raise ValueError("L'obj n'est pas un produit binaire.")
    if isinstance(candidate, Product):
        return candidate
    A0 = NamedSet(set(a[0] for a in candidate), f"proj_0({candidate.name})")
    A1 = NamedSet(set(a[1] for a in candidate), f"proj_1({candidate.name})")
    return Product(A0, A1)


## -- Finite product --


class FiniteProduct(Construct):
    """Take a list of NamedSet and build their finite product, '
    with projections and universal solution.
    Enumeration of the product is the lexicographic order of the enumerations of the operands."""

    def __init__(self, liste: Sequence[Representable]):
        """For a list of NamedSet (X0,..., X_{n-1}),
        Compute the finite product X0 x ... x X_{n-1}
        which is the set of n-tuples (a0,..., a_{n-1}) with ai in Xi."""
        self.operands = tuple(liste)
        self.arity = len(self.operands)
        product_enum = tuple(
            it.product(*(operand.enumeration for operand in self.operands))
        )
        self.elements = set(product_enum)

        self._name = " x ".join(operand.name for operand in self.operands)
        self._obj = NamedSet(self.elements, self.name, enum=product_enum)

    def proj(self, i: int) -> Surjection:
        """Return the i-th projection from the product to the i-th operand,
        defined by p_i(a_0, ..., a_{n-1}) = a_i."""
        if i < 0 or i >= self.arity:
            raise ValueError("Index is out of bounds.")
        regle_proj_i = lambda tuple: tuple[i]
        return Surjection(
            dom=self,
            cod=self.operands[i],
            table=regle_proj_i,
            name=f"p{i}_{self.name}",
        )

    def universal_solution(self, list_f: list[NamedFunction]) -> NamedFunction:
        """Suppose list_f is a list of functions f_i : C -> A_i
        where A_i are the operands of the product.
        Return the unique function 
         u : C -> A_0 x ... x A_{n-1}
        such that 
         p_i ∘ u = f_i for all i."""
        if len(list_f) != self.arity:
            raise ValueError(
                "The number of functions must match the arity of the product."
            )
        for i in range(self.arity):
            if list_f[i].cod != self.operands[i]:
                raise ValueError(
                    f"The codomain of function {i} must match operand {i}."
                )
            C = list_f[0].dom
            if list_f[i].dom != C:
                raise ValueError(f"The domain of function {i} must be equal to C.")
        regle_u = lambda x: tuple(list_f[i](x) for i in range(self.arity))
        return NamedFunction(
            dom=C,
            cod=self,
            table=regle_u,
            name=f"⟨{', '.join(f.name for f in list_f)}⟩",
        )


def finite_product_maps(list_f: list[NamedFunction]) -> NamedFunction:
    """Return the product of a list of functions f_i : A_i → B_i, defined by 

        (f_0 x ... x f_{n-1})(a_0, ..., a_{n-1}) = (f_0(a_0), ..., f_{n-1}(a_{n-1})).
    """
    arity = len(list_f)
    product_dom = FiniteProduct([f.dom for f in list_f])
    product_cod = FiniteProduct([f.cod for f in list_f])
    regle_prod = lambda tuplette: tuple(list_f[i](tuplette[i]) for i in range(arity))
    return NamedFunction(
        dom=product_dom,
        cod=product_cod,
        table=regle_prod,
        name=f" x ".join(f.name for f in list_f),
    )


def iterated_binary_product(liste: list[Representable]) -> Representable:
    """Return the iterated binary product of a list of NamedSet (((A_0 x A_1) x A_2) x ... ) x A_{n-1}
    """
    if len(liste) == 0:
        return Terminal()
    elif len(liste) == 1:
        return liste[0]
    else:
        return Product(iterated_binary_product(liste[:-1]), liste[-1])


def finite_product_detection(candidate: Representable) -> bool:
    """Return True if obj is isomorphic to a finite product"""
    if not isinstance(candidate, Representable):
        return False
    if isinstance(candidate, FiniteProduct):
        return True
    elems = list(candidate)
    if len(elems) == 0:
        return True  # for any A, A x ∅ = ∅, so ∅ is a product
    if len(elems) == 1:
        return True  # Empty product is 1
    if not all(isinstance(a, tuple) for a in elems):
        return False
    arity = len(elems[0])
    if not all(len(a) == arity for a in elems):
        return False
    facteurs = [{a[i] for a in elems} for i in range(arity)]
    attendu = set(it.product(*facteurs))
    return candidate.obj.set == attendu


def finite_product_decomposition(candidate: Representable) -> FiniteProduct:
    """Return the decomposition of candidate as a finite product"""
    if not finite_product_detection(candidate):
        raise ValueError("L'obj n'est pas un produit fini.")
    if isinstance(candidate, FiniteProduct):
        return candidate
    arity = len(next(iter(candidate)))
    operands = []
    for i in range(arity):
        operand_i = NamedSet(
            set(a[i] for a in candidate), f"proj_{i}({candidate.name})"
        )
        operands.append(operand_i)
    return FiniteProduct(operands)


# TODO: add reordering isomorphisms (braidings, associators, unitors) for finite products
# via the action of the permutation group

## -- Pullbacks --


class Pullback(Construct):
    def __init__(self, f0: NamedFunction, f1: NamedFunction):
        """The pullback of f0 : X0 → X2 and f1 : X1 → X2
        This is the set of pairs (x, y) in X0 x X1
        such that f0(x) = f1(y).
        Enumeration is the restriction of the lexicographic order of X0 x X1 to the pullback."""
        self.f0 = f0
        self.f1 = f1
        self.test_codomain()
        self.X0 = f0.dom
        self.X1 = f1.dom
        self.X2 = f0.cod
        pullback_enum = tuple((x, y) for x in self.X0 for y in self.X1 if f0(x) == f1(y))
        self.elements = set(pullback_enum)
        self._name = f"Pullback of ({f0.name}, {f1.name})"
        self._obj = NamedSet(self.elements, self.name, enum=pullback_enum)
        self.fibers = {((x, y), f0(x)) for (x, y) in self.elements}

    def test_codomain(self):
        """Ensure that the codomains of f0 and f1 are the same,
        otherwise the pullback is not defined."""
        cod_f0 = self.f0.cod
        cod_f1 = self.f1.cod
        if cod_f0 != cod_f1:
            raise ValueError(
                f"The codomains of the functions must be equal: {cod_f0.name} != {cod_f1.name}"
            )

    def proj_0(self):
        """Return the first projection

            p0 : X0 x_{X2} X1 → X0

        defined by 

            p0(x, y) = x."""
        return NamedFunction(
            dom=self,
            cod=self.X0,
            table=lambda pair: pair[0],
            name=f"{self.f0.name}*{self.f1.name}",
        )

    def proj_1(self):
        """Return the second projection

            p1 : X0 x_{X2} X1 → X1

        defined by 

            p1(x, y) = y."""
        return NamedFunction(
            dom=self,
            cod=self.X1,
            table=lambda pair: pair[1],
            name=f"{self.f1.name}*{self.f0.name}",
        )

    def universal_solution(self, f: NamedFunction, g: NamedFunction) -> NamedFunction:
        """
        For f : C → X0 and g : C → X1 such that f0 ∘ f = f1 ∘ g,
        return the unique function u : C → X0 x_{X2} X1
        such that 

            p0 ∘ u = f and p1 ∘ u = g

        defined as 

            u(x) = (f(x), g(x)) for all x in C.
        """
        if f.cod != self.X0 or g.cod != self.X1:
            raise ValueError(
                "The codomains of the functions must match the projections."
            )
        if f.dom != g.dom:
            raise ValueError("The domains of the functions must be the same.")
        C = f.dom
        if composition(f, self.f0) != composition(g, self.f1):  # teste f0 ∘ f = f1 ∘ g
            raise ValueError(
                "The compositions of the functions with the projections must be equal."
            )
        regle_u = lambda x: (f(x), g(x))
        return NamedFunction(
            dom=C, cod=self, table=regle_u, name=f"⟨{f.name}, {g.name}⟩"
        )

    def braiding(self):
        """Return the braiding isomorphism of the pullback

            β : X0 x_{X2} X1 ≃ X1 x_{X2} X0

        defined by 
            i(x, y) = (y, x)
        """
        braid = Pullback(self.f1, self.f0)
        regle_braid = lambda pair: (pair[1], pair[0])
        return Bijection(
            dom=self,
            cod=braid,
            table=regle_braid,
            name=f"i_({self.f0.name}, {self.f1.name})",
        )


## -- Equalizers  --


class Equalizer(Construct):
    """The equalizer of two functions f0 : X → Y and g0 : X → Y
    This is the set of elements x in X such that f0(x) = g0(x)."""

    def __init__(self, f0: NamedFunction, g0: NamedFunction):
        """The equalizer of f0 : X → Y and g0 : X → Y is the set of elements
        x in X such that f0(x) = g0(x).
        Order is restricted from the order of X."""
        if f0.dom != g0.dom or f0.cod != g0.cod:
            raise ValueError("The functions must have the same domain and codomain.")
        self.f = f0
        self.g = g0
        equalizer_enum = tuple(x for x in f0.dom if f0(x) == g0(x))
        self.elements = set(equalizer_enum)
        self._name = f"Eq({f0.name}, {g0.name})"
        self._obj = NamedSet(self.elements, self.name, enum=equalizer_enum)

    def inclusion(self) -> NamedFunction:
        """Return the inclusion map 
        
            i : eq(f0,f1) ↣ X 
        """
        incl_regle = lambda x: x
        incl = NamedFunction(
            dom=self, cod=self.f.dom, table=incl_regle, name=f"incl_{self.name}"
        )
        return incl

    def universal_solution(self, h: NamedFunction) -> NamedFunction:
        """Suppose h : C → X is a function such that f0 ∘ h = g0 ∘ h.
        Return the unique function u : C → Eq(f0, g0) such that incl ∘ u = h."""
        if h.cod != self.f.dom:
            raise ValueError("The codomain of h must be the domain of f0 and g0.")
        if composition(h, self.f) != composition(h, self.g):
            raise ValueError("The compositions of h with f0 and g0 must be equal.")
        regle_u = lambda x: h(x)
        return NamedFunction(dom=h.dom, cod=self, table=regle_u, name=f"u_{h.name}")


########################################################
# Colimits

## -- Initial object --


class Initial(Construct):
    """The initial object in the category of sets
    This is the empty set, which we denote by ∅."""

    def __init__(self):
        self.elements = set()
        self._name = "∅"
        empty_enum = tuple()
        self._obj = NamedSet(self.elements, self.name, enum=empty_enum)

    def unique_map(self, A: Representable) -> NamedFunction:
        """Return the unique function

            !_A : ∅ → A

        which is the empty function since there are no elements in the domain."""
        regle_unique = lambda x: None  # fonction vide
        return NamedFunction(dom=self, cod=A, table=regle_unique, name=f"!_{A.name}")


def initial_detection(X: Representable) -> bool:
    """Return True if X is an initial object, i.e.,
    if the underlying set of X is empty"""
    return len(X.obj) == 0


def initial_isomorphism(X: Representable) -> Bijection:
    """Return the unique isomorphism from ∅ ≃ X if X is empty,
    otherwise raise an error."""
    if not initial_detection(X):
        raise ValueError(f"{X.name} n'est pas un objet initial.")
    regle_iso = lambda x: None  # fonction vide
    return Bijection(dom=Initial(), cod=X, table=regle_iso, name=f"iso_∅_to_{X.name}")


def map_to_initial(X: Representable):
    """Test if there is a function X → ∅,
    which is the case if and only if X is empty."""
    if len(HomSet(X, Initial().obj)) == 1:
        assert initial_detection(X)


## -- Coproduct --


class Coproduct(Construct):
    """The coproduct of two named sets X0 and X1
    This is the disjoint union of X0 and X1,
    which can be represented as the set of pairs :

    - (0, a) with a in X0
    - (1, b) with b in X1

    where the first component indicates the origin of the element.
    Obvious enumeration."""

    def __init__(self, X0: Representable, X1: Representable):
        self.X0 = X0
        self.X1 = X1
        coproduct_enum = tuple((0, a) for a in X0.enumeration) + tuple(
            (1, b) for b in X1.enumeration
        )
        self.elements = set(coproduct_enum)
        if isinstance(self.X0, Product | Coproduct) and not isinstance(
            self.X1, Product | Coproduct
        ):
            self._name = f"({X0.name}) + {X1.name}"
        elif isinstance(self.X1, Product | Coproduct) and not isinstance(
            self.X0, Product | Coproduct
        ):
            self._name = f"{X0.name} + ({X1.name})"
        elif isinstance(self.X0, Product | Coproduct) and isinstance(
            self.X1, Product | Coproduct
        ):
            self._name = f"({X0.name}) + ({X1.name})"
        else:
            self._name = f"{X0.name} + {X1.name}"
        self._obj = NamedSet(self.elements, self.name, enum=coproduct_enum)

    def inj_0(self) -> Injection:
        """Return the first injection 

            q0 : X0 ↣ X0 + X1,

        defined by 

            q0(a) = (0, a)
        """
        inj_0_regle = lambda a: (0, a)
        inj_0 = Injection(
            dom=self.X0, cod=self, table=inj_0_regle, name=f"q0_{self.name}"
        )
        return inj_0

    def inj_1(self) -> Injection:
        """Return the second injection

            q1 : X1 ↣ X0 + X1

        defined by

            q1(b) = (1, b)
        """
        inj_1_regle = lambda b: (1, b)
        inj_1 = Injection(
            dom=self.X1, cod=self, table=inj_1_regle, name=f"q1_{self.name}"
        )
        return inj_1

    def universal_solution(self, f0: NamedFunction, f1: NamedFunction) -> NamedFunction:
        """For f0 : X0 → C and f1 : X1 → C with same codomain C.
        Returns the unique function 
        
            u : X0 + X1 → C

        such that 
        
            u ∘ q0 = f0 and u ∘ q1 = f1
        """
        if f0.cod != f1.cod:
            raise ValueError("The codomains must be equal.")
        if f0.dom != self.X0 or f1.dom != self.X1:
            raise ValueError("The domains must be X0 and X1.")
        C = f0.cod
        regle_u = lambda pair: f0(pair[1]) if pair[0] == 0 else f1(pair[1])
        return NamedFunction(
            dom=self, cod=C, table=regle_u, name=f"⟨{f0.name}, {f1.name}⟩"
        )

    def braiding(self) -> Bijection:
        """
        Returns the braiding isomorphism 

            β : X0 + X1 ≃ X1 + X0

        defined by

            - β(0, a) = (1, a)
            - β(1, b) = (0, b)
        """
        braid = Coproduct(self.X1, self.X0)
        regle_braid = lambda pair: (1, pair[1]) if pair[0] == 0 else (0, pair[1])
        return Bijection(
            dom=self,
            cod=braid,
            table=regle_braid,
            name=f"β_({self.X0.name}, {self.X1.name})",
        )


def unitor_coproduct_left(X: Representable) -> Bijection:
    """
    Return the left unitor isomorphism 

        λ : ∅ + X ≃ X

    defined by 

        λ(1, x) = x
    """
    unitor = Coproduct(Initial(), X)
    regle_unitor = lambda pair: pair[1]
    return Bijection(dom=unitor, cod=X, table=regle_unitor, name=f"λ_{X.name}")


def unitor_coproduct_right(X: Representable) -> Bijection:
    """
    Return the right unitor isomorphism 

        ρ : X + ∅ ≃ X

    defined by 

        ρ(0, x) = x
    """
    unitor = Coproduct(X, Initial())
    regle_unitor = lambda pair: pair[1]
    return Bijection(dom=unitor, cod=X, table=regle_unitor, name=f"ρ_{X.name}")


def associator_coproduct(
    X: Representable, Y: Representable, Z: Representable
) -> Bijection:
    """
    Return the associator isomorphism

        α :(X + Y) + Z ≃ X + (Y + Z)

    defined by:

        - α((0, (0, x))) = (0, x)
        - α((0, (1, y))) = (1, (0, y))
        - α((1, z)) = (1, (1, z))
    """
    left_pair = Coproduct(X, Y)
    coprod_gauche = Coproduct(left_pair, Z)
    right_pair = Coproduct(Y, Z)
    coprod_droite = Coproduct(X, right_pair)
    regle_associator = (
        {(0, (0, x)): (0, x) for x in X}
        | {(0, (1, y)): (1, (0, y)) for y in Y}
        | {(1, z): (1, (1, z)) for z in Z}
    )
    return Bijection(
        dom=coprod_gauche,
        cod=coprod_droite,
        table=regle_associator,
        name=f"α_({X.name}, {Y.name}, {Z.name})",
    )


def codiagonal(X: Representable) -> NamedFunction:
    """
    Return the codiagonal function 

        ∇ : X + X → X

    defined by
        
        - ∇(0, x) = x
        - ∇(1, x) = x
    """
    coprod = Coproduct(X, X)
    regle_codiag = lambda pair: pair[1]
    return Surjection(
        dom=coprod, cod=X, table=regle_codiag, name=f"∇_{X.name}"
    )


def coproduct_maps(f: NamedFunction, g: NamedFunction):
    """
    Return the coproduct of two functions f : A0 → B0 and g : A1 → B1,
    defined by

        - (f + g)(0, a) = (0, f(a))
        - (f + g)(1, b) = (1, g(b))
    """
    coproduct_dom = Coproduct(f.dom, g.dom)
    coproduct_cod = Coproduct(f.cod, g.cod)
    regle_coprod = lambda pair: ((0, f(pair[1])) if pair[0] == 0 else (1, g(pair[1])))
    return NamedFunction(
        dom=coproduct_dom,
        cod=coproduct_cod,
        table=regle_coprod,
        name=f"{f.name} + {g.name}",
    )


## -- Distributivity of product over coproduct --


def left_distributivity_isomorphism(
    ext: Representable, paire: tuple[Representable, Representable]
) -> Bijection:
    """
    Return the distributivity isomorphism

        d : X x (Y + Z) ≃ (X x Y) + (X x Z)
    
    defined by:

        - d(x, (0, y)) = (0, (x, y))
        - d(x, (1, z)) = (1, (x, z))
    """
    Y, Z = paire
    X = ext
    coprod = Coproduct(Y, Z)
    prod_gauche = Product(X, coprod)
    prod_droite = Coproduct(
        Product(X, Y),
        Product(X, Z),
    )
    regle_distrib = lambda pair: (
        (0, (pair[0], pair[1][1])) if pair[1][0] == 0 else (1, (pair[0], pair[1][1]))
    )
    return Bijection(
        dom=prod_gauche,
        cod=prod_droite,
        table=regle_distrib,
        name=f"d_({X.name}, ({Y.name}, {Z.name}))",
    )


def right_distributivity_isomorphism(
    ext: Representable, paire: tuple[Representable, Representable]
) -> Bijection:
    """
    Return the distributivity isomorphism

        d : (X + Y) x Z ≃ (X x Z) + (Y x Z)

    defined by:

        - d((0, x), z) = (0, (x, z))
        - d((1, y), z) = (1, (y, z))
    """
    X, Y = paire
    Z = ext
    coprod = Coproduct(X, Y)
    prod_gauche = Product(coprod, Z)
    prod_droite = Coproduct(
        Product(X, Z),
        Product(Y, Z),
    )
    regle_distrib = lambda pair: (
        (0, (pair[0][1], pair[1])) if pair[0][0] == 0 else (1, (pair[0][1], pair[1]))
    )
    return Bijection(
        dom=prod_gauche,
        cod=prod_droite,
        table=regle_distrib,
        name=f"d_(({X.name}, {Y.name}), {Z.name})",
    )


## -- Finite coproducts --


class FiniteCoproduct(Construct):
    """
    Return the finite coproduct of a list of named sets X0, ..., X_{n-1}, 
    whose elements are pairs (i, x) with i in {0, ..., n-1} and x in X_i."""

    def __init__(self, liste: Sequence[Representable]):
        self.arity = len(liste)
        self.operands = liste
        coproduct_enum = tuple((i, x) for i, operand in enumerate(liste) for x in operand)
        self.elements = set(coproduct_enum)
        self._name = " + ".join(operand.name for operand in liste)
        self._obj = NamedSet(self.elements, self.name, enum=coproduct_enum)

    def inj(self, i: int) -> Injection:
        """Return the i-th injection from X_i to the coproduct,
        defined by q_i(x) = (i, x)."""
        if i < 0 or i >= self.arity:
            raise ValueError("Index is out of bounds.")
        regle_inj_i = lambda x: (i, x)
        return Injection(
            dom=self.operands[i],
            cod=self,
            table=regle_inj_i,
            name=f"q{i}_{self.name}",
        )

    def universal_solution(self, list_f: list[NamedFunction]) -> NamedFunction:
        """
        Suppose list_f is a list of functions f_i : X_i -> C
        where X_i are the operands of the coproduct.
        Return the unique function u : X_0 + ... + X_{n-1} -> C
        such that u ∘ q_i = f_i for all i.
        """
        if len(list_f) != self.arity:
            raise ValueError(
                "The number of functions must match the arity of the coproduct."
            )
        C = list_f[0].cod
        for i in range(self.arity):
            if list_f[i].cod != C:
                raise ValueError(f"The codomain of function {i} must match.")
        regle_u = lambda pair: list_f[pair[0]](pair[1])
        return NamedFunction(
            dom=self,
            cod=C,
            table=regle_u,
            name=f"[{', '.join(f.name for f in list_f)}]",
        )


def finite_coproduct_maps(list_f: list[NamedFunction]) -> NamedFunction:
    """
    Return the coproduct of a list of functions f_i : A_i → B_i, defined by 

        (f_0 + ... + f_{n-1})(i, a) = (i, f_i(a))
    """
    coproduct_dom = FiniteCoproduct([f.dom for f in list_f])
    coproduct_cod = FiniteCoproduct([f.cod for f in list_f])
    regle_coprod = lambda pair: (pair[0], list_f[pair[0]](pair[1]))
    return NamedFunction(
        dom=coproduct_dom,
        cod=coproduct_cod,
        table=regle_coprod,
        name=f" + ".join(f.name for f in list_f),
    )


# TODO: add reordering isomorphisms (braidings, associators, unitors) for finite products
# via the action of the permutation group

## -- Pushouts --


class Pushout(Construct):
    """
    Return the pushout of f0 : X2 → X0 and f1 : X2 → X1
    as the quotient of the coproduct X0 + X1
    by the equivalence relation generated by the gluing conditions    
     (0, f0(x)) ~ (1, f1(x)) for all x in X2.
    constructed from a setoid on the coproduct,
    where the equivalence relation is generated by the gluing conditions.
    Enumeration induced from the enumeration of the coproduct and setoid."""

    def __init__(self, f0: NamedFunction, f1: NamedFunction):
        self.f0 = f0
        self.f1 = f1
        self.test_domain()
        self.X0 = f0.cod
        self.X1 = f1.cod
        self.X2 = f0.dom
        self.prequotient = Coproduct(self.X0, self.X1)
        self.gluing_conditions = {((0, self.f0(x)), (1, self.f1(x))) for x in self.X2}
        self.pushout_setoid = Setoid(
            self.prequotient,
            self.gluing_conditions,
        )
        self._name = f"Pushout of ({f0.name}, {f1.name})"
        self._obj = self.pushout_setoid.obj

    def test_domain(self) -> None:
        """Ensure that the domains of f0 and f1 are the same,
        otherwise the pushout is not defined."""
        if self.f0.dom != self.f1.dom:
            raise ValueError("The domains of the functions must be equal.")

    def _class_of_left(self, x):
        return self.pushout_setoid.class_of((0, x))

    def _class_of_right(self, y):
        return self.pushout_setoid.class_of((1, y))

    def inj_0(self) -> NamedFunction:
        """
        Return the injection 

            q_0 : X0 → X0 +_{X2} X1

        defined by 

            q_0(a) = [ (0, a) ]
        """
        inj_0_regle = lambda x: self._class_of_left(x)
        return NamedFunction(
            dom=self.X0, cod=self, table=inj_0_regle, name=f"q_0_{self.name}"
        )

    def inj_1(self) -> NamedFunction:
        """
        Return the injection 

            q_1 : X1 → X0 +_{X2} X1

        defined by 

            q_1(b) = [ (1, b) ]
        """
        inj_1_regle = lambda y: self._class_of_right(y)
        return NamedFunction(
            dom=self.X1, cod=self, table=inj_1_regle, name=f"q_1_{self.name}"
        )

    def universal_solution(self, f: NamedFunction, g: NamedFunction) -> NamedFunction:
        """
        For f : X0 → C and g : X1 → C such that f ∘ f0 = g ∘ f1, returns the universal solution 

            u : X0 +_{X2} X1 → C 
        
        defined by

            - u([0, a]) = f(a)
            - u([1, b]) = g(b)
        """
        if f.dom != self.X0 or g.dom != self.X1:
            raise ValueError("The domains of the functions must be equal to X0 and X1.")
        if f.cod != g.cod:
            raise ValueError("The codomains of the functions must be equal.")
        C = f.cod
        if composition(self.f0, f) != composition(self.f1, g):
            raise ValueError(
                "The compositions of the functions with the injections must be equal."
            )

        def regle_u(equiv_class):
            representative = next(iter(equiv_class))
            return (
                f(representative[1]) if representative[0] == 0 else g(representative[1])
            )

        return NamedFunction(
            dom=self, cod=C, table=regle_u, name=f"⟨{f.name}, {g.name}⟩"
        )

    def braiding(self) -> Bijection:
        """
        Return the braiding isomorphism of the pushout,

            β : X0 +_{X2} X1 ≃ X1 +_{X2} X0

        defined by
        
            - β([(0, a)]) = [(1, a)]
            - β([(1, b)]) = [(0, b)]
        """
        braid = Pushout(self.f1, self.f0)
        regle_braid = lambda cls: (
            self._class_of_right(cls[0][1])
            if cls[0][0] == 0
            else self._class_of_left(cls[0][1])
        )
        return Bijection(
            dom=self,
            cod=braid,
            table=regle_braid,
            name=f"i_({self.f0.name}, {self.f1.name})",
        )


## -- CoEqualizers --


class Coequalizer(Construct):
    """The coequalizer of f0 : X → Y and f1 : X → Y is the quotient of Y
    by the gluing conditions 

        f0(x) ~ f1(x) for all x in X.
        
    Enumeration is induced from the enumeration method of setoids."""

    def __init__(self, f0: NamedFunction, f1: NamedFunction):
        if f0.dom != f1.dom or f0.cod != f1.cod:
            raise ValueError("The functions must have the same domain and codomain.")
        self.f0 = f0
        self.f1 = f1
        self.prequotient = f0.cod
        self.gluing_conditions = {(f0(x), f1(x)) for x in f0.dom}
        self.setoid = Setoid(self.prequotient, self.gluing_conditions)
        self._name = f"Coeq({f0.name}, {f1.name})"
        self._obj = self.setoid.obj

    def projection(self) -> NamedFunction:
        """Return the projection from Y to the coequalizer, defined by proj(y) = [y]."""
        return self.setoid.projection()

    def universal_solution(self, f: NamedFunction) -> NamedFunction:
        """For f : Y → C such that f ∘ f0 = f ∘ f1, return the unique function

            u : Coeq(f0, f1) → C 
            
        such that 
            
            u ∘ proj = f
        """
        if not self.test_coequalize(f):
            raise ValueError("The function f does not coequalize f0 and f1.")
        C = f.cod
        regle_u = lambda cls: f(next(iter(cls)))
        return NamedFunction(dom=self, cod=C, table=regle_u, name=f"u_{f.name}")

    def test_coequalize(self, g: NamedFunction) -> bool:
        if g.dom != self.f0.cod:
            raise ValueError("The domain of g must be the codomain of f0 and f1.")
        if g.cod != g.cod:
            raise ValueError("The codomain of g must be equal to the codomain of g.")
        return composition(self.f0, g) == composition(self.f1, g)


######################################################################################################

# Topos structure: subobjects, characteristic functions, power objects, etc.


def Omega_set() -> NamedSet:
    """Subobject classifier Ω, aka 2. Its elements are the native booleans False and True."""
    return NamedSet({False, True}, "Ω")


def Top() -> Injection:
    """Return the top map from the terminal object to Ω"""
    regle_top = {"*": True}
    return Injection(dom=Terminal(), cod=Omega_set(), table=regle_top, name="⊤")


def charmap(inclusion: Injection) -> NamedFunction:
    """Given a subobject (inclusion) S → A,
    returns the characteristic function χ_S : A → Ω."""
    return list_to_charmap(
        inclusion.cod,
        subobject_to_subset_charmap(inclusion),
        name=f"χ_{inclusion.name}",
    )


def _validate_subset_charmap(
    A: Representable, bits: Sequence[int | bool]
) -> tuple[int, ...]:
    """Validate the rank-ordered binary representation of a subset of ``A``."""
    normalized = tuple(bits)
    if len(normalized) != len(A):
        raise ValueError("A subset charmap must have exactly |A| entries.")
    if any(type(bit) not in (int, bool) or bit not in (0, 1) for bit in normalized):
        raise ValueError("A subset charmap may contain only 0 and 1.")
    return tuple(int(bit) for bit in normalized)


def index_to_subset_charmap(A: Representable, index: int) -> list[int]:
    """Decode ``index`` into bits, with bit ``i`` attached to ``A.unrank(i)``."""
    if not isinstance(index, int) or index < 0 or index >= 2 ** len(A):
        raise ValueError("A subset index must belong to range(2^|A|).")
    return [(index >> i) & 1 for i in range(len(A))]


def subset_charmap_to_index(A: Representable, bits: Sequence[int | bool]) -> int:
    """Encode rank-ordered subset bits as ``sum(bits[i] * 2**i)``."""
    normalized = _validate_subset_charmap(A, bits)
    return sum(bit << i for i, bit in enumerate(normalized))


def subset_charmap_to_subset(
    A: Representable, bits: Sequence[int | bool]
) -> FrozenSetAffiche:
    """Decode rank-ordered subset bits into a ``NamedSet``."""
    normalized = _validate_subset_charmap(A, bits)
    elements = {
        element
        for element, bit in zip(A.enumeration, normalized)
        if bit
    }
    return FrozenSetAffiche(elements)


def subset_to_subset_charmap(A: Representable, subset: Representable) -> list[int]:
    """Encode a represented subset of ``A`` as rank-ordered bits."""
    elements = set(subset)
    if not elements.issubset(A.obj.set):
        raise ValueError("The subset must be included in A.")
    return [int(element in elements) for element in A.enumeration]


def subset_to_index(A: Representable, subset: Representable) -> int:
    """Encode a represented subset of ``A`` by its integer subset code."""
    return subset_charmap_to_index(A, subset_to_subset_charmap(A, subset))


def subobject_to_subset_charmap(inclusion: Injection) -> list[int]:
    """Encode the image of a subobject ``S ↪ A`` as rank-ordered bits."""
    image = set(inclusion.values.values())
    return [int(element in image) for element in inclusion.cod.enumeration]


def subobject_to_index(inclusion: Injection) -> int:
    """Encode the image of a subobject ``S ↪ A`` in ``range(2^|A|)``."""
    return subset_charmap_to_index(
        inclusion.cod, subobject_to_subset_charmap(inclusion)
    )

def charmap_int(inclusion: Injection) -> int:
    """Compatibility alias for :func:`subobject_to_index`."""
    return subobject_to_index(inclusion)

def charmap_list(inclusion: Injection) -> list[int]:
    """Compatibility alias for :func:`subobject_to_subset_charmap`."""
    return subobject_to_subset_charmap(inclusion)

def list_to_charmap(
    A: Representable,
    bits: Sequence[int | bool],
    name: str | None = None,
) -> NamedFunction:
    """Build the characteristic function represented by rank-ordered bits."""
    normalized = _validate_subset_charmap(A, bits)
    values = {
        element: bool(bit)
        for element, bit in zip(A.enumeration, normalized)
    }
    return NamedFunction(
        dom=A,
        cod=Omega_set(),
        table=values,
        name=name or f"χ_{list(normalized)}",
    )

def list_to_subset(A: Representable, bits: Sequence[int | bool]) -> NamedSet:
    """Compatibility alias for :func:`subset_charmap_to_subset`."""
    return subset_charmap_to_subset(A, bits)

def binary_to_charmap(A: Representable, index: int) -> NamedFunction:
    """Build the characteristic function represented by a subset index."""
    return list_to_charmap(A, index_to_subset_charmap(A, index), name=f"χ_{index}")

def index_to_subset(A: Representable, index: int) -> NamedSet:
    """Decode a subset index into a ``NamedSet``."""
    return subset_charmap_to_subset(A, index_to_subset_charmap(A, index))

def index_to_list(A: Representable, index: int) -> list[int]:
    """Compatibility alias for :func:`index_to_subset_charmap`."""
    return index_to_subset_charmap(A, index)


def index_to_binary(A: Representable, index: int) -> list[int]:
    """Compatibility alias for the historical name used in examples."""
    return index_to_subset_charmap(A, index)


def subset_charmap_to_subobject(
    A: Representable, bits: Sequence[int | bool]
) -> Injection:
    """Decode subset bits into the canonical inclusion ``S ↪ A``."""
    return inclusion_subset(set(subset_charmap_to_subset(A, bits)), A)


def index_to_subobject(A: Representable, index: int) -> Injection:
    """Decode a subset index into the canonical inclusion ``S ↪ A``."""
    return subset_charmap_to_subobject(A, index_to_subset_charmap(A, index))


def pullback_charmap(chi: NamedFunction) -> Injection:
    """Given a characteristic function χ : A → Ω,
    return the corresponding subobject S → A as the pullback."""
    if chi.cod != Omega_set():
        raise ValueError("The codomain of the function must be Ω.")
    return Injection.from_function(Pullback(chi, Top()).proj_0())


def charmap_to_subset(chi: NamedFunction) -> NamedSet:
    """Given a characteristic function χ : A → Ω,
    return the corresponding subset of A as a NamedSet."""
    if chi.cod != Omega_set():
        raise ValueError("The codomain of the function must be Ω.")
    return chi.fiber(True)


def subset_charmap(chi: NamedFunction) -> NamedSet:
    """Compatibility alias for :func:`charmap_to_subset`."""
    return charmap_to_subset(chi)


def powerset_formel(A: Representable) -> NamedSet:
    """Powerset of a set A as a named set.
    Its elements are the subsets of A, represented as NamedSet objects.
    Constructs all the χ : A → Ω and take their fibers"""
    sub = set()
    for chi in HomSet(A, Omega_set()):
        subset_set = NamedSet(
            charmap_to_subset(chi).obj.set, f"{charmap_to_subset(chi).__repr__()}"
        )
        sub.add(subset_set)
    return NamedSet(sub, f"P({A.name})")

def powerset(A: Representable) -> NamedSet:
    """Return ``P(A)`` enumerated by subset codes ``0, ..., 2^|A| - 1``."""
    enum = tuple(index_to_subset(A, n) for n in range(2 ** len(A)))
    return NamedSet(set(enum), f"P({A.name})", enum=enum)


def preim(f: NamedFunction, subset_index: int) -> list[int]:
    """Return the charmap of ``f⁻¹(S)`` from the subset code of ``S ⊆ cod(f)``.

    If ``S_B`` is the rank-ordered charmap of ``S``, the returned list is
    exactly ``[S_B[f._raw[i]] for i in range(len(f.dom))]``.
    """
    if subset_index < 0 or subset_index >= 2 ** len(f.cod):
        raise ValueError("A subset index must belong to range(2^|cod(f)|).")
    codomain_charmap = index_to_subset_charmap(f.cod, subset_index)
    return [codomain_charmap[codomain_rank] for codomain_rank in f._raw]


def powerset_contravariant(f: NamedFunction) -> NamedFunction:
    """
    Return the contravariant powerset map induced by f : A → B,
    defined by 

        f*(S) = {x in A | f(x) in S} 

    for any subset S of B."""
    PA = powerset(f.dom)
    PB = powerset(f.cod)

    def regle_powerset(subset: Representable) -> NamedSet:
        subset_index = subset_to_index(f.cod, subset)
        preimage_charmap = preim(f, subset_index)
        return subset_charmap_to_subset(f.dom, preimage_charmap)

    return NamedFunction(dom=PB, cod=PA, table=regle_powerset, name=f"{f.name}*")


def powerset_contravariant_formel(f: NamedFunction) -> NamedFunction:
    """Compatibility alias for :func:`powerset_contravariant`."""
    return powerset_contravariant(f)


def powerset_covariant(f: NamedFunction) -> NamedFunction:
    """
    Return the covariant powerset map induced by f : A → B,
    defined by 

        Σf(S) = {f(x) | x in S} 
    
    for any subset S of A."""
    PA = powerset(f.dom)
    PB = powerset(f.cod)

    def regle_powerset(subset: Representable) -> NamedSet:
        image = {f(x) for x in subset}
        return subset_charmap_to_subset(
            f.cod,
            [int(y in image) for y in f.cod.enumeration],
        )

    return NamedFunction(dom=PA, cod=PB, table=regle_powerset, name=f"Σ{f.name}")


def inclusion_subset(subset_brut: set, A: Representable) -> Injection:
    """Given a subset of A as a set,
    return the corresponding inclusion as an Injection."""
    if not subset_brut.issubset(A.obj.set):
        raise ValueError("The subset must be included in A.")
    regle_incl = lambda x: x
    Subs = NamedSet(subset_brut, f"Sub_{A.name}", enum=A.obj.restrict_enum(subset_brut))
    return Injection(dom=Subs, cod=A, table=regle_incl, name=f"incl_Sub_{A.name}")


def injections_factorization(A: Representable, inj0 : Injection, inj1: Injection) -> bool:
    """Test whether the subobject represented by ``inj0`` factors through ``inj1``."""
    if inj0.cod != A or inj1.cod != A:
        raise ValueError("Both injections must have A as codomain.")
    code0 = subobject_to_index(inj0)
    code1 = subobject_to_index(inj1)
    return (code0 & code1) == code0

def union_subobjects(A: Representable, inj0: Injection, inj1: Injection) -> Injection:
    """Return the union of two subobjects of A as an Injection."""
    if inj0.cod != A or inj1.cod != A:
        raise ValueError("Both injections must have A as codomain.")
    code0 = subobject_to_index(inj0)
    code1 = subobject_to_index(inj1)
    union_code = code0 | code1
    return index_to_subobject(A, union_code)

def intersection_subobjects(A: Representable, inj0: Injection, inj1: Injection) -> Injection:
    """Return the intersection of two subobjects of A as an Injection."""
    if inj0.cod != A or inj1.cod != A:
        raise ValueError("Both injections must have A as codomain.")
    code0 = subobject_to_index(inj0)
    code1 = subobject_to_index(inj1)
    intersection_code = code0 & code1
    return index_to_subobject(A, intersection_code)

def complement_subobject(A: Representable, inj: Injection) -> Injection:
    """Return the complement of a subobject of A as an Injection."""
    if inj.cod != A:
        raise ValueError("The injection must have A as codomain.")
    code = subobject_to_index(inj)
    complement_code = (~code) & ((1 << len(A)) - 1)
    return index_to_subobject(A, complement_code)

def difference_subobjects(A: Representable, inj0: Injection, inj1: Injection) -> Injection:
    """Return the difference of two subobjects of A as an Injection."""
    if inj0.cod != A or inj1.cod != A:
        raise ValueError("Both injections must have A as codomain.")
    code0 = subobject_to_index(inj0)
    code1 = subobject_to_index(inj1)
    difference_code = code0 & (~code1)
    return index_to_subobject(A, difference_code)

def test_disjunction(A: Representable, inj0: Injection, inj1: Injection) -> bool:
    """Test whether the subobjects represented by ``inj0`` and ``inj1`` are disjoint."""
    if inj0.cod != A or inj1.cod != A:
        raise ValueError("Both injections must have A as codomain.")
    code0 = subobject_to_index(inj0)
    code1 = subobject_to_index(inj1)
    return (code0 & code1) == 0

def inclusion_test(subset0: set, subset1: set) -> bool:
    """Test if subset0 is included in subset1."""
    return subset0.issubset(subset1)


def inclusion_test_setnom(subset0: Representable, subset1: Representable) -> bool:
    """Test if subset0 is included in subset1, for NamedSet objects."""
    return subset0.obj.set.issubset(subset1.obj.set)

def inclusion(subset0: Representable, subset1: Representable) -> Injection:
    """Return the inclusion of subset0 in subset1 as an Injection,
    if subset0 is included in subset1, otherwise raise an error."""
    if not inclusion_test_setnom(subset0, subset1):
        raise ValueError(f"{subset0.name} is not included in {subset1.name}.")
    regle_incl = lambda x: x
    return Injection(
        dom=subset0, cod=subset1, table=regle_incl, name=f"incl_{subset0.name}_in_{subset1.name}"
    )


def union_sets(index: Sequence[Representable]) -> Representable | NamedSet:
    """The index is a set of subsets (NamedSet);
    return the union of these subsets as a NamedSet."""
    if not index:
        return Initial()
    union_elements: set = set()
    for S in index:
        union_elements |= S.obj.set
    return NamedSet(union_elements, f"Union_{index}")


def intersection_sets(index: Sequence[Representable]) -> Representable | NamedSet:
    """The index is a set of subsets (NamedSet);
    return the intersection of these subsets as a NamedSet."""
    if not index:
        return Terminal()
    intersection_elements: set = set.intersection(*(set(S.obj.set) for S in index))
    return NamedSet(intersection_elements, f"Intersection_{index}")


def union_subobjects(S0: Injection, S1: Injection) -> Injection:
    """Given two subobjects S0 -> B and S1 -> B,
    return their union as a subobject of B."""
    if S0.cod.obj != S1.cod.obj:
        raise ValueError("The codomains of the Injections must be equal.")
    B = S0.cod
    Im0 = set(S0.image())
    Im1 = set(S1.image())
    union_image = Im0 | Im1
    Union = NamedSet(union_image, f"{S0.name} U {S1.name}")
    regle_union = lambda x: x if x in union_image else None
    return Injection(
        dom=Union, cod=B, table=regle_union, name=f"i_{{{S0.name} U {S1.name}}}"
    )


def intersection_subsets(subset_A: Representable) -> NamedSet:
    """The index is a set of subsets (NamedSet) of A;
    return the intersection of these subsets as a NamedSet."""
    if not subset_A.obj.set:
        return NamedSet(set(), f"Intersection_{subset_A.name}")
    intersection_elements = set.intersection(*(S.obj.set for S in subset_A.obj.set))
    intersection_set = NamedSet(intersection_elements, f"Intersection_{subset_A.name}")
    return intersection_set


def intersection_subobjects(S0: Injection, S1: Injection) -> Injection:
    """Given two subobjects S0 -> B and S1 -> B,
    return their intersection as a subobject of B."""
    if S0.cod.obj != S1.cod.obj:
        raise ValueError("The codomains of the Injections must be equal.")
    B = S0.cod
    inter = {x for x in B.obj.set if x in S0.image() and x in S1.image()}
    regle_intersection = lambda x: x if x in inter else None
    Intersection = NamedSet(inter, f"{S0.name} ∩ {S1.name}")
    return Injection(
        dom=Intersection,
        cod=B,
        table=regle_intersection,
        name=f"i_{{{S0.name} ∩ {S1.name}}}",
    )


def complement_subset(subset: Injection, A: Representable):
    """Given a subset of A as a Representable,
    return its complement in A as a NamedSet."""
    complement_elements = set(A.obj.set) - set(subset.dom.obj.set)
    complement_set = NamedSet(complement_elements, f"Complement_{subset.name}")
    return complement_set


## -- ZF constructions --

"""Some standard constructions of sets in ZF set theory,
such as singletons, pairs, ordered pairs, unions of elements of a set of sets, etc.
Use only NamedSet rather than Representable
Require hashability of the elements"""


def singleton(a) -> NamedSet:
    """Return the singleton set {a} as a NamedSet.
    a may be of any type: int, str, tuple, frozenset, etc.
    as long as it is hashable."""
    return NamedSet({a}, f"{{{a}}}")


def pair(a, b) -> NamedSet:
    """
    Return the pair {a, b} as a NamedSet."""
    if a == b:
        return singleton(a)
    else:
        return NamedSet({a, b}, f"{{{a}, {b}}}")


def ordered_pair(a, b) -> NamedSet:
    """
    Return the ordered pair (a, b) as a NamedSet, defined by 
    
        (a, b) = {{a}, {a, b}}
        
    """
    return pair(singleton(a), pair(a, b))


def union(X: Representable, Y: Representable) -> NamedSet:
    """
    Return the union of two sets X and Y as a NamedSet,
    defined by 

        X ∪ Y = {x | x ∈ X or x ∈ Y}
    """
    union_elements: set = X.obj.set | Y.obj.set
    return NamedSet(union_elements, f"{X.name} ∪ {Y.name}")


def union_of_elements(X: Representable) -> NamedSet:
    """
    Return the union of the elements of X as a NamedSet,
    defined by 

        ⋃X = {x | ∃y ∈ X, x ∈ y}
    """
    union_elements: set = set()
    for y in X:
        if isinstance(y, Representable):
            union_elements |= y.obj.set
        else:
            raise ValueError("The elements of X must be Representable objects.")
    return NamedSet(union_elements, f"⋃{X.name}")


def ordinal(n: int) -> Representable:
    """
    Return Von-Neumann ordinal n as a Representable, defined by:

        - 0 = ∅
        - n+1 = n ∪ {n}
    """
    if n == 0:
        return Initial()
    else:
        return union(ordinal(n - 1), singleton(ordinal(n - 1)))


################################################################

# Relations


class Relation:
    """A binary relation R on a set A is a subset of the Cartesian product A x B.
    We can represent it as a set of pairs (a, b) with a in A and b in B such that (a, b) ∈ R.
    """

    def __init__(
        self,
        dom: Representable,
        cod: Representable,
        table: set[tuple],
        name: str,
    ):
        """Initialize a relation R on A x B
        given by a set of pairs (a, b) with a in A and b in B.
        Represent it as an injection from R ↣ A x B."""
        self.dom = dom
        self.cod = cod
        self.pairs = table
        self._name = name
        self._obj = NamedSet(table, self._name)
        self.mono = Injection(
            dom=self._obj,
            cod=Product(dom, cod),
            table=lambda pair: pair,
            name=f"i_{self._name}",
        )

    @classmethod
    def from_function(cls, f: NamedFunction):
        """Given a function f : A → B,
        return the graph of f as a relation on A x B."""
        pairs = {(a, f(a)) for a in f.dom}
        return cls(dom=f.dom, cod=f.cod, table=pairs, name=f"Graph({f.name})")

    @classmethod
    def from_subobject(cls, A: Representable, B: Representable, subobject: FrozenSetAffiche, name: str | None = None):
        """Given a subobject of A x B, return the corresponding relation."""
        return cls(dom=A, cod=B, table=set(subobject), name=f"{name}" if name else "R")

    def gluing(self):
        """Return the setoid on A x B generated by the pairs of the relation,
        i.e., the smallest equivalence relation on A x B that contains the pairs of R"""
        pairs_embedded = {((0, pair[0]), (1, pair[1])) for pair in self.pairs}
        return Setoid(Coproduct(self.dom, self.cod), pairs_embedded)

    def __repr__(self):
        """Return the string 

            R ↣ A x B
        """
        graph_str = f"{self._name} ↣ {self.dom.name} x {self.cod.name}"
        for pair in self.pairs:
            graph_str += f"\n \t \t  {pair[0]} ~ {pair[1]}"
        return graph_str

    def __str__(self):
        """Return the name of the relation."""
        return self._name

    def __len__(self):
        """Return the number of pairs in the relation."""
        return len(self.pairs)

    def __eq__(self, other):
        """Return True if the relations are equal, i.e., have the same pairs."""
        return (
            isinstance(other, Relation)
            and self.dom == other.dom
            and self.cod == other.cod
            and self.pairs == other.pairs
        )

    def __hash__(self):
        """Return a hash of the relation based on its pairs."""
        return hash((self.dom, self.cod, frozenset(self.pairs)))

    def display(self):
        """Print the relation as a list
        a R b for each pair (a, b) in R."""
        print(self.__repr__())
        for pair in self.pairs:
            print(f"  {pair[0]} {self._name} {pair[1]}")


def composition_relations(R: Relation, S: Relation) -> Relation:
    """Given two relations R on A x B and S on B x C,
    return their composition S ∘ R on A x C defined by
    
        S ∘ R = { (a, c) | ∃b ∈ B, (a, b) ∈ R and (b, c) ∈ S }
    """
    if R.cod != S.dom:
        raise ValueError("The codomain of R must match the domain of S.")
    composed_pairs: set = {
        (a, c) for (a, b1) in R.pairs for (b2, c) in S.pairs if b1 == b2
    }
    return Relation(
        dom=R.dom, 
        cod=S.cod, 
        table=composed_pairs, 
        name=f"{S._name} ∘ {R._name}"
    )


def join_relations(R: Relation, S: Relation) -> Relation:
    """Given two relations R and S on the same sets A x B,
    return their join R ∨ S defined by 

        R ∨ S = { (a, b) | (a, b) ∈ R or (a, b) ∈ S }
    """
    if R.dom != S.dom or R.cod != S.cod:
        raise ValueError("The domains and codomains of R and S must be the same.")
    joined_pairs: set = R.pairs | S.pairs
    return Relation(
        dom=R.dom, 
        cod=R.cod, 
        table=joined_pairs, 
        name=f"{R._name} ∨ {S._name}"
    )


def meet_relations(R: Relation, S: Relation) -> Relation:
    """Given two relations R and S on the same sets A x B,
    return their meet R ∧ S defined by

        R ∧ S = { (a, b) | (a, b) ∈ R and (a, b) ∈ S }
    """
    if R.dom != S.dom or R.cod != S.cod:
        raise ValueError("The domains and codomains of R and S must be the same.")
    met_pairs: set = R.pairs & S.pairs
    return Relation(
        dom=R.dom, 
        cod=R.cod, 
        table=met_pairs, 
        name=f"{R._name} ∧ {S._name}"
    )


## -- Relation homset --


class Rel(Construct):
    """The homset of relations from A to B is the set of all relations on A x B,
    ordered by inclusion. It can be represented as a NamedSet of Relation objects."""

    def __init__(self, dom: Representable, cod: Representable):
        self.dom = dom
        self.cod = cod
        self._name = f"Rel({dom.name}, {cod.name})"
        subsets = powerset(Product(dom, cod))
        relation_enum = tuple(
            Relation.from_subobject(dom, cod, subset, name=f"R_{i}")
            for i, subset in enumerate(subsets)
        )
        self.enum = relation_enum
        self._obj = NamedSet(set(relation_enum), self._name, enum=relation_enum)


# TODO: add infinite sets via inductive constructions or generators
# for example: N = {∅, {∅}, {{∅}}, ...} or N = {0, 1, 2, ...} with 0 = ∅, 1 = {0}, 2 = {0, 1}, etc.
# recursive, recursively enumerable, definable distinctions
