# ---------------------------------------------------------------------------
# Imports to adapt if your public API changes.
#
# Recommended style:
#   from seth import NamedSet, NamedFunction
#   from hyp import Hypergraph
#
# If your __init__.py exports everything useful with:
#   from .core import *
# then these imports should work directly.
# ---------------------------------------------------------------------------

try:
    import seth
    import hyp
except ImportError as exc:
    print(
        "Could not import seth or hyp. "
        "Make sure you installed the project with `pip install -e .` "
        "from the repository root."
    )