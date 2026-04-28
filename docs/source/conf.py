import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "hypypyp"
author = "Axel Osmond"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "furo"
templates_path = ["_templates"]
exclude_patterns = []
html_static_path = ["_static"]