# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Sphinx configuration for the cudaq-algorithms documentation.

This is a plain, checked-in ``conf.py`` (no CMake templating) since the
library is pure Python and needs no Doxygen/C++ toolchain. autodoc imports
``cudaq_algorithms`` -- and therefore ``cudaq`` -- so the build environment
must have those importable (see docs/Makefile).
"""

import os
import sys

# Make the pure-Python package importable for autodoc. ``cudaq`` itself is
# expected on PYTHONPATH already (env-specific: a local install or the CI
# container), so it is not hard-coded here.
sys.path.insert(0, os.path.abspath("../../python"))

# -- Project information -----------------------------------------------------

project = "CUDA-Q Algorithms"
copyright = "2026, NVIDIA Corporation & Affiliates"
author = "NVIDIA Corporation & Affiliates"
version = os.getenv("CUDAQ_ALGORITHMS_VERSION", "latest")
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]

# Render `text` as inline code (matches the backtick style of the source
# markdown we migrate from).
default_role = "code"
pygments_style = "lightbulb"

# Prefix autosection labels with the document path so identically-titled
# sections across pages do not collide (important under -W).
autosectionlabel_prefix_document = True

# -- autodoc / autosummary / napoleon ----------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
# Our docstrings mix Google-style and NumPy-style Parameters blocks.
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": False,
    "prev_next_buttons_location": "both",
    "style_nav_header_background": "#76b900",  # NVIDIA green
}
html_static_path = ["_static"]
html_css_files = ["cudaq_override.css"]
htmlhelp_basename = "cudaqAlgorithmsDoc"

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
