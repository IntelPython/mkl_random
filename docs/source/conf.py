# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mkl_random"
copyright = "2017-2026, Intel Corp."
author = "Intel Corp."
release = "1.6.0dev0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "numpydoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.programoutput",
    # "sphinxcontrib.googleanalytics",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

# Generate per-method stub pages for autosummary ":toctree:" tables.
autosummary_generate = True

# Let api.rst list class members via autosummary
# (avoid numpydoc duplicating them).
numpydoc_show_class_members = False

# Shared external link, referenced as `MKL Documentation`_ from docstrings.
rst_epilog = """
.. _MKL Documentation:
   https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
"""


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["mkl_random-custom.css"]
