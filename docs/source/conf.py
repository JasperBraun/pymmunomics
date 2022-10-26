# import pymmunomics

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pymmunomics"
copyright = "2022, Jasper Braun"
author = "Jasper Braun"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "numpydoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.napoleon",
    # "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_sidebars = { '**': ['searchbox.html', 'globaltoc.html']}#, 'relations.html', 'sourcelink.html', ] }
html_short_title = "pymmunomics"
html_theme_options = {
   "navbar_align": "left",
   "show_prev_next": False,
   "page_sidebar_items": [],
}

# autosummary
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
autosummary_generate = True

# autodoc settings
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# autodoc_typehints = "description" # Doesn't work with numpydoc
autoclass_content = "both"

# doctest settings
# https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html
# doctest_global_setup = """
# import pandas as pd
# """

# Napoleon settings
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#module-sphinx.ext.napoleon
# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = False
# napoleon_preprocess_types = False
# napoleon_type_aliases = None
# napoleon_attr_annotations = True
