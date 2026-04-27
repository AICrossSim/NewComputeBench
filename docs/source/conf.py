import os
import sys

# -- Project information -------------------------------------------------------

project = "NewComputeBench"
copyright = "2025, AICrossSim"
author = "AICrossSim"
release = "0.1.0"

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/images/logo.png"
html_title = "NewComputeBench"

html_theme_options = {
    "repository_url": "https://github.com/AICrossSim/NewComputeBench",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "master",
    "path_to_docs": "docs/source",
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
}
