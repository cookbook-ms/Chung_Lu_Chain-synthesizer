import os
import sys


project = 'PowerGridSynth'
copyright = '2025, PowerGridSynth Developers'
author = 'PowerGridSynth Developers'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # Generates docs from code docstrings
    'sphinx.ext.napoleon',     # Parses Google-style docstrings
    'sphinx.ext.viewcode',     # Adds links to source code
    'myst_parser',             # Allows using Markdown (.md) instead of .rst
]

# Support for Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to match ReadTheDocs.org
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']