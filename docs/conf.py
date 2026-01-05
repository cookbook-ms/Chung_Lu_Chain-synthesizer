import os
import sys

from inspect import getsourcefile

# Point Sphinx to the source code so it can read docstrings
sys.path.insert(0, os.path.abspath('../'))

project = 'PowerGridSynth'
copyright = '2025, PowerGridSynth Developers'
author = 'PowerGridSynth Developers'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # Generates docs from code docstrings
    'sphinx.ext.napoleon',     # Parses Google-style docstrings
    'sphinx.ext.viewcode',     # Adds links to source code
    'sphinx.ext.mathjax',
    'myst_parser',             # Allows using Markdown (.md) instead of .rst
    'nbsphinx',
    'nbsphinx_link',
]

suppress_warnings = [
    'config.cache',
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

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}


# -- nbsphinx ----------------------------------------------------------------
nbsphinx_execute = 'never'

# Don't load require.js because it conflicts with bootstrap.min.js being
# loaded after it. If you need require.js, load it manually by adding it
# to the html_js_files list.
nbsphinx_requirejs_path = ""

# -- pandoc (required by nbsphinx) -------------------------------------------
# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir
    pypandoc.ensure_pandoc_installed(
        targetfolder=pandoc_dir,
        delete_installer=True,
    )