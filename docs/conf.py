# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — Sphinx configuration                                ║
# ║  « Napoleon + autodoc + Furo »                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boutscope import __version__  # noqa: E402

project = "boutscope"
author = "Bart R. H. Geurten"
copyright = "2026, Bart R. H. Geurten"  # noqa: A001
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["cv2"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "furo"
html_title = f"boutscope {version}"
html_theme_options = {
    "source_repository": "https://github.com/zerotonin/boutscope",
    "source_branch": "main",
    "source_directory": "docs/",
}
