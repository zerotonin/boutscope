# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — boutscope                                           ║
# ║  « ground truth for behavioural categorisers »                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Sample labelled bouts from any categoriser, watch them in the   ║
# ║  original video, and rate them.                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""boutscope: ground truth for behavioural categorisers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("boutscope")
except PackageNotFoundError:
    __version__ = "0.0.0"
