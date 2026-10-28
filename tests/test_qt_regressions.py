"""Guards against known crashes in the Qt binding.

PySide6 6.12.0 corrupts a reference count when a signal carries a
QPointF; pyqtgraph does exactly that on every mouse move over a plot, so
dragging the histogram range aborted the interpreter.  pyproject.toml
caps PySide6 below 6.12; this test fails (by crashing the worker) if a
binding with the bug is installed again.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PySide6")

EMITS: int = 300_000

STRESS = textwrap.dedent(f"""
    import gc
    from PySide6.QtCore import QEvent, QPointF, Signal
    from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsSceneMouseEvent

    app = QApplication([])

    class Scene(QGraphicsScene):
        moved = Signal(object)

    scene, got = Scene(), []
    scene.moved.connect(lambda p: got.append(p.x()))
    for i in range({EMITS}):
        event = QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMouseMove)
        event.setScenePos(QPointF(i, 1))
        scene.moved.emit(event.scenePos())
        if i % 20_000 == 0:
            got.clear()
            gc.collect()
    print("ok")
""")


def test_signal_with_qpointf_does_not_crash():
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [sys.executable, "-c", STRESS], env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]
