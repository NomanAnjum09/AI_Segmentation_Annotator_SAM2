
import os, os.path as osp
os.environ.pop("QT_PLUGIN_PATH", None)                 # avoid cv2's plugin dir
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")        # prefer X11 on Linux

try:
    import PyQt5  # import first to locate its plugins
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = osp.join(
        osp.dirname(PyQt5.__file__), "Qt", "plugins", "platforms"
    )
except Exception:
    pass
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets   # import PyQt5 BEFORE cv2
import cv2   
from typing import List, Optional, Tuple
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class StatsWindow(QtWidgets.QMainWindow):
    """Independent window with a tab that shows a bar chart of class counts."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Stats")
        self.resize(640, 420)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Chart tab
        self.chart_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.chart_tab, "Counts")

        self.chart_layout = QtWidgets.QVBoxLayout(self.chart_tab)
        self.canvas = FigureCanvas(plt.Figure(figsize=(6, 4), dpi=100))
        self.chart_layout.addWidget(self.canvas)

    def plot_counts(self, labels: list, counts: list):
        """Plot a simple bar chart (labels = class names, counts = totals)."""
        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)
        ax.bar(range(len(counts)), counts)  # no explicit colors/styles per your constraints
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Annotated Objects per Class")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

def _label_path_for( img_path: str) -> str:
        return os.path.splitext(img_path)[0] + ".txt"

def compute_class_distribution(class_names, images) -> (list, list):
    """
    Scan all label .txt files next to images.
    Returns (class_names_list, counts_list).
    """
    # Ensure we have up-to-date class names
    
    counts = [0] * len(class_names)

    for img_path in images:
        txt = _label_path_for(img_path)
        if not os.path.exists(txt):
            continue
        try:
            with open(txt, "r", encoding="utf-8") as f:
                for ln in f:
                    parts = ln.strip().split()
                    if not parts:
                        continue
                    # first token is class id
                    try:
                        cls_id = int(float(parts[0]))
                    except Exception:
                        continue
                    if 0 <= cls_id < len(counts):
                        counts[cls_id] += 1
        except Exception:
            # ignore unreadable files
            pass

    return class_names, counts