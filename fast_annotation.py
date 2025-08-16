import argparse
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
import sys
from PyQt5 import QtWidgets   # import PyQt5 BEFORE cv2
from src.segmentor import Segmenter
from src.annotator import Annotator




# -------------------- Main --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images", help="Folder with images to annotate")
    ap.add_argument("--model_type", default="large", help="SAM2 model type: tiny/small/large")
    ap.add_argument("--classes", nargs="*", default=["A1","A2","A3","A4","A5","A6","B1","B2","B3","B4","B5",
                                                     "B6","C1","C2","C3","C4","C5","C6"], help="Initial class names")
    return ap.parse_args()

def get_model_config(type):
    if type == "large":
        return "models/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif type == "small":
        return "models/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"
    else:
        return "models/sam2.1_hiera_tiny.pt","configs/sam2.1/sam2.1_hiera_t.yaml"

def main():
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")  # consistent across platforms

    checkpoint , config = get_model_config(args.model_type)
    seg = Segmenter(checkpoint, config)
    win = Annotator(args.images, seg, args.classes)
    # Default window size
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()