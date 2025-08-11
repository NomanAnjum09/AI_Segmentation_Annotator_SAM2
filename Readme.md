# SAM2 Fast Segmentation Annotator (PyQt5)

A fast, lightweight UI for **interactive segmentation** powered by **SAM2**.  
Hover to preview masks, **left-click** to lock a mask, then **click a class** to assign it. Export labels to **YOLO segmentation format** (one `.txt` per image).

This app uses the SAM2 backend from: https://github.com/Gy920/segment-anything-2-real-time

---

## Features

- **Live hover preview** of SAM2 masks under the cursor.
- **Click to lock** a mask; **click a class** to assign/update it.
- **No duplicates:** clicking an already-annotated object selects that instance & class.
- **Undo/Redo** and **auto-load** of existing labels per image.
- **Zoom & pan**: Ctrl+`+`, Ctrl+`-`, Ctrl+`0` (reset). Scrollbars appear when zoomed.
- **Export to YOLO-Seg**: `class_id x1 y1 x2 y2 ...` (normalized polygon).

---

## Quickstart

python -m venv venv
source venv/bin/activate

# SAM2 backend (installed editable so its modules are importable)
    git clone https://github.com/Gy920/segment-anything-2-real-time.git
    cd segment-anything-2-real-time/
    pip install -e .
    cd ..

# App dependencies
    pip install -r requirements.txt

# Run
    python fast_annotation.py \
    --images /path/to/images \
    --checkpoint /path/to/sam2_checkpoint.pt \
    --config configs/sam2.1/sam2.1_hiera_l.yaml


# Notes
- Python 3.12 supported.
- GPU recommended; falls back to CPU if CUDA isn’t available.
- Provide a valid SAM2 checkpoint and matching config.

# Usage
- Launch with --images pointing to a folder (jpg/jpeg/png/bmp/tif/tiff). By default it will look for images folder in the same directory.
- A matching .txt is created next to each image on export.
- Hover to preview a mask; left-click to lock the preview (dashed outline).
- Click a class in the right sidebar to assign it.
- Clicking another class updates the same (pending) instance.
- Clicking a different object locks a new mask (repeat the process).
- Existing instance? Click on it to select; class clicks update its class.
- Export labels with Export (YOLO) or press Ctrl+S.

# Controls & Shortcuts
- Hover: live mask preview
- Left-click: lock mask (or select existing instance)
- Class click: assign/modify class for locked/selected instance
- Delete / Backspace: remove last added instance
- Ctrl+Z / Ctrl+Shift+Z: undo / redo
- Ctrl+S: export YOLO labels for current image
- Prev / Next buttons: navigate images (labels auto-load if present)
- Zoom: Ctrl +, Ctrl -, Ctrl+0 (reset). Scrollbars appear when zoomed.
- Esc: cancel current locked preview
- F11: toggle fullscreen

# Label Format (YOLO Segmentation)
##### Each line in <image>.txt:
- class_id x1 y1 x2 y2 x3 y3 ... xN yN
- Coordinates are normalized to [0, 1]; polygon must have ≥3 points.
- One line per instance.

###### Example:
    2 0.5123 0.3409 0.5481 0.3712 0.5604 0.4200 0.5150 0.4333

- When revisiting an image, existing labels are auto-loaded into the UI.

- CLI Options
    ```bash
--images      Path to an image folder (jpg/jpeg/png/bmp/tif/tiff)
--checkpoint  Path to a SAM2 checkpoint file (.pt / .pth)
--config      Path to a SAM2 config YAML (e.g., sam2.1_hiera_l.yaml)
--classes     Optional list of initial class names

Example:

    python fast_annotation.py \
    --images data/images \
    --checkpoint models/sam2.1_hiera_large.pt \
    --config configs/sam2.1/sam2.1_hiera_l.yaml \
    --classes A1 A2 A3 B1 B2 C1

# Troubleshooting
- Qt “xcb” plugin errors on Linux
    Use headless OpenCV and install system libs:
    ```bash
    pip uninstall -y opencv-python opencv-contrib-python
    pip install --no-cache-dir opencv-python-headless

    sudo apt-get install -y libxcb1 libx11-xcb1 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxcb-randr0 libxcb-xfixes0 libxcb-shape0 libxcb-sync1 libxcb-xinerama0 \
    libglu1-mesa libopengl0

# Roadmap
- Right-click to cancel a locked preview
- Class search/filter in the sidebar
- Box/brush assist modes for tricky objects
- Multi-image export & dataset stats