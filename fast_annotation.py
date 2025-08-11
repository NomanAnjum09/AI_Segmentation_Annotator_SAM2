"""
SAM2 Fast Segmentation Annotator (PyQt5)

Features
- Hover over the image to preview a SAM/SAM2 mask at the cursor.
- If hovering/clicking over an *already annotated* object, highlight it and select its class (no duplicate added).
- Left‑click on background/object → confirm a new mask and add it to the annotation.
- Sidebar to pick/add classes. Each confirmed instance is assigned the selected class.
- Delete/Backspace to remove last; Ctrl+Z/Shift+Ctrl+Z for undo/redo.
- Save to YOLO segmentation format (polygon): `class_id x1 y1 x2 y2 ...` (normalized).
- Loads an image folder; creates matching `.txt` next to images on Save.
- Auto-select UI theme (light/dark) based on system settings (can override with `--theme`).

Requirements
  pip install pyqt5 opencv-python-headless numpy torch
  # For models (choose one):
  # SAM 2 (recommended):
  #   pip install -U git+https://github.com/facebookresearch/segment-anything-2.git
  #   (provide a SAM2 checkpoint path)
  # OR Original SAM (fallback):
  #   pip install git+https://github.com/facebookresearch/segment-anything.git

Run
  python sam2_annotator.py --images /path/to/images \
                           --checkpoint /path/to/sam2_or_sam_checkpoint.pth \
                           --model-type sam2_hiera_l   # or vit_h for original SAM \
                           --theme auto  # or light/dark
"""
from __future__ import annotations
import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets

# -------------------- SAM / SAM2 loader --------------------
class Segmenter:
    def __init__(self, ckpt_path: str, config: str, device: str = None):
        self.device = device or ("cuda" if self._torch_cuda_available() else "cpu")
        self.kind = None  # "sam2" or "sam"
        self.predictor = None
        self._init_model(ckpt_path, config)

    def _torch_cuda_available(self) -> bool:
        try:
            import torch  # noqa
            return torch.cuda.is_available()
        except Exception:
            return False

    def _init_model(self, ckpt_path: str, config: str):
        # Try SAM2 first
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2 = build_sam2(config, ckpt_path)
            sam2.to(self.device)
            self.predictor = SAM2ImagePredictor(sam2)
            self.kind = "sam2"
            return
    
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SAM2 or SAM. Ensure packages/checkpoint/model_type are correct. Error: {e}"
            )

    def set_image(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

    def predict_from_point(self, x: int, y: int, positive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mask, score). Chooses highest-scoring mask among outputs.
        mask: HxW bool array
        score: float in [0,1]
        """
        input_point = np.array([[x, y]])
        input_label = np.array([1 if positive else 0])
        masks, scores, _ = self.predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
        idx = int(np.argmax(scores))
        return masks[idx].astype(bool), float(scores[idx])

# -------------------- Geometry helpers --------------------

def mask_to_polygon(mask: np.ndarray, approx_eps: float = 2.0) -> Optional[np.ndarray]:
    """Extract the largest external contour as polygon. Returns Nx2 int array (x,y) or None."""
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # largest area contour
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:  # too small
        return None
    epsilon = approx_eps
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    poly = approx.reshape(-1, 2)
    return poly


def normalize_poly(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    poly = poly.astype(np.float32)
    poly[:, 0] = np.clip(poly[:, 0] / float(w), 0.0, 1.0)
    poly[:, 1] = np.clip(poly[:, 1] / float(h), 0.0, 1.0)
    return poly


# -------------------- Data models --------------------
@dataclass
class Instance:
    poly: np.ndarray  # Nx2 int (image coords)
    cls_id: int
    inst_id: int = -1


# -------------------- Main Widget --------------------
class Annotator(QtWidgets.QMainWindow):
    def __init__(self, images_dir: str, segmenter: Segmenter, class_names: List[str]):
        super().__init__()
        self.setWindowTitle("SAM2 Fast Segmentation Annotator")
        self.segmenter = segmenter
        self.images = self._collect_images(images_dir)
        if not self.images:
            raise SystemExit(f"No images found in {images_dir}")
        self.img_idx = 0
        self.image_bgr = None  # current image (BGR)
        self.preview_mask = None  # bool mask under cursor
        self.preview_point = None  # (x,y)
        self.instances: List[Instance] = []
        self.undo_stack: List[List[Instance]] = []
        self.redo_stack: List[List[Instance]] = []
        self.hover_instance_idx: Optional[int] = None  # local hit-test result
        self._next_inst_id = 0
        self.preview_locked = False
        self.pending_inst_idx: Optional[int] = None
        # UI
        self._build_ui(class_names)
        self._load_current_image()

        # timers for hover preview throttling
        self.hover_timer = QtCore.QTimer()
        self.hover_timer.setInterval(40)  # ~25 FPS hover preview
        self.hover_timer.timeout.connect(self._update_preview)
        self.hover_timer.start()

    # ---------- UI construction ----------
    def cancel_preview(self):
        self.preview_mask = None
        self.preview_locked = False
        self.pending_inst_idx = None
        self.view.set_temp_poly(None)
        self.view.set_highlight(None)
        self.view.set_overlay(None, [inst.poly for inst in self.instances])
        self.info_label.setText("Preview cancelled.")

    def _build_ui(self, class_names: List[str]):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(8)

        # Left: image view expands
        self.view = ImageView(self)
        self.view.setMouseTracking(True)
        self.view.mouse_moved.connect(self.on_mouse_move)
        self.view.left_clicked.connect(self.on_left_click)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Right: sidebar container with ScrollArea (top) + footer (bottom)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # ---- Scrollable top content ----
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        scroll_inner = QtWidgets.QWidget()
        s = QtWidgets.QVBoxLayout(scroll_inner)
        s.setContentsMargins(8, 8, 8, 8)
        s.setSpacing(8)

        title = QtWidgets.QLabel("Classes")
        title.setStyleSheet("font-weight: 600;")
        s.addWidget(title)

        self.class_list = QtWidgets.QListWidget()
        # Keep it flexible but not unbounded
        self.class_list.setMinimumHeight(200)
        self.class_list.setMaximumHeight(500)  # prevents runaway height on tall screens
        for name in class_names:
            self.class_list.addItem(name)
        self.class_list.setCurrentRow(0)
        s.addWidget(self.class_list)

        self.info_label = QtWidgets.QLabel("Hover to preview, click to confirm. Del = remove last. Ctrl+S = save.")
        self.info_label.setWordWrap(True)
        s.addWidget(self.info_label)

        add_row_w = QtWidgets.QWidget()
        add_row = QtWidgets.QHBoxLayout(add_row_w)
        add_row.setContentsMargins(0, 0, 0, 0)
        self.new_class_edit = QtWidgets.QLineEdit()
        self.new_class_edit.setPlaceholderText("Add new class name")
        self.add_class_btn = QtWidgets.QPushButton("+")
        self.add_class_btn.setFixedWidth(28)
        self.add_class_btn.clicked.connect(self.add_class)
        add_row.addWidget(self.new_class_edit)
        add_row.addWidget(self.add_class_btn)
        s.addWidget(add_row_w)

        s.addStretch(1)  # push content up inside the scroll area
        scroll.setWidget(scroll_inner)

        # ---- Sticky footer with navigation + save ----
        footer_w = QtWidgets.QWidget()
        footer = QtWidgets.QHBoxLayout(footer_w)
        footer.setContentsMargins(8, 0, 8, 0)
        footer.setSpacing(8)

        self.prev_btn = QtWidgets.QPushButton("⟵ Prev")
        self.save_btn = QtWidgets.QPushButton("Export (YOLO)")
        self.next_btn = QtWidgets.QPushButton("Next ⟶")

        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_yolo)
        self.class_list.currentRowChanged.connect(self.on_class_selected)

        self.prev_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.save_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.next_btn.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

        footer.addWidget(self.prev_btn)
        footer.addWidget(self.save_btn)
        footer.addWidget(self.next_btn)

        # Assemble right panel
        right_layout.addWidget(scroll, 1)   # takes remaining space and scrolls if needed
        right_layout.addWidget(footer_w, 0) # always visible

        # Put into main layout
        main.addWidget(self.view, 1)
        main.addWidget(right, 0)

        # Keyboard shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self.save_yolo)
        QtWidgets.QShortcut(QtGui.QKeySequence("Delete"), self, activated=self.undo_last)
        QtWidgets.QShortcut(QtGui.QKeySequence("Backspace"), self, activated=self.undo_last)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.cancel_preview)
        # Reasonable default/floor sizes
        self.setMinimumSize(900, 600)


    # ---------- Image handling ----------
    def _collect_images(self, d: str) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if os.path.splitext(f.lower())[1] in exts]
        return files

    def _load_current_image(self):
        img_path = self.images[self.img_idx]
        self.image_bgr = cv2.imread(img_path)
        if self.image_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        self.segmenter.set_image(self.image_bgr)
        self.view.set_image(self.image_bgr)
        self.instances.clear(); self.undo_stack.clear(); self.redo_stack.clear()
        self.preview_mask = None; self.preview_point = None; self.hover_instance_idx = None
        self.view.set_highlight(None)
        self.update()

    # ---------- Events ----------
    def on_class_selected(self, row: int):
    # Ignore if nothing selected
        if row is None or row < 0:
            return

        # Case A: We have a locked preview mask → first class click commits it.
        if self.preview_locked and self.preview_mask is not None:
            poly = mask_to_polygon(self.preview_mask, approx_eps=2.0)
            if poly is None or len(poly) < 3:
                self.info_label.setText("Mask too small or invalid to save.")
                return

            # If first time for this locked mask → create a new instance
            if self.pending_inst_idx is None:
                self._push_undo()
                self.instances.append(Instance(poly=poly, cls_id=row, inst_id=self._next_inst_id))
                self._next_inst_id += 1
                self.pending_inst_idx = len(self.instances) - 1
                self.info_label.setText(f"Added new instance with class #{row}.")
            else:
                # Already created; subsequent class clicks just update it
                self.instances[self.pending_inst_idx].cls_id = row
                self.info_label.setText(f"Updated instance #{self.pending_inst_idx} → class #{row}.")

            # After first commit, stop using the temp overlay; highlight the instance instead
            self.preview_mask = None
            self.preview_locked = False
            self.view.set_temp_poly(None)
            self.view.set_highlight(self.pending_inst_idx)
            self.view.set_overlay(None, [inst.poly for inst in self.instances])
            return

        # Case B: Editing an existing instance via class change
        if self.pending_inst_idx is not None and 0 <= self.pending_inst_idx < len(self.instances):
            self.instances[self.pending_inst_idx].cls_id = row
            self.view.set_overlay(None, [inst.poly for inst in self.instances])
            self.info_label.setText(f"Updated instance #{self.pending_inst_idx} → class #{row}.")

    def on_mouse_move(self, x: int, y: int):
        self.preview_point = (x, y)
        # Throttled by timer -> _update_preview

    def _instance_at_point(self, x: int, y: int) -> Optional[int]:
        """Return index of the first instance whose polygon contains (x,y), else None."""
        for i, inst in enumerate(self.instances):
            poly = inst.poly.reshape(-1, 1, 2).astype(np.int32)
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return i
        return None

    def _update_preview(self):
        if self.preview_locked:
            return  # keep the locked preview as-is
        if self.preview_point is None or self.image_bgr is None:
            return
        x, y = self.preview_point
        h, w = self.image_bgr.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            self.preview_mask = None
            self.hover_instance_idx = None
            self.view.set_highlight(None)
            self.view.set_overlay(None, [inst.poly for inst in self.instances])
            return
        idx = self._instance_at_point(x, y)
        if idx is not None:
            self.hover_instance_idx = idx
            self.preview_mask = None
            self.view.set_highlight(idx)
            self.view.set_overlay(None, [inst.poly for inst in self.instances])
            return
        try:
            mask, score = self.segmenter.predict_from_point(x, y, positive=True)
        except Exception as e:
            self.info_label.setText(f"Prediction error: {e}")
            return
        self.preview_mask = mask
        self.hover_instance_idx = None
        self.view.set_highlight(None)
        self.view.set_overlay(mask, [inst.poly for inst in self.instances])
        # don't touch temp outline here (only set on left-click)


    def on_left_click(self, x: int, y: int):
        # 1) Click on existing instance → select/edit it
        hit_idx = self._instance_at_point(x, y)
        if hit_idx is not None:
            self.pending_inst_idx = hit_idx
            self.preview_locked = False
            self.preview_mask = None
            self.view.set_temp_poly(None)
            self.view.set_highlight(hit_idx)
            cls_id = self.instances[hit_idx].cls_id
            if 0 <= cls_id < self.class_list.count():
                self.class_list.setCurrentRow(cls_id)
            self.info_label.setText(f"Selected existing instance #{hit_idx} → class #{cls_id}. Change class to update.")
            return

        # 2) New click → lock a fresh preview and wait for class click to commit
        try:
            mask, score = self.segmenter.predict_from_point(x, y, positive=True)
        except Exception as e:
            self.info_label.setText(f"Prediction error: {e}")
            return

        self.preview_mask = mask
        self.preview_locked = True
        self.pending_inst_idx = None  # new mask; next class click will create a new instance

        poly = mask_to_polygon(mask, approx_eps=2.0)
        if poly is not None and len(poly) >= 3:
            self.view.set_temp_poly(poly)
        else:
            self.view.set_temp_poly(None)

        # Clear current class selection so user explicitly picks one
        self.class_list.clearSelection()
        self.class_list.setCurrentItem(None)

        self.view.set_highlight(None)
        self.view.set_overlay(mask, [inst.poly for inst in self.instances])
        self.info_label.setText(f"Preview locked (score={score:.2f}). Click a class to apply.")

    # ---------- Undo/Redo ----------
    def _push_undo(self):
        snapshot = [Instance(poly=inst.poly.copy(), cls_id=inst.cls_id) for inst in self.instances]
        self.undo_stack.append(snapshot)
        self.redo_stack.clear()

    def undo_last(self):
        if not self.instances:
            return
        self._push_undo()
        self.instances.pop()
        self.view.set_highlight(None)
        self.view.set_overlay(self.preview_mask, [inst.poly for inst in self.instances])

    def undo(self):
        if not self.undo_stack:
            return
        prev = self.undo_stack.pop()
        current = [Instance(poly=inst.poly.copy(), cls_id=inst.cls_id) for inst in self.instances]
        self.redo_stack.append(current)
        self.instances = [Instance(poly=inst.poly.copy(), cls_id=inst.cls_id) for inst in prev]
        self.view.set_highlight(None)
        self.view.set_overlay(self.preview_mask, [inst.poly for inst in self.instances])

    def redo(self):
        if not self.redo_stack:
            return
        nxt = self.redo_stack.pop()
        self._push_undo()
        self.instances = [Instance(poly=inst.poly.copy(), cls_id=inst.cls_id) for inst in nxt]
        self.view.set_highlight(None)
        self.view.set_overlay(self.preview_mask, [inst.poly for inst in self.instances])


    def _instance_at_point(self, x: int, y: int) -> Optional[int]:
        # unchanged: point-in-polygon hit test
        for i, inst in enumerate(self.instances):
            poly = inst.poly.reshape(-1, 1, 2).astype(np.int32)
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return i
        return None

    def _mask_iou_with_instance(self, mask: np.ndarray, inst: Instance) -> float:
        """IoU between preview mask (HxW bool) and an instance polygon rasterized to same size."""
        h, w = mask.shape[:2]
        poly = inst.poly.astype(np.int32).reshape(-1, 1, 2)
        raster = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(raster, [poly], 1)
        inter = np.logical_and(mask, raster.astype(bool)).sum()
        union = np.logical_or(mask,  raster.astype(bool)).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    def _find_overlapping_instance(self, mask: np.ndarray, iou_thresh: float = 0.6) -> Optional[int]:
        """Return index of an existing instance that significantly overlaps the preview mask."""
        best_iou, best_idx = 0.0, None
        for i, inst in enumerate(self.instances):
            iou = self._mask_iou_with_instance(mask, inst)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        return best_idx if (best_idx is not None and best_iou >= iou_thresh) else None


    # ---------- Save ----------
    def save_yolo(self):
        if self.image_bgr is None:
            return
        img_path = self.images[self.img_idx]
        h, w = self.image_bgr.shape[:2]
        lines = []
        for inst in self.instances:
            poly_norm = normalize_poly(inst.poly, w, h)
            flat = poly_norm.reshape(-1)
            coords = " ".join(f"{v:.6f}" for v in flat)
            lines.append(f"{inst.cls_id} {coords}")
        label_path = os.path.splitext(img_path)[0] + ".txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.info_label.setText(f"Saved {len(lines)} instances → {os.path.basename(label_path)}")

    # ---------- Navigation ----------
    def prev_image(self):
        if self.img_idx > 0:
            self.img_idx -= 1
            self._load_current_image()

    def next_image(self):
        if self.img_idx < len(self.images) - 1:
            self.img_idx += 1
            self._load_current_image()

    # ---------- Classes ----------
    def add_class(self):
        name = self.new_class_edit.text().strip()
        if not name:
            return
        self.class_list.addItem(name)
        if self.class_list.currentRow() < 0:
            self.class_list.setCurrentRow(self.class_list.count() - 1)
        self.new_class_edit.clear()


# -------------------- ImageView widget --------------------
class ImageView(QtWidgets.QLabel):
    """Aspect-fit image view that keeps masks/polys in sync.
       - Draws the image scaled to fit while preserving aspect.
       - Draws overlays scaled into the same displayed rect.
       - Emits mouse positions converted to IMAGE coordinates.
    """
    mouse_moved = QtCore.pyqtSignal(int, int)   # image coords
    left_clicked = QtCore.pyqtSignal(int, int)  # image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
        self._img_rgb: Optional[np.ndarray] = None  # HxWx3 RGB
        self._img_w = 0
        self._img_h = 0
        self.overlay_mask: Optional[np.ndarray] = None  # HxW bool (image size)
        self.polys: List[np.ndarray] = []              # image coords
        self.highlight_idx: Optional[int] = None
        self.temp_poly: Optional[np.ndarray] = None
        # cached rendering
        self._canvas = QtGui.QPixmap()   # full label size
        self._display_rect = QtCore.QRect()  # where the image is drawn inside the label

    # ----------- public API -----------
    def set_image(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._img_rgb = rgb
        self._img_h, self._img_w = rgb.shape[:2]
        self._invalidate_and_repaint()
    def set_temp_poly(self, poly: Optional[np.ndarray]):
        self.temp_poly = poly
        self._invalidate_and_repaint()
        
    def set_overlay(self, mask: Optional[np.ndarray], polys: List[np.ndarray]):
        self.overlay_mask = mask
        self.polys = polys
        self._invalidate_and_repaint()

    def set_highlight(self, idx: Optional[int]):
        self.highlight_idx = idx
        self._invalidate_and_repaint()

    # ----------- render pipeline -----------
    def resizeEvent(self, ev: QtGui.QResizeEvent):
        super().resizeEvent(ev)
        self._invalidate_and_repaint()

    def _invalidate_and_repaint(self):
        """Repaint the label: draw base image, then mask, then polygons, then temp outline."""
        if self._img_rgb is None or self.width() <= 1 or self.height() <= 1:
            return

        # Prepare a canvas the size of the label
        canvas_size = QtCore.QSize(self.width(), self.height())
        if self._canvas.size() != canvas_size:
            self._canvas = QtGui.QPixmap(canvas_size)
        # Clear to window color
        self._canvas.fill(self.palette().color(QtGui.QPalette.Window))

        painter = QtGui.QPainter(self._canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        # Compute aspect-fit rect for the image within the label
        img_w, img_h = self._img_w, self._img_h
        view_w, view_h = self.width(), self.height()
        if img_w == 0 or img_h == 0:
            painter.end()
            self.setPixmap(self._canvas)
            return

        scale = min(view_w / img_w, view_h / img_h)
        disp_w = int(round(img_w * scale))
        disp_h = int(round(img_h * scale))
        off_x = (view_w - disp_w) // 2
        off_y = (view_h - disp_h) // 2
        self._display_rect = QtCore.QRect(off_x, off_y, disp_w, disp_h)

        # --- Draw the BASE IMAGE first ---
        # Ensure contiguous memory for QImage
        rgb = np.ascontiguousarray(self._img_rgb)
        qimg = QtGui.QImage(rgb.data, img_w, img_h, 3 * img_w, QtGui.QImage.Format_RGB888)
        painter.drawImage(self._display_rect, qimg)

        # scale factors for overlays
        sx = disp_w / img_w
        sy = disp_h / img_h

        # 1) Draw hover/locked mask first (so outlines appear on top)
        if self.overlay_mask is not None:
            if self.overlay_mask.shape[:2] != (img_h, img_w):
                mask_u8 = (self.overlay_mask.astype(np.uint8) * 255)
                mask_u8 = cv2.resize(mask_u8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_u8 = (self.overlay_mask.astype(np.uint8) * 255)

            mask_rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            mask_rgba[mask_u8 > 0] = [255, 0, 0, 80]  # red with alpha

            qmask = QtGui.QImage(mask_rgba.data, img_w, img_h, 4 * img_w, QtGui.QImage.Format_RGBA8888)
            painter.drawImage(self._display_rect, qmask)

        # Helper to draw a polygon with a given pen
        def _draw_poly(poly: np.ndarray, pen: QtGui.QPen):
            if poly is None or len(poly) < 2:
                return
            pts = np.stack([
                off_x + poly[:, 0] * sx,
                off_y + poly[:, 1] * sy
            ], axis=1).astype(np.float32)
            painter.setPen(pen)
            path = QtGui.QPainterPath()
            path.moveTo(float(pts[0, 0]), float(pts[0, 1]))
            for j in range(1, len(pts)):
                path.lineTo(float(pts[j, 0]), float(pts[j, 1]))
            path.closeSubpath()
            painter.drawPath(path)

        # 2) Draw confirmed polygons (green, golden if highlighted)
        for i, poly in enumerate(self.polys):
            if poly is None or len(poly) < 2:
                continue
            pen = QtGui.QPen(QtGui.QColor(255, 215, 0) if (self.highlight_idx is not None and i == self.highlight_idx) else QtGui.QColor(0, 255, 0))
            pen.setWidth(3 if (self.highlight_idx is not None and i == self.highlight_idx) else 2)
            _draw_poly(poly, pen)

        # 3) Draw temporary (locked preview) polygon outline (blue dashed)
        if getattr(self, 'temp_poly', None) is not None and len(self.temp_poly) >= 2:
            pen = QtGui.QPen(QtGui.QColor(0, 153, 255))
            pen.setWidth(2)
            pen.setStyle(QtCore.Qt.DashLine)
            _draw_poly(self.temp_poly, pen)

        painter.end()
        self.setPixmap(self._canvas)


    # ----------- coordinate transforms -----------
    def _view_to_image(self, pos: QtCore.QPoint) -> Optional[Tuple[int, int]]:
        """Map label coords -> image coords. Return None if outside displayed image."""
        if self._img_rgb is None or self._display_rect.isNull():
            return None
        if not self._display_rect.contains(pos):
            return None
        x = (pos.x() - self._display_rect.left())
        y = (pos.y() - self._display_rect.top())
        # scale back to image coordinates
        sx = self._img_w / self._display_rect.width()
        sy = self._img_h / self._display_rect.height()
        ix = int(np.clip(round(x * sx), 0, self._img_w - 1))
        iy = int(np.clip(round(y * sy), 0, self._img_h - 1))
        return ix, iy

    # ----------- mouse events -----------
    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        mapped = self._view_to_image(ev.pos())
        if mapped is not None:
            self.mouse_moved.emit(mapped[0], mapped[1])
        else:
            # Signal out-of-bounds with -1,-1 so caller can clear preview
            self.mouse_moved.emit(-1, -1)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton:
            mapped = self._view_to_image(ev.pos())
            if mapped is not None:
                self.left_clicked.emit(mapped[0], mapped[1])
        super().mousePressEvent(ev)



# -------------------- Theming --------------------

def is_dark_mode(app: QtWidgets.QApplication) -> bool:
    # Heuristic: use the palette window color lightness
    pal = app.palette()
    win_color = pal.color(QtGui.QPalette.Window)
    # Convert to perceived luminance
    r, g, b, _ = win_color.getRgb()
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 128



# -------------------- Main --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images", help="Folder with images to annotate")
    ap.add_argument("--checkpoint", default="models/sam2.1_hiera_large.pt", help="Path to SAM2/SAM checkpoint .pth")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 (e.g., sam2_hiera_l) or SAM (e.g., vit_h)")
    ap.add_argument("--classes", nargs="*", default=["A1","A2","A3","A4","A5","A6","B1","B2","B3","B4","B5",
                                                     "B6","C1","C2","C3","C4","C5","C6"], help="Initial class names")
    return ap.parse_args()


def main():
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")  # consistent across platforms

    seg = Segmenter(args.checkpoint, args.config)
    win = Annotator(args.images, seg, args.classes)
    # Default window size
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()