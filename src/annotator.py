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
from typing import List, Optional
from src.segmentor import Segmenter
from src.image_view import ImageView
from src.helpers.dataclasses import Instance
from src.helpers.utils import mask_to_polygon, normalize_poly



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
        self.class_list.setMinimumHeight(320)
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
        # Put the image view inside a scroll area so it can scroll when zoomed
        self.img_scroll = QtWidgets.QScrollArea()
        self.img_scroll.setWidgetResizable(False)                 # we control the content size
        self.img_scroll.setAlignment(QtCore.Qt.AlignCenter)       # keep image centered when smaller
        self.img_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.img_scroll.setWidget(self.view)
        main.addWidget(self.img_scroll, 1)

        main.addWidget(right, 0)

        # Keyboard shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self.save_yolo)
        QtWidgets.QShortcut(QtGui.QKeySequence("Delete"), self, activated=self.undo_last)
        QtWidgets.QShortcut(QtGui.QKeySequence("Backspace"), self, activated=self.undo_last)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl++"), self, activated=self.view.zoom_in)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+="), self, activated=self.view.zoom_in)  # some keyboards
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+-"), self, activated=self.view.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+0"), self, activated=self.view.reset_zoom)
        QtWidgets.QShortcut(QtGui.QKeySequence("F11"),   self, activated=lambda: self.setWindowState(self.windowState() ^ QtCore.Qt.WindowFullScreen))
        QtWidgets.QShortcut(QtGui.QKeySequence("Return"), self, activated=self.commit_edit)
        QtWidgets.QShortcut(QtGui.QKeySequence("Enter"),  self, activated=self.commit_edit)
        def _esc_action():
            if self.view.edit_mode:
                self.cancel_edit()
            else:
                self.cancel_preview()
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, activated=_esc_action)

        # Insert/Delete vertex using last cursor pos
        QtWidgets.QShortcut(QtGui.QKeySequence("D"), self,
            activated=lambda: (self._edit_at_cursor("del")))
        QtWidgets.QShortcut(QtGui.QKeySequence("I"), self,
            activated=lambda: (self._edit_at_cursor("ins")))

        # Reasonable default/floor sizes
        self.setMinimumSize(900, 600)

        self.zoom_out_btn = QtWidgets.QPushButton("−")
        self.zoom_out_btn.setFixedWidth(36)
        self.zoom_in_btn  = QtWidgets.QPushButton("+")
        self.zoom_in_btn.setFixedWidth(36)

        self.zoom_out_btn.clicked.connect(lambda: (self.view.zoom_out(), self.info_label.setText("Zoom out")))
        self.zoom_in_btn.clicked.connect(lambda: (self.view.zoom_in(),  self.info_label.setText("Zoom in")))

        footer.addWidget(self.zoom_out_btn)   # add before Export
        footer.addWidget(self.zoom_in_btn)

        self.zoom_in_btn.clicked.connect(lambda: (self.view.zoom_in(), self.info_label.setText(f"Zoom {self.view.zoom*100:.0f}%")))
        self.zoom_out_btn.clicked.connect(lambda: (self.view.zoom_out(), self.info_label.setText(f"Zoom {self.view.zoom*100:.0f}%")))
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl++"), self, activated=lambda: (self.view.zoom_in(), self.info_label.setText(f"Zoom {self.view.zoom*100:.0f}%")))


    def _edit_at_cursor(self, op: str):
        if not self.view.edit_mode:
            return
        cur = self.view.cursor_image_pos()
        if cur is None:
            return
        x,y = cur
        if op == "del":
            self.view.delete_nearest_vertex(x, y)
            self.info_label.setText("Deleted nearest vertex.")
        elif op == "ins":
            self.view.insert_vertex_at(x, y)
            self.info_label.setText("Inserted vertex on nearest edge.")

    ## Load history from label file
    def _label_path_for(self, img_path: str) -> str:
        return os.path.splitext(img_path)[0] + ".txt"

    def _ensure_class_index(self, cls_id: int):
        """If labels contain a class id beyond current list, pad class list."""
        while cls_id >= self.class_list.count():
            self.class_list.addItem(f"Class{self.class_list.count()}")

    def load_yolo_labels(self, img_path: str, w: int, h: int) -> int:
        """Load YOLO-seg labels into self.instances. Returns count loaded."""
        label_path = self._label_path_for(img_path)
        if not os.path.exists(label_path):
            return 0

        loaded = []
        with open(label_path, "r", encoding="utf-8") as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except Exception:
                    continue
                coords = [float(x) for x in parts[1:]]
                if len(coords) < 6:  # need at least 3 points
                    continue
                # denormalize to image coords
                pts = []
                it = iter(coords)
                for x, y in zip(it, it):
                    ix = int(np.clip(round(x * w), 0, w - 1))
                    iy = int(np.clip(round(y * h), 0, h - 1))
                    pts.append([ix, iy])
                if len(pts) < 3:
                    continue
                poly = np.asarray(pts, dtype=np.int32)
                self._ensure_class_index(cls_id)
                loaded.append(Instance(poly=poly, cls_id=cls_id))

        self.instances = loaded
        # refresh next instance id
        self._next_inst_id = len(self.instances)
        return len(self.instances)



    # ---------- Image handling ----------
    def _collect_images(self, d: str) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if os.path.splitext(f.lower())[1] in exts]
        return files

    def _load_current_image(self):
        self.view.cancel_edit()
        img_path = self.images[self.img_idx]
        self.image_bgr = cv2.imread(img_path)
        if self.image_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        self.segmenter.set_image(self.image_bgr)
        self.view.set_image(self.image_bgr)

        # reset transient UI state
        self.undo_stack.clear(); self.redo_stack.clear()
        self.preview_mask = None; self.preview_point = None
        self.hover_instance_idx = None
        self.preview_locked = False
        self.pending_inst_idx = None
        self.view.set_highlight(None)

        # NEW: load any existing labels into instances
        h, w = self.image_bgr.shape[:2]
        n_loaded = self.load_yolo_labels(img_path, w, h)

        # show loaded polys
        self.view.set_temp_poly(None)
        self.view.set_overlay(None, [inst.poly for inst in self.instances])
        self.info_label.setText(f"Loaded {n_loaded} instances from {os.path.basename(self._label_path_for(img_path))}" if n_loaded else "No labels found; start annotating.")


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

            # BEGIN EDIT MODE
            self.view.begin_edit(self.instances[hit_idx].poly)
            cls_id = self.instances[hit_idx].cls_id
            if 0 <= cls_id < self.class_list.count():
                self.class_list.setCurrentRow(cls_id)
            self.info_label.setText(
                "Editing polygon: drag vertices; right-click edge to insert; "
                "D=delete vertex; Enter=apply; Esc=cancel."
            )
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

    
    def commit_edit(self):
        if self.pending_inst_idx is None:
            return
        edited = self.view.commit_edit()
        if edited is None or len(edited) < 3:
            self.info_label.setText("No edit to apply.")
            return
        self._push_undo()
        self.instances[self.pending_inst_idx].poly = edited
        self.view.set_overlay(None, [inst.poly for inst in self.instances])
        self.view.set_highlight(self.pending_inst_idx)
        self.info_label.setText(f"Applied edits to instance #{self.pending_inst_idx}.")

    def cancel_edit(self):
        if self.view.edit_mode:
            self.view.cancel_edit()
            self.view.set_overlay(None, [inst.poly for inst in self.instances])
            self.info_label.setText("Edit cancelled.")


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
        if self.view.edit_mode:
          self.info_label.setText("Finish (Enter) or cancel (Esc) the current edit before saving.")
          return
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