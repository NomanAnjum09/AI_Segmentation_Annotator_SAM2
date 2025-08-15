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

        self.zoom = 1.0           # 1.0 = fit-to-view baseline
        self.min_zoom = 0.1
        self.max_zoom = 8.0
        self.user_zoomed = False   # becomes True after first manual zoom
        # editable segments
        self.edit_mode = False
        self.edit_poly: Optional[np.ndarray] = None     # Nx2 (image coords)
        self.drag_vtx = -1                               # active vertex index while dragging
        self.dragging_whole = False
        self._last_cursor_img = None                     # (x,y) in image coords
        self.handle_px = 8


    # ----------- public API -----------
        # ---------- edit mode public API ----------
    def begin_edit(self, poly: np.ndarray):
        """Enter edit mode with a copy of poly (image coords)."""
        if poly is None or len(poly) < 3:
            return
        self.edit_mode = True
        self.edit_poly = poly.copy().astype(np.float32)
        self.drag_vtx = -1
        self.dragging_whole = False
        self._invalidate_and_repaint()

    def commit_edit(self) -> Optional[np.ndarray]:
        """Exit edit mode and return edited polygon (int image coords)."""
        if not self.edit_mode or self.edit_poly is None:
            return None
        out = self.edit_poly.copy().astype(np.int32)
        self.edit_mode = False
        self.edit_poly = None
        self.drag_vtx = -1
        self.dragging_whole = False
        self._invalidate_and_repaint()
        return out

    def cancel_edit(self):
        """Exit edit mode and discard changes."""
        self.edit_mode = False
        self.edit_poly = None
        self.drag_vtx = -1
        self.dragging_whole = False
        self._invalidate_and_repaint()

    def insert_vertex_at(self, x_img: int, y_img: int):
        """Insert a vertex at the closest point on the nearest edge (image coords)."""
        if not self.edit_mode or self.edit_poly is None or len(self.edit_poly) < 1:
            return
        p = np.array([x_img, y_img], dtype=np.float32)
        poly = self.edit_poly
        # find nearest segment
        best_i = -1; best_d2 = 1e30; best_q = None
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i+1) % len(poly)]
            ab = b - a
            t = float(np.dot(p - a, ab)) / float(np.dot(ab, ab) + 1e-6)
            t = np.clip(t, 0.0, 1.0)
            q = a + t * ab
            d2 = float(np.sum((p - q) ** 2))
            if d2 < best_d2:
                best_d2 = d2; best_i = i; best_q = q
        if best_i >= 0 and best_q is not None:
            self.edit_poly = np.insert(poly, best_i + 1, best_q, axis=0)
            self._invalidate_and_repaint()

    def delete_nearest_vertex(self):
        """Delete the vertex nearest to the cursor (edit mode only; keep >=3 points)."""
        if not self.edit_mode or self.edit_poly is None or len(self.edit_poly) <= 3:
            return
        if self._last_cursor_img is None:
            return
        p = np.array(self._last_cursor_img, dtype=np.float32)
        d2 = np.sum((self.edit_poly - p) ** 2, axis=1)
        i = int(np.argmin(d2))
        if len(self.edit_poly) > 3:
            self.edit_poly = np.delete(self.edit_poly, i, axis=0)
            self._invalidate_and_repaint()


    def cursor_image_pos(self) -> Optional[Tuple[int,int]]:
        return None if self._last_cursor_img is None else (int(self._last_cursor_img[0]), int(self._last_cursor_img[1]))


    def set_zoom(self, z: float):
        self.zoom = float(np.clip(z, self.min_zoom, self.max_zoom))
        self.user_zoomed = True
        self._invalidate_and_repaint()

    def zoom_in(self, step: float = 0.2):
        self.set_zoom(self.zoom * (1.0 + step))

    def zoom_out(self, step: float = 0.2):
        self.set_zoom(self.zoom / (1.0 + step))

    def reset_zoom(self):
        self.zoom = 1.0
        self.user_zoomed = False
        self._invalidate_and_repaint()


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

    def clear_all_overlays(self):
        """Drop every overlay/highlight/temp and force a repaint now."""
        self.overlay_mask = None
        self.polys = []
        self.temp_poly = None
        self.highlight_idx = None
        self._invalidate_and_repaint()
        self.update()
        self.repaint()
        QtWidgets.QApplication.processEvents()

    # ----------- render pipeline -----------
    def resizeEvent(self, ev: QtGui.QResizeEvent):
        super().resizeEvent(ev)
        # If the user hasn't manually zoomed, keep the image fitted (zoom=1.0 baseline)
        if not self.user_zoomed:
            self.zoom = 1.0
        self._invalidate_and_repaint()


    def _invalidate_and_repaint(self):
        if self._img_rgb is None or self.width() <= 1 or self.height() <= 1:
            # When inside a QScrollArea, width/height may be 0 before the first setFixedSize.
            # We can still proceed once we know image size.
            pass

        # If we have no image, bail early.
        if self._img_rgb is None:
            return

        # Canvas size is the widget size (will be setFixedSize below)
        view_w = self.parent().width() if self.parent() else self.width()
        view_h = self.parent().height() if self.parent() else self.height()

        img_w, img_h = self._img_w, self._img_h
        if img_w == 0 or img_h == 0:
            return

        # Compute fit scale and apply zoom (render-only)
        scale_fit = min(max(view_w, 1) / img_w, max(view_h, 1) / img_h)
        scale = scale_fit * getattr(self, "zoom", 1.0)

        disp_w = int(round(img_w * scale))
        disp_h = int(round(img_h * scale))

        # IMPORTANT: tell the scroll area how big the content is
        self.setFixedSize(disp_w, disp_h)
        self._display_rect = QtCore.QRect(0, 0, disp_w, disp_h)

        # Prepare canvas exactly the widget size
        canvas_size = QtCore.QSize(disp_w, disp_h)
        if self._canvas.size() != canvas_size:
            self._canvas = QtGui.QPixmap(canvas_size)
        self._canvas.fill(self.palette().color(QtGui.QPalette.Window))

        painter = QtGui.QPainter(self._canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        # Draw base image scaled to widget
        rgb = np.ascontiguousarray(self._img_rgb)
        qimg = QtGui.QImage(rgb.data, img_w, img_h, 3 * img_w, QtGui.QImage.Format_RGB888)
        painter.drawImage(self._display_rect, qimg)

        sx = disp_w / img_w
        sy = disp_h / img_h

        # Mask underneath
        if self.overlay_mask is not None:
            mask_u8 = (self.overlay_mask.astype(np.uint8) * 255)
            if mask_u8.shape[:2] != (img_h, img_w):
                mask_u8 = cv2.resize(mask_u8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            mask_rgba[mask_u8 > 0] = [255, 0, 0, 80]
            qmask = QtGui.QImage(mask_rgba.data, img_w, img_h, 4 * img_w, QtGui.QImage.Format_RGBA8888)
            painter.drawImage(self._display_rect, qmask)

        # Helper to draw a polygon (image coords → widget coords)
        def _draw_poly(poly: np.ndarray, pen: QtGui.QPen):
            if poly is None or len(poly) < 2:
                return
            pts = np.stack([poly[:, 0] * sx, poly[:, 1] * sy], axis=1).astype(np.float32)
            painter.setPen(pen)
            path = QtGui.QPainterPath()
            path.moveTo(float(pts[0, 0]), float(pts[0, 1]))
            for j in range(1, len(pts)):
                path.lineTo(float(pts[j, 0]), float(pts[j, 1]))
            path.closeSubpath()
            painter.drawPath(path)

        # Committed polys
        for i, poly in enumerate(self.polys):
            if poly is None or len(poly) < 2:
                continue
            pen = QtGui.QPen(QtGui.QColor(255, 215, 0) if (self.highlight_idx is not None and i == self.highlight_idx) else QtGui.QColor(0, 255, 0))
            pen.setWidth(3 if (self.highlight_idx is not None and i == self.highlight_idx) else 2)
            _draw_poly(poly, pen)

        # Temp outline on top
        if getattr(self, "temp_poly", None) is not None and len(self.temp_poly) >= 2:
            pen = QtGui.QPen(QtGui.QColor(0, 153, 255))
            pen.setWidth(2)
            pen.setStyle(QtCore.Qt.DashLine)
            _draw_poly(self.temp_poly, pen)


        if self.edit_mode and self.edit_poly is not None and len(self.edit_poly) >= 2:
            pen = QtGui.QPen(QtGui.QColor(0, 153, 255))
            pen.setWidth(2)
            pen.setStyle(QtCore.Qt.DashLine)
            _draw_poly(self.edit_poly.astype(np.float32), pen)

            # draw handles
            painter.setPen(QtGui.QPen(QtGui.QColor(0,0,0)))
            r = self.handle_px
            for pt in self.edit_poly:
                cx = float(pt[0] * sx)
                cy = float(pt[1] * sy)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0,153,255)))
                painter.drawEllipse(QtCore.QPointF(cx, cy), r, r)

        painter.end()
        self.setPixmap(self._canvas)



    # ----------- coordinate transforms -----------
    def _view_to_image(self, pos: QtCore.QPoint):
        if self._img_rgb is None:
            return None
        if not (0 <= pos.x() < self.width() and 0 <= pos.y() < self.height()):
            return None
        # Map widget coords → image coords using current drawn size
        sx = self._img_w / max(self.width(), 1)
        sy = self._img_h / max(self.height(), 1)
        ix = int(np.clip(round(pos.x() * sx), 0, self._img_w - 1))
        iy = int(np.clip(round(pos.y() * sy), 0, self._img_h - 1))
        return ix, iy


    # ----------- mouse events -----------
    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        mapped = self._view_to_image(ev.pos())
        if mapped is not None:
            self._last_cursor_img = mapped
            self.mouse_moved.emit(mapped[0], mapped[1])
        else:
            self._last_cursor_img = None
            self.mouse_moved.emit(-1, -1)
        # Drag in edit mode
        if self.edit_mode and self.edit_poly is not None and (self.drag_vtx >= 0 or self.dragging_whole):
            x, y = (mapped if mapped is not None else (None, None))
            if x is None:
                return
            if self.drag_vtx >= 0:
                self.edit_poly[self.drag_vtx] = [x, y]
            elif self.dragging_whole:
                # move by delta from previous move
                # Use event global position delta in image coords for stability
                # simpler: recompute from last anchor
                pass  # handled in press/move pair below
            self._invalidate_and_repaint()


    def _hit_vertex(self, x_img: int, y_img: int) -> int:
        """Return vertex index if (x,y) is near a vertex (threshold in widget px scaled)."""
        if not self.edit_mode or self.edit_poly is None:
            return -1
        # Convert hit radius from widget px to image px
        sx = self._img_w / max(self.width(), 1)
        sy = self._img_h / max(self.height(), 1)
        rx = self.handle_px * sx
        ry = self.handle_px * sy
        p = np.array([x_img, y_img], dtype=np.float32)
        d = np.abs(self.edit_poly - p)
        hit = np.where((d[:,0] <= rx) & (d[:,1] <= ry))[0]
        if hit.size == 0:
            # fallback true distance
            r = float(max(rx, ry))
            d2 = np.sum((self.edit_poly - p) ** 2, axis=1)
            i = int(np.argmin(d2))
            return i if d2[i] <= r*r else -1
        return int(hit[0])

    def _point_in_polygon(self, x:int, y:int, poly: np.ndarray) -> bool:
        path = poly.reshape(-1,1,2).astype(np.int32)
        return cv2.pointPolygonTest(path, (float(x), float(y)), False) >= 0

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        mapped = self._view_to_image(ev.pos())

        # --- Edit-mode behaviors ---
        if self.edit_mode and self.edit_poly is not None:
            if ev.button() == QtCore.Qt.LeftButton and mapped is not None:
                x, y = mapped
                v = self._hit_vertex(x, y)
                if v >= 0:
                    self.drag_vtx = v
                    self.dragging_whole = False
                    ev.accept(); return
                elif self._point_in_polygon(x, y, self.edit_poly):
                    self.dragging_whole = True
                    self._drag_anchor_img = np.array([x, y], dtype=np.float32)
                    self._poly_at_anchor = self.edit_poly.copy()
                    ev.accept(); return
                # Clicked elsewhere while editing → do nothing special; fall through

            elif ev.button() == QtCore.Qt.RightButton and mapped is not None:
                # insert vertex on edge
                self.insert_vertex_at(mapped[0], mapped[1])
                ev.accept(); return

        # --- Normal (non-edit) behavior: emit your signal ---
        if ev.button() == QtCore.Qt.LeftButton and mapped is not None:
            self.left_clicked.emit(mapped[0], mapped[1])
            ev.accept(); return

        # Otherwise default
        super().mousePressEvent(ev)


    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self.edit_mode and ev.button() == QtCore.Qt.LeftButton:
            self.drag_vtx = -1
            self.dragging_whole = False
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        # keep the earlier code plus dragging for whole polygon
        mapped = self._view_to_image(ev.pos())
        if mapped is not None:
            self._last_cursor_img = mapped
            self.mouse_moved.emit(mapped[0], mapped[1])
        else:
            self._last_cursor_img = None
            self.mouse_moved.emit(-1, -1)
        if self.edit_mode and self.edit_poly is not None:
            if self.drag_vtx >= 0 and mapped is not None:
                self.edit_poly[self.drag_vtx] = [mapped[0], mapped[1]]
                self._invalidate_and_repaint()
            elif self.dragging_whole and mapped is not None:
                delta = np.array([mapped[0], mapped[1]], dtype=np.float32) - self._drag_anchor_img
                self.edit_poly = (self._poly_at_anchor + delta).clip([0,0],[self._img_w-1,self._img_h-1])
                self._invalidate_and_repaint()
