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


    # ----------- public API -----------

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