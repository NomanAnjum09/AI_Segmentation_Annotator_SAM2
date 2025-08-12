from typing import Optional
import numpy as np
import cv2

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