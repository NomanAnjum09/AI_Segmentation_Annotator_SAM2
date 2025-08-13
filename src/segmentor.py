import numpy as np
import cv2   
from typing import Tuple

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



