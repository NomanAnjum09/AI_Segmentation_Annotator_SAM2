import numpy as np
from dataclasses import dataclass


@dataclass
class Instance:
    poly: np.ndarray  # Nx2 int (image coords)
    cls_id: int
    inst_id: int = -1