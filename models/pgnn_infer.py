import torch
import numpy as np
from models.pgnn import HeatSinkPGNN
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # go up one level
CKPT_PATH = BASE_DIR / "pgnn.pt"

_ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)


_model = HeatSinkPGNN()
_model.load_state_dict(_ckpt["model"])
_model.eval()

_mean = _ckpt["mean"]
_std = _ckpt["std"]

def pgnn_predict(x: np.ndarray) -> float:
    x = (x - _mean) / _std
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return float(_model(x).item())
