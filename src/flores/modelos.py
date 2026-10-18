"""Componentes de modelado: descriptores clásicos, CNN pequeña, aumentación y DINOv2."""
from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def descriptores_clasicos(img: Image.Image) -> np.ndarray:
    """Histograma HSV 16x4x4 normalizado + HOG (9 orientaciones, celdas 16 px) a 128x128."""
    small = img.resize((128, 128), Image.BILINEAR)
    hsv = np.asarray(small.convert("HSV"))
    hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(16, 4, 4),
                             range=((0, 256), (0, 256), (0, 256)))
    hist = hist.ravel() / hist.sum()
    gris = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    h = hog(gris, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return np.concatenate([hist, h]).astype(np.float32)


def regresion_logistica(xtr, ytr, xva, yva, rejilla_c) -> tuple:
    """Elige C por macro-F1 en validación; el escalador se ajusta solo con entrenamiento."""
    mejor = None
    for c in rejilla_c:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
        clf.fit(xtr, ytr)
        f1 = f1_score(yva, clf.predict(xva), average="macro")
        if mejor is None or f1 > mejor[0]:
            mejor = (f1, c, clf)
    return mejor[2], mejor[1]


class SmallCNN(nn.Module):
    """CNN del tutorial de TensorFlow usada en la U2A4: 3 bloques conv + densa 128."""

    def __init__(self, n_clases: int = 5, lado: int = 96):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.2), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 * (lado // 8) ** 2, 128), nn.ReLU(),
                                  nn.Linear(128, n_clases))

    def forward(self, x):
        return self.head(self.features(x))


def aumentar(x: torch.Tensor, g: torch.Generator, rot: float = 0.1, zoom: float = 0.2):
    """Aumentación combinada de la U2A4: volteo H, rotación ±rot·2π y zoom ±zoom."""
    n = x.shape[0]
    flip = torch.rand(n, generator=g) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    ang = (torch.rand(n, generator=g) * 2 - 1) * rot * 2 * np.pi
    escala = 1 + (torch.rand(n, generator=g) * 2 - 1) * zoom
    cos, sin = torch.cos(ang) / escala, torch.sin(ang) / escala
    cero = torch.zeros(n)
    theta = torch.stack([torch.stack([cos, -sin, cero], 1),
                         torch.stack([sin, cos, cero], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)


def dinov2(nombre: str, lado: int):
    """Backbone DINOv2 congelado (timm) y su transformación de entrada documentada."""
    vit = timm.create_model(nombre, pretrained=True, num_classes=0, img_size=lado).eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_data_config({}, model=vit)
    cfg.update(input_size=(3, lado, lado), crop_pct=lado / 256)
    return vit, timm.data.create_transform(**cfg), cfg


@torch.inference_mode()
def extraer(modelo, imgs, tf, lote: int = 32) -> np.ndarray:
    """Embeddings (token CLS tras la norma final) para una lista de imágenes PIL."""
    return np.concatenate([modelo(torch.stack([tf(im) for im in imgs[i:i + lote]])).numpy()
                           for i in range(0, len(imgs), lote)])
