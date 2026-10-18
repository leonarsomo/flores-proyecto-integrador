"""Auditoría inicial de calidad de tf_flowers (flower_photos.tgz).

Genera out/inventario.csv con una fila por imagen: clase, autor de Flickr,
dimensiones, hashes (bytes, píxeles, perceptual), métricas de calidad técnica.
Uso: python code/a_auditoria.py data/flower_photos out
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve, laplace

LICENSE_RE = re.compile(
    r"^(?P<file>\S+\.jpg) CC-BY by (?P<author>.*?) - "
    r"https://www\.flickr\.com/photos/(?P<owner>[^/]+)/(?P<photo>\d+)/?\s*$"
)
# Núcleo de Immerkær (1996) para estimar la desviación del ruido gaussiano.
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)


def parse_license(path: Path) -> pd.DataFrame:
    """Lee LICENSE.txt y devuelve archivo, autor, identificador de Flickr y foto."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LICENSE_RE.match(line.strip())
        if m:
            rows.append(m.groupdict())
    return pd.DataFrame(rows)


def noise_sigma(gray: np.ndarray) -> float:
    """Estimador de ruido de Immerkær sobre la imagen en escala de grises."""
    h, w = gray.shape
    conv = np.abs(convolve(gray, IMMERKAER))
    return float(np.sqrt(np.pi / 2) * conv[1:-1, 1:-1].sum() / (6 * (w - 2) * (h - 2)))


def describe(path: Path) -> dict:
    """Calcula hashes y métricas de calidad de una imagen."""
    raw = path.read_bytes()
    with Image.open(path) as im:
        mode, (w, h) = im.mode, im.size
        rgb = im.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float64)
    gray = arr @ np.array([0.299, 0.587, 0.114])
    hsv_s = np.asarray(rgb.convert("HSV"), dtype=np.float64)[..., 1] / 255.0
    return {
        "mode": mode,
        "width": w,
        "height": h,
        "md5_bytes": hashlib.md5(raw).hexdigest(),
        "md5_pixels": hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest(),
        "phash": str(imagehash.phash(rgb)),
        "saturation_mean": float(hsv_s.mean()),
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "laplacian_var": float(laplace(gray).var()),
        "noise_sigma": noise_sigma(gray),
    }


def main(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lic = parse_license(root / "LICENSE.txt")
    files = sorted(p for p in root.glob("*/*.jpg"))
    rows = []
    for p in files:
        rel = f"{p.parent.name}/{p.name}"
        rows.append({"file": rel, "label": p.parent.name, **describe(p)})
    inv = pd.DataFrame(rows).merge(lic, on="file", how="left", validate="one_to_one")
    inv.to_csv(out / "inventario.csv", index=False)
    print(f"imágenes: {len(inv)}  con licencia: {inv['owner'].notna().sum()}  "
          f"líneas de licencia: {len(lic)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
