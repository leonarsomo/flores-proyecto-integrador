"""Diagnóstico complementario de la línea base E03 (DINOv2 congelado + regresión logística).

1. Validación cruzada agrupada por autor (StratifiedGroupKFold, k=5) sobre train+val:
   estima la varianza entre grupos que una sola partición no muestra. Prueba no se toca.
2. Robustez en prueba ante cuatro perturbaciones de captura con celular (severidad media),
   con el clasificador ya ajustado: desenfoque, compresión JPEG, subexposición y baja resolución.
Uso: python src/flores/diagnostico.py data/flower_photos outputs
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flores import modelos  # noqa: E402

CLASES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
PERTURBACIONES = {
    "desenfoque_gaussiano_r2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "jpeg_calidad_20": lambda im: _jpeg(im, 20),
    "subexposicion_x0.5": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
    "baja_resolucion_x0.25": lambda im: im.resize((max(1, im.width // 4), max(1, im.height // 4)),
                                                  Image.BILINEAR).resize(im.size, Image.BILINEAR),
}


def _jpeg(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main(raiz: Path, out: Path) -> None:
    p = pd.read_csv(out / "particion.csv")
    y = p.label.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    x = np.load(out / "E03_embeddings.npy")
    dev = np.where(p.split != "test")[0]
    te = np.where(p.split == "test")[0]
    res: dict = {}

    # 1. CV agrupada por autor sobre train+val (C fijo = el elegido en validación)
    c_elegido = json.loads((out / "E03_resultados.json").read_text())["extra"]["C_elegido"]
    f1s = []
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(
            x[dev], y[dev], p.grupo.to_numpy()[dev]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c_elegido, max_iter=3000))
        clf.fit(x[dev][tr], y[dev][tr])
        f1s.append(f1_score(y[dev][va], clf.predict(x[dev][va]), average="macro"))
    res["cv_agrupada_k5"] = {"macro_f1_pliegues": [float(v) for v in f1s],
                             "media": float(np.mean(f1s)), "de": float(np.std(f1s, ddof=1))}

    # 2. Robustez en prueba
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c_elegido, max_iter=3000))
    tr = np.where(p.split == "train")[0]
    clf.fit(x[tr], y[tr])
    limpio = f1_score(y[te], clf.predict(x[te]), average="macro")
    torch.set_num_threads(2)
    vit, tf, _ = modelos.dinov2("vit_small_patch14_dinov2.lvd142m", 224)
    imgs = []
    for f in p.file.to_numpy()[te]:
        with Image.open(raiz / f) as im:
            imgs.append(im.convert("RGB"))
    rob = {"limpio": float(limpio)}
    for nombre, fn in PERTURBACIONES.items():
        xp = modelos.extraer(vit, [fn(im) for im in imgs], tf)
        rob[nombre] = float(f1_score(y[te], clf.predict(xp), average="macro"))
        print(nombre, round(rob[nombre], 4), flush=True)
    rob["peor_caso"] = min(v for k, v in rob.items() if k != "limpio")
    rob["caida_relativa_peor_%"] = 100 * (limpio - rob["peor_caso"]) / limpio
    res["robustez_prueba_E03"] = rob
    (out / "E03_diagnostico.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
