"""Tiempos de entrenamiento separados del resto del run (indicación del docente, U4S2).

Uso: python src/flores/tiempos.py data/flower_photos outputs
Para cada línea base mide por separado: extracción de rasgos (E01, E03), selección de C con
la rejilla en validación, ajuste final con el C elegido y, en E02, la suma de las 30 épocas
(entrenamiento con aumentación y evaluación en validación por época). Escribe
outputs/tiempos_entrenamiento.json. CPU, mismos hilos que la corrida oficial.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flores import modelos  # noqa: E402
from flores.experimentos import Datos  # noqa: E402

raiz, out = Path(sys.argv[1]), Path(sys.argv[2])
d = Datos(raiz, out / "particion.csv")
tr, va = d.idx["train"], d.idx["val"]
rejilla = [0.01, 0.1, 1.0, 10.0]
res = {"cpu_hilos": os.cpu_count()}


def lineal(x, c_elegido):
    t = time.perf_counter()
    modelos.regresion_logistica(x[tr], d.y[tr], x[va], d.y[va], rejilla)
    sel = time.perf_counter() - t
    t = time.perf_counter()
    make_pipeline(StandardScaler(), LogisticRegression(C=c_elegido, max_iter=3000)).fit(x[tr], d.y[tr])
    return sel, time.perf_counter() - t


t = time.perf_counter()
x1 = np.stack([modelos.descriptores_clasicos(im) for im in d.imgs])
ext1 = time.perf_counter() - t
c1 = json.loads((out / "E01_resultados.json").read_text())["extra"]["C_elegido"]
sel1, fit1 = lineal(x1, c1)
res["E01"] = {"extraccion_s": ext1, "seleccion_C_s": sel1, "ajuste_final_s": fit1}

res["E02"] = {}
for s in (11, 22, 33):
    cur = pd.read_csv(out / f"E02_s{s}_curvas.csv")
    res["E02"][f"s{s}"] = {"entrenamiento_30_epocas_s": float(cur.seg.sum())}
res["E02"]["media_s"] = float(np.mean([v["entrenamiento_30_epocas_s"] for v in res["E02"].values()]))

torch.set_num_threads(os.cpu_count() or 2)
vit, tf, _ = modelos.dinov2("vit_small_patch14_dinov2.lvd142m", 224)
t = time.perf_counter()
x3 = modelos.extraer(vit, d.imgs, tf)
ext3 = time.perf_counter() - t
c3 = json.loads((out / "E03_resultados.json").read_text())["extra"]["C_elegido"]
sel3, fit3 = lineal(x3, c3)
res["E03"] = {"extraccion_embeddings_s": ext3, "seleccion_C_s": sel3, "ajuste_final_s": fit3,
              "embeddings_iguales_a_corrida_oficial": bool(np.allclose(x3, np.load(out / "E03_embeddings.npy"), atol=1e-4))}
(out / "tiempos_entrenamiento.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
