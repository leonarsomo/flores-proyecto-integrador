"""Predicción selectiva y comparaciones pareadas de E03 (sin reentrenar el tronco).

1. Umbral de confianza elegido en VALIDACIÓN para precisión >= 99 % en lo aceptado;
   se reporta la cobertura y la precisión resultantes en prueba.
2. McNemar exacto de E03 frente a E01 y a cada semilla de E02 sobre las mismas fotos.
Uso: python src/flores/seleccion.py outputs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flores.metricas import mcnemar_exacto  # noqa: E402

CLASES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
PRECISION_OBJETIVO = 0.99


def umbral_por_precision(proba: np.ndarray, y: np.ndarray, objetivo: float) -> float:
    """Menor umbral de confianza cuya precisión en lo aceptado alcanza el objetivo."""
    for u in np.linspace(0.30, 0.999, 700):
        m = proba.max(1) >= u
        if m.sum() and (proba.argmax(1)[m] == y[m]).mean() >= objetivo:
            return float(u)
    return 1.0


def main(out: Path) -> None:
    p = pd.read_csv(out / "particion.csv")
    y = p.label.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    tr, va, te = ((p.split == s).to_numpy() for s in ("train", "val", "test"))
    emb = np.load(out / "E03_embeddings.npy")
    c = json.loads((out / "E03_resultados.json").read_text())["extra"]["C_elegido"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000)).fit(emb[tr], y[tr])
    pv, pt = clf.predict_proba(emb[va]), np.load(out / "E03_proba_test.npy")
    assert np.allclose(pt, clf.predict_proba(emb[te]), atol=1e-6), "E03 no se reproduce"
    u = umbral_por_precision(pv, y[va], PRECISION_OBJETIVO)
    mt = pt.max(1) >= u
    res = {"seleccion": {"precision_objetivo_val": PRECISION_OBJETIVO, "umbral_val": u,
                         "cobertura_val": float((pv.max(1) >= u).mean()),
                         "cobertura_test": float(mt.mean()),
                         "precision_aceptadas_test": float((pt.argmax(1)[mt] == y[te][mt]).mean())}}
    ok03 = pt.argmax(1) == y[te]
    res["mcnemar_E03_vs"] = {
        k: mcnemar_exacto(ok03, np.load(out / f"{k}_proba_test.npy").argmax(1) == y[te])
        for k in ("E01", "E02_s11", "E02_s22", "E02_s33")}
    (out / "E03_seleccion_y_comparaciones.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
