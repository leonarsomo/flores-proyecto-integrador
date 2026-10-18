"""Métricas de calidad, incertidumbre y eficiencia del proyecto (funciones sin estado)."""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, recall_score)

CLASES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def ece(proba: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Error de calibración esperado (Guo et al., 2017) con contenedores de igual ancho."""
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def calidad(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Métrica primaria (macro-F1) y secundarias sobre un conjunto etiquetado."""
    pred = proba.argmax(1)
    out = {"accuracy": accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro"),
           "balanced_acc": balanced_accuracy_score(y, pred),
           "log_loss": log_loss(y, np.clip(proba, 1e-7, 1), labels=range(len(CLASES))),
           "ece": ece(proba, y)}
    rec = recall_score(y, pred, average=None, labels=range(len(CLASES)))
    out.update({f"recall_{c}": r for c, r in zip(CLASES, rec)})
    return {k: float(v) for k, v in out.items()}


def ic95_por_grupos(y, pred, grupos, fn, n: int = 2000, seed: int = 2026) -> tuple[float, float]:
    """IC 95 % por remuestreo de grupos de fuga (autores), coherente con la partición."""
    rng = np.random.default_rng(seed)
    unicos = np.unique(grupos)
    idx_por_g = {g: np.where(grupos == g)[0] for g in unicos}
    stats = [fn(y[i], pred[i]) for i in
             (np.concatenate([idx_por_g[g] for g in rng.choice(unicos, len(unicos))])
              for _ in range(n))]
    return tuple(float(v) for v in np.percentile(stats, [2.5, 97.5]))


def mcnemar_exacto(acierta_a: np.ndarray, acierta_b: np.ndarray) -> dict:
    """Prueba exacta de McNemar sobre aciertos pareados en el mismo conjunto de prueba."""
    b = int((acierta_a & ~acierta_b).sum())
    c = int((~acierta_a & acierta_b).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"solo_A_acierta": b, "solo_B_acierta": c, "p_valor": float(p)}


def latencia_ms(fn, x, reps: int = 50, warmup: int = 10) -> dict[str, float]:
    """Latencia por imagen (lote 1) en CPU, con calentamiento: P50 y P95 en ms."""
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - s) * 1e3)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}
