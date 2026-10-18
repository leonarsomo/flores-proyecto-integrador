"""Pruebas unitarias de las métricas propias."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flores.metricas import calidad, ece, mcnemar_exacto  # noqa: E402


def test_ece_perfectamente_calibrado():
    y = np.array([0, 1] * 50)
    proba = np.eye(2)[y]
    assert ece(proba, y) == 0.0


def test_macro_f1_clase_mayoritaria():
    y = np.repeat(np.arange(5), 20)
    proba = np.tile(np.eye(5)[0], (100, 1))
    m = calidad(y, proba)
    assert abs(m["accuracy"] - 0.2) < 1e-9 and m["macro_f1"] < 0.1


def test_mcnemar_simetrico():
    a = np.array([True, False] * 10)
    r = mcnemar_exacto(a, ~a)
    assert r["solo_A_acierta"] == r["solo_B_acierta"] == 10 and r["p_valor"] == 1.0
