"""Calcula las métricas desagregadas de la prueba y rellena los marcadores {{...}} de la ficha y del README.

Fuente única: outputs/ (probabilidades guardadas por la fase final) y modelos/finalista.json.
Uso: PYTHONPATH=src python scripts/a2/rellenar_fichas.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from flores.ajuste_fino import CLASES
from sklearn.metrics import confusion_matrix

OUT, A2 = Path("outputs"), Path("outputs/a2")
def proportion_confint(k: int, n: int, method: str = "wilson", z: float = 1.959964):
    """Intervalo de Wilson del 95 % para una proporción k / n."""
    ph = k / n
    centro = (ph + z * z / (2 * n)) / (1 + z * z / n)
    medio = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centro - medio, centro + medio


c = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")  # noqa: E731


def main() -> None:
    p = pd.read_csv(OUT / "particion.csv")
    te = np.where(p.split == "test")[0]
    t = p.iloc[te].reset_index(drop=True)
    y = t.label.map({k: i for i, k in enumerate(CLASES)}).to_numpy()
    meta = json.loads(Path("modelos/finalista.json").read_text())
    sel = meta["origen"]
    r = json.loads((A2 / f"{sel}_resultados.json").read_text())
    pf, p3 = np.load(A2 / f"{sel}_proba_test.npy"), np.load(OUT / "E03_proba_test.npy")
    acc_f = pf.argmax(1) == y
    acept = pf.max(1) >= meta["umbral"]
    fotos_autor = t.owner.map(p.owner.value_counts())     # fotos del autor en todo el conjunto
    lado = t[["width", "height"]].min(axis=1)
    grupos = {
        "MARGARITA": y == 0, "DIENTE": y == 1, "ROSA": y == 2, "GIRASOL": y == 3, "TULIPAN": y == 4,
        "AUTOR_UNA": (fotos_autor == 1).to_numpy(), "AUTOR_VARIAS": (fotos_autor > 1).to_numpy(),
        "RES_BAJA": (lado < 200).to_numpy(), "RES_ALTA": (lado >= 200).to_numpy(),
        "MONO": (t.saturation_mean < 0.08).to_numpy(), "COLOR": (t.saturation_mean >= 0.08).to_numpy(),
        "OSCURA": (t.brightness_mean < 50).to_numpy(), "SOBREEXP": (t.brightness_mean > 200).to_numpy(),
        "ILUM_NORMAL": ((t.brightness_mean >= 50) & (t.brightness_mean <= 200)).to_numpy(),
    }
    v = {}
    filas = []
    for k, m in grupos.items():
        n, ok = int(m.sum()), int(acc_f[m].sum())
        lo, hi = proportion_confint(ok, n, method="wilson") if n else (np.nan, np.nan)
        e03 = float((p3.argmax(1)[m] == y[m]).mean()) if n else np.nan
        v[f"N_{k}"] = str(n)
        v[f"F_{k}"] = f"{c(ok / n)} ({c(lo)} a {c(hi)})" if n else "sin casos"
        v[f"C_{k}"] = f"{c(acept[m].mean() * 100, 1)} %" if n else "sin casos"
        v[f"E03_{k}"] = c(e03) if n else "sin casos"
        filas.append({"factor": k, "n": n, "exactitud_final": ok / n if n else np.nan, "ic_lo": lo, "ic_hi": hi,
                      "cobertura_final": float(acept[m].mean()) if n else np.nan, "exactitud_E03": e03})
    pd.DataFrame(filas).to_csv(A2 / "tablas" / "desagregado_prueba.csv", index=False)
    claves = {"baja_resolucion_x0.25": "BAJA_RES", "desenfoque_gaussiano_r2": "DESENFOQUE",
              "jpeg_calidad_20": "JPEG", "subexposicion_x0.5": "SUBEXP"}
    for k, n in claves.items():
        v[f"ROB_{n}"] = c(r["robustez"][k])
        v[f"C_{n}"] = f"{c(r['cobertura_degradada'][k] * 100, 1)} %"
    dm = grupos["DIENTE"] & grupos["MONO"]
    v["N_DIENTE_MONO"] = str(int(dm.sum()))
    v["F_DIENTE_MONO"] = c(acc_f[dm].mean()) if dm.any() else "sin casos"
    pb = np.load(A2 / f"{sel}_proba_test_baja_resolucion_x0.25.npy")
    rb = grupos["RES_BAJA"]
    v["F_RES_BAJA_X025"] = c((pb.argmax(1)[rb] == y[rb]).mean()) if rb.any() else "sin casos"
    cm = confusion_matrix(y, pf.argmax(1), labels=range(5))
    v["CONF_ROSA_TULIPAN"] = str(int(cm[2, 4] + cm[4, 2]))
    res = json.loads((A2 / "FIN_final_resumen.json").read_text())
    f1s = np.array([res[s]["test_macro_f1"] for s in res])
    cvf = pd.read_csv(A2 / "CV_FIN_cv.csv").macro_f1
    lo, hi = r["macro_f1_ic95"]
    v.update({"TEST_MACRO_F1": c(r["test"]["macro_f1"]), "TEST_IC95": f"{c(lo)} a {c(hi)}",
              "TEST_MEDIA_SEMILLAS": c(f1s.mean()), "TEST_DE_SEMILLAS": c(f1s.std(ddof=1)),
              "CV_MEDIA": c(cvf.mean()), "CV_DE": c(cvf.std()), "ECE_TEST": c(r["test"]["ece"]),
              "COBERTURA_TEST": f"{c(r['cobertura_test'] * 100, 1)} %",
              "PRECISION_ACEPTADAS": f"{c(r['precision_aceptadas_test'] * 100, 1)} %",
              "P95_MS": c(r["p95_ms"], 1), "MCNEMAR_P": c(r["mcnemar_vs_E03"]["p_valor"]),
              "TEMPERATURA": c(meta["temperatura"]), "UMBRAL": c(meta["umbral"]),
              "SEMILLA_PUBLICADA": sel.split("_s")[-1], "SHA256_PESOS": meta["sha256"],
              "TAMANO_MB": c(Path("modelos/finalista.pt").stat().st_size / 1e6, 1),
              "MLFLOW_MODEL_ID": f"run `{sel}` del experimento flores-proyecto-integrador (mlflow.db)"})
    (A2 / "tablas" / "marcadores_ficha.json").write_text(json.dumps(v, indent=2, ensure_ascii=False))
    for ruta in (Path("docs/ficha_modelo.md"), Path("README.md")):
        s = ruta.read_text()
        s = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda mm: v.get(mm.group(1), mm.group(0)), s)
        ruta.write_text(s)
        faltan = re.findall(r"\{\{[A-Z0-9_]+\}\}", s)
        print(ruta, "marcadores sin llenar:", faltan)


if __name__ == "__main__":
    main()
