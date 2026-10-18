"""Experimentos de la Actividad 2: ajuste fino de DINOv2 (E04-E08) con registro en MLflow.

Protocolo (plan de la A1, Tablas 3 y 6):
- `estudio`: un valor por combinación de hiperparámetros de ajuste, una semilla, partición
  fija; elige por pérdida de validación y guarda curvas, probabilidades de validación y
  el estado del mejor checkpoint. La prueba no se toca.
- `cv`: validación cruzada agrupada k = 5 sobre entrenamiento + validación, con los mismos
  pliegues que E03 (StratifiedGroupKFold, semilla 2026). Dentro de cada pliegue, un 15 % de
  los grupos de entrenamiento se separa como calibración para fijar temperatura y umbral de
  abstención (selección anidada); el pliegue retenido solo se evalúa.
- `final`: la configuración finalista con las semillas 11, 22 y 33 en la partición fija y una
  única evaluación en prueba, más robustez.
Uso: python -m flores.experimentos_a2 <accion> configs/a2/<archivo>.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metricas
from .ajuste_fino import CLASES, Configuracion, ColaViT, ajustar, cargar_vit, temperatura
from .seleccion import umbral_por_precision

OUT = Path("outputs")
A2 = OUT / "a2"
CACHE = OUT / "cache"
_VIT = None


def vit():
    global _VIT
    if _VIT is None:
        _VIT, _ = cargar_vit()
    return _VIT


def particion():
    p = pd.read_csv(OUT / "particion.csv")
    y = p.label.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    return p, y


def tokens(nombre: str, desde: int = 8) -> torch.Tensor:
    """Tokens en float16 desde la caché (memmap -> tensor)."""
    return torch.from_numpy(np.load(CACHE / f"{nombre}_b{desde}.npy", mmap_mode="r")[:])


def softmax(z, t=1.0):
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def entrenar(cfg: Configuracion, idx_tr, idx_va, y, limpio, vistas, registrar=None):
    base = vit()                          # cargar antes de sembrar: la semilla gobierna solo la cola
    torch.manual_seed(cfg.semilla)
    modelo = ColaViT(base, cfg.desde, cfg.ajustar, lora_r=cfg.lora_r, drop_path=cfg.drop_path)
    # vista 0 = limpia (fase LP y épocas pares); con aumentación se alternan las vistas guardadas
    xtr = [limpio[idx_tr]] + ([v[idx_tr] for v in vistas] if cfg.aumentacion else [])
    yt, yv = torch.from_numpy(y[idx_tr]), torch.from_numpy(y[idx_va])
    res = ajustar(modelo, xtr, yt, limpio[idx_va], yv, cfg, registrar)
    return modelo, res


def logits(modelo, x, lote=128):
    modelo.eval()
    with torch.no_grad():
        return torch.cat([modelo(x[i:i + lote].float()) for i in range(0, len(x), lote)]).numpy()


def cargar_cfg(ruta: Path) -> tuple[str, str, list[Configuracion]]:
    d = yaml.safe_load(ruta.read_text())
    base = Configuracion(**d.get("base", {}))
    rejilla = d.get("rejilla", [{}])
    return d["id"], d["pregunta"], [replace(base, **r) for r in rejilla]


def estudio(ruta: Path) -> None:
    """Rejilla de ajuste en la partición fija; criterio: pérdida de validación del mejor checkpoint."""
    ide, pregunta, cfgs = cargar_cfg(ruta)
    p, y = particion()
    tr, va = np.where(p.split == "train")[0], np.where(p.split == "val")[0]
    limpio = tokens("limpio")
    vistas = [tokens(f"aumentada_v{v}") for v in range(3)] if any(c.aumentacion for c in cfgs) else []
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("flores-proyecto-integrador")
    filas = []
    for k, cfg in enumerate(cfgs):
        nombre = f"{ide}_{k:02d}"
        with mlflow.start_run(run_name=nombre):
            mlflow.set_tags({"id": ide, "estudio": ide, "pregunta": pregunta, "fase": "estudio"})
            mlflow.log_params({**asdict(cfg), "particion": "fija"})
            reg = lambda f: mlflow.log_metrics({k2: v for k2, v in f.items() if k2 != "epoca" and v == v}, step=f["epoca"])  # noqa: E731
            modelo, res = entrenar(cfg, tr, va, y, limpio, vistas, reg)
            zv = logits(modelo, limpio[va])
            f1v = f1_score(y[va], zv.argmax(1), average="macro")
            mlflow.log_metrics({"val_macro_f1": f1v, "mejor_epoca": res.mejor_epoca,
                                "segundos_ft": res.segundos, "entrenables": res.entrenables})
            pd.DataFrame(res.historial).to_csv(A2 / f"{nombre}_curvas.csv", index=False)
            np.save(A2 / f"{nombre}_logits_val.npy", zv)
            torch.save({k2: v for k2, v in modelo.state_dict().items()}, A2 / f"{nombre}_mejor.pt")
            fila = {"run": nombre, **asdict(cfg), "val_loss": min(h["val_loss"] for h in res.historial),
                    "val_macro_f1": f1v, "mejor_epoca": res.mejor_epoca, "segundos_ft": res.segundos,
                    "epocas_hechas": len(res.historial) - 1, "entrenables": res.entrenables}
            filas.append(fila)
            print(json.dumps(fila))
    pd.DataFrame(filas).to_csv(A2 / f"{ide}_estudio.csv", index=False)


def pliegues(p, y):
    dev = np.where(p.split != "test")[0]
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=2026)
    return dev, list(sgkf.split(dev, y[dev], p.grupo.to_numpy()[dev]))


def calibracion_interna(p, y, idx_fit, semilla=2026):
    """Separa ~15 % de los grupos de entrenamiento del pliegue para temperatura y umbral."""
    sg = StratifiedGroupKFold(7, shuffle=True, random_state=semilla)
    a, b = next(sg.split(idx_fit, y[idx_fit], p.grupo.to_numpy()[idx_fit]))
    return idx_fit[a], idx_fit[b]


def metricas_pliegue(y, z_ho, z_lr, z_cal, y_cal):
    t = temperatura(z_cal, y_cal)
    pc, ph, pl = softmax(z_cal, t), softmax(z_ho, t), softmax(z_lr, t)
    u = umbral_por_precision(pc, y_cal, 0.99)
    acc = ph.max(1) >= u
    return {"macro_f1": f1_score(y, ph.argmax(1), average="macro"),
            "macro_f1_baja_res": f1_score(y, pl.argmax(1), average="macro"),
            "ece": metricas.ece(ph, y), "temperatura": t, "umbral": u,
            "cobertura": float(acc.mean()),
            "precision_aceptadas": float((ph.argmax(1)[acc] == y[acc]).mean()) if acc.any() else float("nan"),
            "recall_min": float(min(metricas.calidad(y, ph)[f"recall_{c}"] for c in CLASES))}


def cv(ruta: Path) -> None:
    """Validación cruzada agrupada k = 5 con calibración anidada para una configuración."""
    d = yaml.safe_load(ruta.read_text())
    ide = d["id"]
    p, y = particion()
    dev, folds = pliegues(p, y)
    limpio, baja = tokens("limpio"), tokens("baja_resolucion_x0.25")
    filas = []
    if d.get("tipo") == "sonda_E03":
        # E03 con el mismo protocolo: C elegido en la calibración interna, sin tocar el pliegue
        repr_ = cola_cls(limpio)
        repr_b = cola_cls(baja)
        for f, (a, b) in enumerate(folds):
            fit, cal = calibracion_interna(p, y, dev[a])
            mejor = None
            for c in (0.01, 0.1, 1.0, 10.0):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000)).fit(repr_[fit], y[fit])
                s = f1_score(y[cal], clf.predict(repr_[cal]), average="macro")
                if mejor is None or s > mejor[0]:
                    mejor = (s, c, clf)
            clf = mejor[2]
            lg = lambda X: clf.decision_function(X)  # noqa: E731
            m = metricas_pliegue(y[dev[b]], lg(repr_[dev[b]]), lg(repr_b[dev[b]]), lg(repr_[cal]), y[cal])
            filas.append({"pliegue": f, "C": mejor[1], **m})
            print(json.dumps(filas[-1]))
    else:
        cfg = replace(Configuracion(**d["base"]), presupuesto_s=1e9)
        vistas = [tokens(f"aumentada_v{v}") for v in range(3)] if cfg.aumentacion else []
        for f, (a, b) in enumerate(folds):
            fit, cal = calibracion_interna(p, y, dev[a])
            t0 = time.time()
            # época fija (la elegida en el estudio): sin parada temprana sobre el pliegue retenido
            cfg_f = replace(cfg, epocas=d["epocas"])
            modelo, _ = entrenar(cfg_f, fit, cal, y, limpio, vistas)
            m = metricas_pliegue(y[dev[b]], logits(modelo, limpio[dev[b]]), logits(modelo, baja[dev[b]]),
                                 logits(modelo, limpio[cal]), y[cal])
            filas.append({"pliegue": f, "segundos": time.time() - t0, **m})
            print(json.dumps(filas[-1]))
    df = pd.DataFrame(filas)
    df.to_csv(A2 / f"{ide}_cv.csv", index=False)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("flores-proyecto-integrador")
    with mlflow.start_run(run_name=f"{ide}_cv"):
        mlflow.set_tags({"id": ide, "fase": "cv_agrupada_k5"})
        mlflow.log_params({k: v for k, v in d.items() if isinstance(v, (int, float, str))})
        mlflow.log_metrics({f"cv_{k}_media": float(df[k].mean()) for k in df.columns if k != "pliegue"})
        mlflow.log_metrics({f"cv_{k}_de": float(df[k].std()) for k in df.columns if k != "pliegue"})
        mlflow.log_artifact(str(A2 / f"{ide}_cv.csv"))


def cola_cls(x):
    """Token de clase de DINOv2 congelado a partir de la caché (equivale a la sonda de E03)."""
    cola = ColaViT(vit(), 8, 0).eval()
    with torch.no_grad():
        return torch.cat([cola.representacion(x[i:i + 128].float()) for i in range(0, len(x), 128)]).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["estudio", "cv"])
    ap.add_argument("config", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("HILOS", "2")))
    A2.mkdir(parents=True, exist_ok=True)
    {"estudio": estudio, "cv": cv}[a.accion](a.config)


if __name__ == "__main__":
    main()
