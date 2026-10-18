"""Experimentos de línea base con un contrato común (BaseExperimento) y registro en MLflow.

Cada experimento recibe la misma partición materializada (outputs/particion.csv), ajusta
lo que necesite solo con entrenamiento, selecciona hiperparámetros solo con validación
y evalúa una única vez en prueba.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from . import metricas, modelos

CLASES = metricas.CLASES


def fijar_semilla(seed: int, determinista: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(determinista)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Datos:
    """Imágenes, etiquetas, grupos y partición compartidos por todos los experimentos."""
    raiz: Path
    particion: Path
    df: pd.DataFrame = field(init=False)
    imgs: list = field(init=False)

    def __post_init__(self):
        self.df = pd.read_csv(self.particion)
        self.df["y"] = self.df.label.map({c: i for i, c in enumerate(CLASES)})
        self.imgs = []
        for f in self.df.file:
            with Image.open(self.raiz / f) as im:
                self.imgs.append(im.convert("RGB"))
        self.idx = {s: np.where(self.df.split == s)[0] for s in ("train", "val", "test")}
        self.y = self.df.y.to_numpy()
        self.grupos = self.df.grupo.to_numpy()


class BaseExperimento(ABC):
    """Contrato: ajustar() devuelve probabilidades de validación y prueba."""

    def __init__(self, cfg: dict, datos: Datos, salida: Path):
        self.cfg, self.d, self.salida = cfg, datos, salida
        self.extra: dict = {}

    @abstractmethod
    def ajustar(self) -> tuple[np.ndarray, np.ndarray]:
        """Entrena con train, selecciona con val y devuelve (proba_val, proba_test)."""

    def ejecutar(self) -> dict:
        d = self.d
        mlflow.set_tracking_uri(self.cfg.get("mlflow_uri", "sqlite:///mlflow.db"))
        mlflow.set_experiment(self.cfg["experimento_mlflow"])
        with mlflow.start_run(run_name=self.cfg["run"]):
            mlflow.set_tags({"id": self.cfg["id"], "familia": self.cfg["familia"],
                             "pregunta": self.cfg["pregunta"]})
            mlflow.log_params({k: v for k, v in self.cfg.items()
                               if isinstance(v, (int, float, str)) and k != "pregunta"})
            mlflow.log_params({"sha256_particion": sha256(d.particion)[:16],
                               "python": platform.python_version(),
                               "torch": torch.__version__, "cpu_threads": os.cpu_count()})
            t0 = time.time()
            pv, pt = self.ajustar()
            yv, yt = d.y[d.idx["val"]], d.y[d.idx["test"]]
            res = {"val": metricas.calidad(yv, pv), "test": metricas.calidad(yt, pt)}
            pred = pt.argmax(1)
            g = d.grupos[d.idx["test"]]
            res["test"]["macro_f1_ic95"] = metricas.ic95_por_grupos(
                yt, pred, g, lambda a, b: f1_score(a, b, average="macro"))
            res["test"]["accuracy_ic95"] = metricas.ic95_por_grupos(yt, pred, g, accuracy_score)
            res["segundos"] = time.time() - t0
            res["extra"] = self.extra
            for part in ("val", "test"):
                mlflow.log_metrics({f"{part}_{k}": v for k, v in res[part].items()
                                    if isinstance(v, float)})
            mlflow.log_metric("segundos_totales", res["segundos"])
            for k, v in self.extra.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            base = self.salida / self.cfg["id"]
            np.save(f"{base}_proba_test.npy", pt)
            Path(f"{base}_resultados.json").write_text(json.dumps(res, indent=2))
            mlflow.log_artifact(f"{base}_resultados.json")
        return res


class Mayoritaria(BaseExperimento):
    """E00: predice siempre la clase más frecuente en entrenamiento (piso heurístico)."""

    def ajustar(self):
        prior = np.bincount(self.d.y[self.d.idx["train"]], minlength=len(CLASES))
        p = np.full(len(CLASES), 2e-4)
        p[prior.argmax()] = 1 - 2e-4 * (len(CLASES) - 1)
        return (np.tile(p, (len(self.d.idx["val"]), 1)),
                np.tile(p, (len(self.d.idx["test"]), 1)))


class ClasicoLR(BaseExperimento):
    """E01: HOG + histograma HSV + regresión logística (¿hace falta deep learning?)."""

    def ajustar(self):
        d = self.d
        x = np.stack([modelos.descriptores_clasicos(im) for im in d.imgs])
        clf, c = modelos.regresion_logistica(x[d.idx["train"]], d.y[d.idx["train"]],
                                             x[d.idx["val"]], d.y[d.idx["val"]],
                                             self.cfg["rejilla_c"])
        im0 = d.imgs[d.idx["test"][0]]
        lat = metricas.latencia_ms(
            lambda im: clf.predict_proba(modelos.descriptores_clasicos(im)[None]), im0)
        self.extra = {"C_elegido": c, "dim": int(x.shape[1]), **lat}
        return clf.predict_proba(x[d.idx["val"]]), clf.predict_proba(x[d.idx["test"]])


class Entrenador:
    """Estado del entrenamiento de una red (modelo, optimizador, historial, mejor checkpoint)."""

    def __init__(self, modelo, optimizador, g: torch.Generator, cfg: dict):
        self.modelo, self.opt, self.g, self.cfg = modelo, optimizador, g, cfg
        self.historial: list[dict] = []
        self.mejor = (np.inf, None, 0)

    def epoca(self, x, y):
        self.modelo.train()
        perm = torch.randperm(len(x), generator=self.g)
        total = 0.0
        for i in range(0, len(x), self.cfg["lote"]):
            b = perm[i:i + self.cfg["lote"]]
            xb = modelos.aumentar(x[b], self.g, self.cfg["rotacion"], self.cfg["zoom"])
            loss = F.cross_entropy(self.modelo(xb), y[b])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item() * len(b)
        return total / len(x)

    @torch.no_grad()
    def evaluar(self, x, y):
        self.modelo.eval()
        logits = self.modelo(x)
        return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()

    def ajustar(self, xtr, ytr, xva, yva):
        for ep in range(1, self.cfg["epocas"] + 1):
            t = time.time()
            tl = self.epoca(xtr, ytr)
            vl, va = self.evaluar(xva, yva)
            self.historial.append({"epoca": ep, "train_loss": tl, "val_loss": vl,
                                   "val_acc": va, "seg": time.time() - t})
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl, "val_acc": va}, step=ep)
            if vl < self.mejor[0]:   # mejor checkpoint según pérdida de validación
                self.mejor = (vl, {k: v.clone() for k, v in self.modelo.state_dict().items()}, ep)
        self.modelo.load_state_dict(self.mejor[1])
        return self.modelo.eval()


class CNNDesdeCero(BaseExperimento):
    """E02: CNN pequeña sin preentrenamiento (¿la complejidad adicional aporta?)."""

    def tensores(self, idx):
        lado = self.cfg["lado"]
        a = np.stack([np.asarray(self.d.imgs[i].resize((lado, lado), Image.BILINEAR))
                      for i in idx])
        return torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255.0

    def ajustar(self):
        d, cfg = self.d, self.cfg
        fijar_semilla(cfg["semilla"])
        g = torch.Generator().manual_seed(cfg["semilla"])
        xtr, xva, xte = (self.tensores(d.idx[s]) for s in ("train", "val", "test"))
        ytr, yva = (torch.from_numpy(d.y[d.idx[s]]) for s in ("train", "val"))
        modelo = modelos.SmallCNN(len(CLASES), cfg["lado"])
        ent = Entrenador(modelo, torch.optim.Adam(modelo.parameters(), lr=cfg["lr"]), g, cfg)
        ent.ajustar(xtr, ytr, xva, yva)
        pd.DataFrame(ent.historial).to_csv(self.salida / f"{cfg['id']}_curvas.csv", index=False)
        torch.save(ent.mejor[1], self.salida / f"{cfg['id']}_mejor.pt")
        with torch.no_grad():
            pv, pt = (F.softmax(modelo(x), 1).numpy() for x in (xva, xte))
        self.extra = {"mejor_epoca": ent.mejor[2],
                      "parametros": int(sum(p.numel() for p in modelo.parameters())),
                      **metricas.latencia_ms(lambda x: modelo(x), xte[:1])}
        return pv, pt


class SondaLinealDINOv2(BaseExperimento):
    """E03: DINOv2 congelado + regresión logística (¿qué aporta la representación fundacional?)."""

    def ajustar(self):
        d, cfg = self.d, self.cfg
        torch.set_num_threads(os.cpu_count() or 2)
        vit, tf, pipe = modelos.dinov2(cfg["backbone"], cfg["lado"])
        cache = self.salida / f"{cfg['id']}_embeddings.npy"
        x = modelos.extraer(vit, d.imgs, tf)
        np.save(cache, x)
        clf, c = modelos.regresion_logistica(x[d.idx["train"]], d.y[d.idx["train"]],
                                             x[d.idx["val"]], d.y[d.idx["val"]],
                                             cfg["rejilla_c"])
        # Control de fuga (scikit-learn, Data leakage): etiquetas permutadas en train
        rng = np.random.default_rng(cfg["semilla"])
        ctrl, _ = modelos.regresion_logistica(
            x[d.idx["train"]], rng.permutation(d.y[d.idx["train"]]),
            x[d.idx["val"]], d.y[d.idx["val"]], [1.0])
        f1_perm = f1_score(d.y[d.idx["val"]], ctrl.predict(x[d.idx["val"]]), average="macro")
        im0 = d.imgs[d.idx["test"][0]]
        lat = metricas.latencia_ms(lambda im: clf.predict_proba(vit(tf(im)[None]).numpy()), im0)
        self.extra = {"C_elegido": c, "dim": int(x.shape[1]),
                      "parametros_backbone": int(sum(p.numel() for p in vit.parameters())),
                      "val_macro_f1_etiquetas_permutadas": float(f1_perm), **lat}
        mlflow.log_dict({k: (list(v) if isinstance(v, tuple) else v) for k, v in pipe.items()},
                        "preprocesamiento_dinov2.json")
        return clf.predict_proba(x[d.idx["val"]]), clf.predict_proba(x[d.idx["test"]])


REGISTRO = {"mayoritaria": Mayoritaria, "clasico_lr": ClasicoLR,
            "cnn_desde_cero": CNNDesdeCero, "sonda_dinov2": SondaLinealDINOv2}
