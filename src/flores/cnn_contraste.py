"""E10: CNN preentrenada de contraste (EfficientNet-B0, ImageNet) con el mismo protocolo.

Responde a la pregunta: bajo la misma partición y el mismo presupuesto de tiempo, ¿una CNN
convolucional preentrenada y ajustada por completo alcanza a DINOv2? Las imágenes se
precargan a 256 px (lado corto, recorte central cuadrado) para que el costo medido sea el
del modelo y no el de decodificar JPEG.
Uso: python -m flores.cnn_contraste configs/a2/e10_cnn_effnet.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import transforms as T

from .ajuste_fino import CLASES, DEGRADACIONES

OUT = Path("outputs")
A2 = OUT / "a2"
RAIZ = Path("data/flower_photos")
MEDIA, DE = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def precargar(archivos, fn=None) -> list[Image.Image]:
    ims = []
    for f in archivos:
        with Image.open(RAIZ / f) as im:
            im = im.convert("RGB")
            if fn:
                im = fn(im)
            ims.append(T.CenterCrop(256)(T.Resize(256, interpolation=T.InterpolationMode.BICUBIC)(im)))
    return ims


def tensor_eval(ims):
    tf = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(MEDIA, DE)])
    return torch.stack([tf(im) for im in ims])


def evaluar(modelo, x, y, lote=64):
    modelo.eval()
    with torch.no_grad():
        z = torch.cat([modelo(x[i:i + lote]) for i in range(0, len(x), lote)])
    return F.cross_entropy(z, torch.from_numpy(y)).item(), (z.argmax(1).numpy() == y).mean(), z.numpy()


def epocas_por_presupuesto(modelo, ims_tr, ytr, aug, xva, cfg, lotes_piloto: int = 4) -> int:
    """Mide el costo de unos lotes de entrenamiento completo y de una evaluación de validación
    sobre una copia del modelo, y devuelve cuántas épocas caben en el presupuesto."""
    m = copy.deepcopy(modelo)
    for q in m.parameters():
        q.requires_grad_(True)
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    m.train()
    t = time.time()
    for i in range(lotes_piloto):
        b = list(range(i * cfg["lote"], (i + 1) * cfg["lote"]))
        loss = F.cross_entropy(m(torch.stack([aug(ims_tr[j]) for j in b])), torch.from_numpy(ytr[b]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    por_lote = (time.time() - t) / lotes_piloto
    t = time.time()
    evaluar(m, xva[:128], np.zeros(min(128, len(xva)), dtype=np.int64))
    eval_s = (time.time() - t) * len(xva) / 128
    epoca_s = por_lote * math.ceil(len(ytr) / cfg["lote"]) + eval_s
    return max(1, int(cfg["presupuesto_s"] // epoca_s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    a = ap.parse_args()
    d = yaml.safe_load(a.config.read_text())
    torch.set_num_threads(2)
    A2.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(OUT / "particion.csv")
    y = p.label.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    tr, va = np.where(p.split == "train")[0], np.where(p.split == "val")[0]
    ims_tr = precargar(p.file[tr])
    xva = tensor_eval(precargar(p.file[va]))
    aug = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                     T.RandomRotation(15), T.ColorJitter(saturation=0.3, hue=0.05),
                     T.ToTensor(), T.Normalize(MEDIA, DE)])
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("flores-proyecto-integrador")
    filas = []
    for k, r in enumerate(d["rejilla"]):
        cfg = {**d["base"], **r}
        torch.manual_seed(cfg["semilla"])
        g = torch.Generator().manual_seed(cfg["semilla"])
        modelo = timm.create_model(d["backbone"], pretrained=True, num_classes=len(CLASES))
        nombre = f"{d['id']}_{k:02d}"
        hist = []
        with mlflow.start_run(run_name=nombre):
            mlflow.set_tags({"id": d["id"], "estudio": d["id"], "pregunta": d["pregunta"], "fase": "estudio"})
            mlflow.log_params({**cfg, "backbone": d["backbone"]})
            # LP: solo el clasificador, unas épocas cortas
            cab = modelo.get_classifier()
            for q in modelo.parameters():
                q.requires_grad_(False)
            for q in cab.parameters():
                q.requires_grad_(True)
            opt = torch.optim.AdamW(cab.parameters(), lr=1e-3, weight_decay=1e-4)
            fases = [("lp", cfg["epocas_lp"], opt, None)]
            # FT: toda la red. El número de épocas se fija ANTES de entrenar convirtiendo el
            # presupuesto de tiempo en pasos (piloto de costo por lote), para que OneCycle complete
            # su calendario y no quede truncado por el reloj (Google Developers, s. f., number of steps).
            todos = list(modelo.parameters())
            if cfg["epocas"] == "auto":
                cfg["epocas"] = epocas_por_presupuesto(modelo, ims_tr, y[tr], aug, xva, cfg)
            mlflow.log_param("epocas_ft", cfg["epocas"])
            opt_ft = torch.optim.AdamW(todos, lr=cfg["lr"], weight_decay=cfg["decaimiento"])
            pasos = cfg["epocas"] * math.ceil(len(tr) / cfg["lote"])
            sch = torch.optim.lr_scheduler.OneCycleLR(opt_ft, max_lr=cfg["lr"], total_steps=pasos, pct_start=0.1)
            fases.append(("ft", cfg["epocas"], opt_ft, sch))
            mejor = (np.inf, None, 0)
            t_ft, ep_glob = None, 0
            for fase, epocas, opt_, sch_ in fases:
                if fase == "ft":
                    for q in todos:
                        q.requires_grad_(True)
                    t_ft = time.time()
                for _ in range(epocas):
                    ep_glob += 1
                    modelo.train()
                    te, tot, ok, normas = time.time(), 0.0, 0, []
                    perm = torch.randperm(len(tr), generator=g)
                    for i in range(0, len(tr), cfg["lote"]):
                        b = perm[i:i + cfg["lote"]].tolist()
                        xb = torch.stack([aug(ims_tr[j]) for j in b])
                        yb = torch.from_numpy(y[tr][b])
                        z = modelo(xb)
                        loss = F.cross_entropy(z, yb)
                        opt_.zero_grad()
                        loss.backward()
                        normas.append(float(torch.norm(torch.stack([q.grad.norm() for q in modelo.parameters() if q.grad is not None]))))
                        opt_.step()
                        if sch_:
                            sch_.step()
                        tot += loss.item() * len(b)
                        ok += (z.argmax(1) == yb).sum().item()
                    vl, vacc, _ = evaluar(modelo, xva, y[va])
                    fila = {"epoca": ep_glob, "fase": fase, "train_loss": tot / len(tr), "train_acc": ok / len(tr),
                            "val_loss": vl, "val_acc": float(vacc), "seg": time.time() - te,
                            "grad_norm_media": float(np.mean(normas)), "grad_norm_max": float(np.max(normas)),
                            "lr_fin_epoca": float(opt_.param_groups[0]["lr"])}
                    hist.append(fila)
                    mlflow.log_metrics({k2: v for k2, v in fila.items() if k2 not in ("epoca", "fase")}, step=ep_glob)
                    print(json.dumps(fila))
                    if fase == "ft" and vl < mejor[0]:
                        mejor = (vl, copy.deepcopy(modelo.state_dict()), ep_glob)
                    if fase == "ft" and time.time() - t_ft > cfg["presupuesto_s"]:
                        break
            modelo.load_state_dict(mejor[1])
            _, _, zv = evaluar(modelo, xva, y[va])
            f1v = f1_score(y[va], zv.argmax(1), average="macro")
            mlflow.log_metrics({"val_macro_f1": f1v, "mejor_epoca": mejor[2], "segundos_ft": time.time() - t_ft})
            pd.DataFrame(hist).to_csv(A2 / f"{nombre}_curvas.csv", index=False)
            np.save(A2 / f"{nombre}_logits_val.npy", zv)
            torch.save(mejor[1], A2 / f"{nombre}_mejor.pt")
            fila = {"run": nombre, **cfg, "val_loss": mejor[0], "val_macro_f1": f1v, "mejor_epoca": mejor[2],
                    "segundos_ft": time.time() - t_ft, "entrenables": int(sum(q.numel() for q in modelo.parameters()))}
            filas.append(fila)
            print(json.dumps(fila))
    pd.DataFrame(filas).to_csv(A2 / f"{d['id']}_estudio.csv", index=False)


if __name__ == "__main__":
    main()
