"""Ajuste fino parcial de DINOv2 (LP-FT y LoRA) con caché de tokens intermedios.

Idea de eficiencia: si los primeros 12 - k bloques del ViT están congelados y no hay
aleatoriedad en ellos, su salida es una función fija de la imagen. Se calcula una sola vez
(`construir_cache`) y el entrenamiento recorre solo los k bloques finales (`ColaViT`).
Para la aumentación se guardan varias vistas aleatorias por imagen de entrenamiento.
Uso de la caché: python -m flores.ajuste_fino cache <raiz_datos> <outputs> --desde 8 --vistas 3
"""
from __future__ import annotations

import argparse
import copy
import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T

CLASES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
BACKBONE = "vit_small_patch14_dinov2.lvd142m"
LADO = 224

# Degradaciones de la prueba de robustez (mismas funciones que src/flores/diagnostico.py)
def _jpeg(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


DEGRADACIONES = {
    "desenfoque_gaussiano_r2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "jpeg_calidad_20": lambda im: _jpeg(im, 20),
    "subexposicion_x0.5": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
    "baja_resolucion_x0.25": lambda im: im.resize((max(1, im.width // 4), max(1, im.height // 4)),
                                                  Image.BILINEAR).resize(im.size, Image.BILINEAR),
}


def cargar_vit() -> tuple[nn.Module, T.Compose]:
    """DINOv2 ViT-S/14 congelado y la transformación de evaluación documentada por timm."""
    vit = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=LADO).eval()
    cfg = timm.data.resolve_data_config({}, model=vit)
    cfg.update(input_size=(3, LADO, LADO), crop_pct=LADO / 256)
    return vit, timm.data.create_transform(**cfg)


def transformacion_aumentada(eval_tf: T.Compose) -> T.Compose:
    """Aumentación de entrenamiento con familias distintas de las degradaciones de prueba.

    Recorte aleatorio, volteo, rotación leve y cambios de saturación y matiz. No usa brillo,
    desenfoque, compresión ni cambios de resolución, que se reservan para evaluar robustez
    (Hendrycks y Dietterich, 2019).
    """
    norm = [t for t in eval_tf.transforms if isinstance(t, T.Normalize)][0]
    return T.Compose([
        T.RandomResizedCrop(LADO, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(saturation=0.3, hue=0.05),
        T.ToTensor(), norm,
    ])


@torch.inference_mode()
def tokens_hasta(vit: nn.Module, x: torch.Tensor, desde: int) -> torch.Tensor:
    """Secuencia de tokens a la entrada del bloque `desde` (bloques 0..desde-1 aplicados)."""
    x = vit.patch_embed(x)
    x = vit._pos_embed(x)
    x = vit.patch_drop(x)
    x = vit.norm_pre(x)
    for blk in vit.blocks[:desde]:
        x = blk(x)
    return x


def construir_cache(raiz: Path, out: Path, desde: int, vistas: int, lote: int = 32, semilla: int = 2026) -> None:
    """Escribe en outputs/cache/ los tokens del bloque `desde` (float16, memmap)."""
    torch.manual_seed(semilla)
    p = pd.read_csv(out / "particion.csv")
    vit, tf = cargar_vit()
    aug = transformacion_aumentada(tf)
    dst = out / "cache"
    dst.mkdir(exist_ok=True)
    n, ntok, dim = len(p), 1 + (LADO // 14) ** 2, vit.embed_dim

    def volcar(nombre: str, idx: np.ndarray, fn) -> None:
        ruta = dst / f"{nombre}_b{desde}.f16"
        if ruta.exists():
            print("ya existe", ruta.name)
            return
        mm = np.lib.format.open_memmap(ruta.with_suffix(".npy"), "w+", np.float16, (len(idx), ntok, dim))
        t0 = time.time()
        for i in range(0, len(idx), lote):
            ims = []
            for j in idx[i:i + lote]:
                with Image.open(raiz / p.file[j]) as im:
                    ims.append(fn(im.convert("RGB")))
            mm[i:i + len(ims)] = tokens_hasta(vit, torch.stack(ims), desde).numpy().astype(np.float16)
        mm.flush()
        ruta.touch()
        print(f"{nombre}: {len(idx)} imágenes en {time.time() - t0:.0f} s")

    todo = np.arange(n)
    volcar("limpio", todo, tf)
    volcar("baja_resolucion_x0.25", todo, lambda im: tf(DEGRADACIONES["baja_resolucion_x0.25"](im)))
    prueba = np.where(p.split == "test")[0]
    for nom in ("desenfoque_gaussiano_r2", "jpeg_calidad_20", "subexposicion_x0.5"):
        volcar(f"{nom}_test", prueba, lambda im, f=DEGRADACIONES[nom]: tf(f(im)))
    for v in range(vistas):
        volcar(f"aumentada_v{v}", todo, aug)


# --------------------------------------------------------------------------------------
class LoRALineal(nn.Module):
    """Capa lineal congelada más una actualización de rango bajo B·A (Hu et al., 2022)."""

    def __init__(self, base: nn.Linear, r: int, alfa: float):
        super().__init__()
        self.base = base
        for q in self.base.parameters():
            q.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.escala = alfa / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.escala

    @torch.no_grad()
    def fusionar(self) -> nn.Linear:
        """Devuelve una lineal equivalente sin rama LoRA (sin latencia adicional)."""
        lin = copy.deepcopy(self.base)
        lin.weight += self.escala * self.B @ self.A
        return lin


class ColaViT(nn.Module):
    """Bloques finales de DINOv2, norma final y cabeza lineal sobre el token de clase."""

    def __init__(self, vit: nn.Module, desde: int, ajustar: int, n_clases: int = 5,
                 lora_r: int = 0, lora_alfa: float = 16.0):
        super().__init__()
        self.bloques = copy.deepcopy(vit.blocks[desde:])
        self.norm = copy.deepcopy(vit.norm)
        self.cabeza = nn.Linear(vit.embed_dim, n_clases)
        n = len(self.bloques)
        congelados = n - ajustar               # bloques de la caché que siguen fijos
        for i, blk in enumerate(self.bloques):
            for q in blk.parameters():
                q.requires_grad_(i >= congelados and lora_r == 0)
            if lora_r and i >= congelados:
                blk.attn.qkv = LoRALineal(blk.attn.qkv, lora_r, lora_alfa)
                blk.attn.proj = LoRALineal(blk.attn.proj, lora_r, lora_alfa)
        ajusta_norm = ajustar > 0 and lora_r == 0
        for q in self.norm.parameters():
            q.requires_grad_(ajusta_norm)

    def representacion(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for blk in self.bloques:
            x = blk(x)
        return self.norm(x)[:, 0]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.representacion(tokens))

    def entrenables(self) -> int:
        return int(sum(q.numel() for q in self.parameters() if q.requires_grad))


# --------------------------------------------------------------------------------------
@dataclass
class Configuracion:
    """Hiperparámetros científicos, de ajuste y fijos de un run (Google Developers, s. f.)."""
    ajustar: int = 2                 # bloques finales que se ajustan (científico)
    lora_r: int = 0                  # 0 = LP-FT; >0 = LoRA (científico)
    aumentacion: bool = False        # científico en E07
    optimizador: str = "adamw"       # adamw | sgd
    lr: float = 3e-5                 # tasa de los bloques (ajuste)
    lr_cabeza: float = 1e-3          # tasa de la cabeza en la fase LP
    decaimiento: float = 0.05        # weight decay (ajuste)
    epocas_lp: int = 30              # fase de sonda lineal (LP)
    epocas: int = 12                 # fase de ajuste fino (FT), tope
    presupuesto_s: float = 900       # presupuesto de tiempo de la fase FT
    lote: int = 32
    semilla: int = 2026
    desde: int = 8                   # bloque desde el que arranca la cola (caché)


@dataclass
class Resultado:
    historial: list = field(default_factory=list)
    mejor_epoca: int = 0
    segundos: float = 0.0
    entrenables: int = 0


def _iterar(n: int, lote: int, g: torch.Generator):
    perm = torch.randperm(n, generator=g)
    for i in range(0, n, lote):
        yield perm[i:i + lote]


def _evaluar(modelo: ColaViT, x: torch.Tensor, y: torch.Tensor, lote: int = 128) -> tuple[float, float, np.ndarray]:
    modelo.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(x), lote):
            logits.append(modelo(x[i:i + lote].float()))
    z = torch.cat(logits)
    return F.cross_entropy(z, y).item(), (z.argmax(1) == y).float().mean().item(), z.numpy()


def sonda_lineal(modelo: ColaViT, xtr, ytr, cfg: Configuracion, g: torch.Generator) -> None:
    """Fase LP de LP-FT (Kumar et al., 2022): se entrena solo la cabeza con la cola congelada."""
    with torch.no_grad():
        modelo.eval()
        feats = torch.cat([modelo.representacion(xtr[i:i + 128].float()) for i in range(0, len(xtr), 128)])
    opt = torch.optim.AdamW(modelo.cabeza.parameters(), lr=cfg.lr_cabeza, weight_decay=1e-4)
    for _ in range(cfg.epocas_lp):
        for b in _iterar(len(feats), 64, g):
            loss = F.cross_entropy(modelo.cabeza(feats[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()


def ajustar(modelo: ColaViT, vistas_tr: list[torch.Tensor], ytr, xva, yva, cfg: Configuracion,
            registrar=None) -> Resultado:
    """Fase FT con presupuesto de tiempo; guarda el mejor estado según la pérdida de validación."""
    g = torch.Generator().manual_seed(cfg.semilla)
    sonda_lineal(modelo, vistas_tr[0], ytr, cfg, g)
    params = [q for q in modelo.parameters() if q.requires_grad]
    if cfg.optimizador == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.decaimiento)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.decaimiento)
    pasos = cfg.epocas * math.ceil(len(ytr) / cfg.lote)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr] * len(opt.param_groups),
                                                total_steps=pasos, pct_start=0.1)
    res = Resultado(entrenables=modelo.entrenables())
    vl, va, _ = _evaluar(modelo, xva, yva)
    res.historial.append({"epoca": 0, "train_loss": np.nan, "train_acc": np.nan, "val_loss": vl, "val_acc": va, "seg": 0.0})
    mejor = (vl, copy.deepcopy(modelo.state_dict()), 0)
    t0 = time.time()
    for ep in range(1, cfg.epocas + 1):
        modelo.train()
        xtr = vistas_tr[(ep - 1) % len(vistas_tr)] if cfg.aumentacion else vistas_tr[0]
        tot, ok = 0.0, 0
        te = time.time()
        for b in _iterar(len(ytr), cfg.lote, g):
            z = modelo(xtr[b].float())
            loss = F.cross_entropy(z, ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            ok += (z.argmax(1) == ytr[b]).sum().item()
        vl, va, _ = _evaluar(modelo, xva, yva)
        fila = {"epoca": ep, "train_loss": tot / len(ytr), "train_acc": ok / len(ytr),
                "val_loss": vl, "val_acc": va, "seg": time.time() - te}
        res.historial.append(fila)
        if registrar:
            registrar(fila)
        if vl < mejor[0]:
            mejor = (vl, copy.deepcopy(modelo.state_dict()), ep)
        if time.time() - t0 > cfg.presupuesto_s:
            break
    modelo.load_state_dict(mejor[1])
    res.mejor_epoca, res.segundos = mejor[2], time.time() - t0
    return res


def temperatura(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Escalado de temperatura por mínima NLL en validación (Guo et al., 2017)."""
    z, y = torch.tensor(logits_val), torch.tensor(y_val)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def cierre():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(cierre)
    return float(log_t.exp())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["cache"])
    ap.add_argument("raiz", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--desde", type=int, default=8)
    ap.add_argument("--vistas", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    construir_cache(a.raiz, a.out, a.desde, a.vistas)


if __name__ == "__main__":
    main()
