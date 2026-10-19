"""Inventario capa por capa de las dos redes de la A2 (DINOv2 ViT-S/14 y EfficientNet-B0).

Para cada bloque registra la forma de su salida con una imagen de 224 x 224, el número de
parámetros y qué parte se entrena en cada configuración (E03 sonda, E04 LP-FT k = 2,
E05 LP-FT k = 4, E06 LoRA r = 8 en 4 bloques, E10 ajuste completo de la CNN).
Uso: python -m flores.arquitectura  ->  outputs/a2/arquitectura_{vit,cnn}.csv
"""
from __future__ import annotations

import pandas as pd
import timm
import torch

from .ajuste_fino import BACKBONE, ColaViT

A2 = "outputs/a2"


def n(m: torch.nn.Module) -> int:
    return int(sum(q.numel() for q in m.parameters()))


def inventario_vit() -> pd.DataFrame:
    vit = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=224).eval()
    x = torch.zeros(1, 3, 224, 224)
    filas = []
    with torch.no_grad():
        t = vit.patch_embed(x)
        filas.append({"capa": "patch_embed (conv 14x14, paso 14)", "salida": str(tuple(t.shape)), "parametros": n(vit.patch_embed)})
        t = vit._pos_embed(t)
        filas.append({"capa": "token CLS + embedding posicional", "salida": str(tuple(t.shape)),
                      "parametros": int(vit.cls_token.numel() + vit.pos_embed.numel())})
        for i, blk in enumerate(vit.blocks):
            t = blk(t)
            filas.append({"capa": f"bloque {i} (LN, MHSA 6 cabezas, LayerScale, LN, MLP 384-1536-384)",
                          "salida": str(tuple(t.shape)), "parametros": n(blk), "bloque": i})
        t = vit.norm(t)
        filas.append({"capa": "LayerNorm final", "salida": str(tuple(t.shape)), "parametros": n(vit.norm)})
    filas.append({"capa": "cabeza lineal sobre CLS (384 -> 5)", "salida": "(1, 5)", "parametros": 384 * 5 + 5})
    df = pd.DataFrame(filas)
    # parámetros entrenables por configuración (contados sobre la cola real)
    for nombre, kw in {"E03": dict(ajustar=0), "E04_k2": dict(ajustar=2), "E05_k4": dict(ajustar=4),
                       "E06_lora_r8": dict(ajustar=4, lora_r=8)}.items():
        cola = ColaViT(vit, 8, **kw)
        df.attrs[nombre] = cola.entrenables() if nombre != "E03" else 384 * 5 + 5
    return df


def inventario_cnn() -> pd.DataFrame:
    m = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=5).eval()
    x = torch.zeros(1, 3, 224, 224)
    filas = []
    with torch.no_grad():
        t = m.bn1(m.conv_stem(x))
        filas.append({"capa": "conv_stem 3x3 paso 2 + BN + SiLU", "salida": str(tuple(t.shape)), "parametros": n(m.conv_stem) + n(m.bn1)})
        for i, etapa in enumerate(m.blocks):
            t = etapa(t)
            tipo = type(etapa[0]).__name__
            filas.append({"capa": f"etapa {i}: {len(etapa)} x {tipo}", "salida": str(tuple(t.shape)), "parametros": n(etapa)})
        t = m.bn2(m.conv_head(t))
        filas.append({"capa": "conv_head 1x1 + BN + SiLU", "salida": str(tuple(t.shape)), "parametros": n(m.conv_head) + n(m.bn2)})
        t = m.global_pool(t)
        filas.append({"capa": "global average pooling", "salida": str(tuple(t.shape)), "parametros": 0})
    filas.append({"capa": "dropout + clasificador lineal (1280 -> 5)", "salida": "(1, 5)", "parametros": n(m.classifier)})
    df = pd.DataFrame(filas)
    df.attrs["E10"] = n(m)
    return df


def main() -> None:
    torch.set_num_threads(1)
    v, c = inventario_vit(), inventario_cnn()
    v.to_csv(f"{A2}/arquitectura_vit.csv", index=False)
    c.to_csv(f"{A2}/arquitectura_cnn.csv", index=False)
    resumen = {**v.attrs, **c.attrs, "vit_total": int(v.parametros.sum()), "cnn_total": int(c.parametros.sum())}
    pd.Series(resumen).to_csv(f"{A2}/arquitectura_entrenables.csv", header=["parametros"])
    print(v.to_string(), c.to_string(), resumen, sep="\n\n")


if __name__ == "__main__":
    main()
