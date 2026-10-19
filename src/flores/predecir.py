"""Inferencia del modelo desplegable: imagen(es) -> clase, probabilidad calibrada o abstención.

Carga DINOv2 ViT-S/14 desde timm, reemplaza los bloques finales por la cola ajustada guardada en
`modelos/<id>.pt` y aplica la temperatura y el umbral de abstención fijados en validación
(`modelos/<id>.json`). Los pesos de la cola se distribuyen con el repositorio; el resto del
transformador se descarga de Hugging Face la primera vez.
Uso: python -m flores.predecir modelos/finalista.pt foto1.jpg [foto2.jpg ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .ajuste_fino import CLASES, ColaViT, cargar_vit

NOMBRES = {"daisy": "margarita", "dandelion": "diente de león", "roses": "rosa",
           "sunflowers": "girasol", "tulips": "tulipán"}


def cargar(ruta_pesos: Path) -> tuple[torch.nn.Module, object, dict]:
    meta = json.loads(ruta_pesos.with_suffix(".json").read_text())
    vit, tf = cargar_vit()
    cola = ColaViT(vit, meta["desde"], meta["ajustar"], lora_r=meta.get("lora_r", 0))
    cola.load_state_dict(torch.load(ruta_pesos, map_location="cpu"))
    # ensamblar el ViT completo: bloques congelados + cola ajustada (LoRA fusionada si aplica)
    from . import experimentos_a2 as ex
    ex._VIT = vit
    return ex.modelo_completo(cola), tf, meta


@torch.inference_mode()
def predecir(modelo, tf, meta: dict, imagenes: list[Path]) -> list[dict]:
    salida = []
    for ruta in imagenes:
        with Image.open(ruta) as im:
            z = modelo(tf(im.convert("RGB"))[None])[0] / meta["temperatura"]
        p = z.softmax(0)
        k = int(p.argmax())
        conf = float(p[k])
        salida.append({"imagen": str(ruta), "clase": NOMBRES[CLASES[k]] if conf >= meta["umbral"] else "me abstengo",
                       "confianza": round(conf, 4),
                       "probabilidades": {NOMBRES[c]: round(float(v), 4) for c, v in zip(CLASES, p)}})
    return salida


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos", type=Path)
    ap.add_argument("imagenes", type=Path, nargs="+")
    a = ap.parse_args()
    modelo, tf, meta = cargar(a.pesos)
    for r in predecir(modelo, tf, meta, a.imagenes):
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
