"""Diagramas del pipeline y de las dos arquitecturas (figuras para el informe y la presentación).

Uso: python scripts/a2/diagramas.py  ->  outputs/a2/figuras/diagrama_{pipeline,arquitecturas}.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

FIG = Path("outputs/a2/figuras")
AZUL, GRIS, VERDE, NARANJA = "#cfe1f2", "#e6e6e6", "#cdebc5", "#fde0c5"


def caja(ax, x, y, w, h, texto, color, fs=7.5, negrita=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=color, ec="#555555", lw=0.8))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs, weight="bold" if negrita else "normal", wrap=True)


def flecha(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.9))


def pipeline():
    fig, ax = plt.subplots(figsize=(9.6, 2.9))
    ax.set(xlim=(0, 10), ylim=(0, 3)); ax.axis("off")
    pasos = [("tf_flowers\n3 659 fotos\nCC-BY 2.0", GRIS), ("Auditoría y\ngrupos de fuga\n1 531 grupos", GRIS),
             ("Partición agrupada\npor autor\n2 564 / 548 / 547", AZUL), ("Estudios E04 a E10\nrejilla de tasa\nen validación", VERDE),
             ("CV agrupada k = 5\ncalibración anidada\nfrente a E03", VERDE), ("Prueba única\n3 semillas\nMcNemar, IC", NARANJA),
             ("Modelo publicado\ntemperatura +\nabstención", NARANJA)]
    w, gap = 1.3, 0.11
    for i, (t, c) in enumerate(pasos):
        x = 0.1 + i * (w + gap)
        caja(ax, x, 1.15, w, 1.25, t, c, fs=7)
        if i:
            flecha(ax, x - gap, 1.775, x, 1.775)
    ax.text(5, 0.55, "MLflow registra parámetros, métricas por época, etiquetas y artefactos de cada run · semillas 2026, 11, 22, 33 · "
            "la prueba no interviene en ninguna selección", ha="center", fontsize=7.5, style="italic")
    ax.text(0.1, 2.75, "Datos", fontsize=8, weight="bold"); ax.text(4.4, 2.75, "Selección y comparación", fontsize=8, weight="bold")
    ax.text(8.5, 2.75, "Evaluación y uso", fontsize=8, weight="bold")
    fig.savefig(FIG / "diagrama_pipeline.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def arquitecturas():
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set(xlim=(0, 10), ylim=(0, 4.6)); ax.axis("off")
    # ViT
    ax.text(0.1, 4.35, "a) DINOv2 ViT-S/14 con LP-FT (21,6 M parámetros; 3,55 M entrenables con k = 2)", fontsize=8.5, weight="bold")
    y = 3.2
    vit = [("Imagen\n224 × 224", GRIS, 0.9), ("Parches 14 × 14\nconv → 256 × 384\n+ CLS + posición", AZUL, 1.35),
           ("Bloques 0 a 9\ncongelados\n(caché en el 8)", AZUL, 1.35), ("Bloques 10 y 11\nLN · MHSA 6 cabezas\nLayerScale · MLP 1 536", VERDE, 1.6),
           ("LayerNorm\nfinal", VERDE, 0.85), ("Token CLS\n384", GRIS, 0.7), ("Cabeza lineal\n384 → 5\n(inicio: sonda E03)", NARANJA, 1.25),
           ("Tempe-\nratura +\numbral", NARANJA, 0.85)]
    x = 0.1
    for i, (t, c, w) in enumerate(vit):
        caja(ax, x, y, w, 0.95, t, c, fs=6.8)
        if i:
            flecha(ax, x - 0.12, y + 0.475, x, y + 0.475)
        x += w + 0.12
    # CNN
    ax.text(0.1, 2.3, "b) EfficientNet-B0 ajustada por completo (4,01 M parámetros, todos entrenables)", fontsize=8.5, weight="bold")
    y = 1.15
    cnn = [("Imagen\n224 × 224", GRIS, 0.9), ("Tallo conv 3 × 3\npaso 2 · BN · SiLU\n32 × 112²", VERDE, 1.25),
           ("7 etapas MBConv\nexpansión 1 × 1 · depthwise\nSE · proyección 1 × 1", VERDE, 1.9), ("mapas\n16 × 112²\na 320 × 7²", GRIS, 1.2),
           ("Conv 1 × 1\n1 280 canales", VERDE, 1.05), ("Promedio\nglobal", GRIS, 0.85), ("Dropout +\nlineal 1 280 → 5", NARANJA, 1.2)]
    x = 0.1
    for i, (t, c, w) in enumerate(cnn):
        caja(ax, x, y, w, 0.95, t, c, fs=6.8)
        if i:
            flecha(ax, x - 0.12, y + 0.475, x, y + 0.475)
        x += w + 0.12
    for i, (t, c) in enumerate([("congelado", AZUL), ("entrenado", VERDE), ("cabeza y calibración", NARANJA), ("sin parámetros o entrada", GRIS)]):
        caja(ax, 0.1 + i * 2.2, 0.15, 0.35, 0.3, "", c)
        ax.text(0.55 + i * 2.2, 0.3, t, va="center", fontsize=7.5)
    fig.savefig(FIG / "diagrama_arquitecturas.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    pipeline()
    arquitecturas()
