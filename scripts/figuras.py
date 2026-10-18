"""Figuras del plan experimental a partir de outputs/ (sin recalcular modelos)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
FIG = OUT / "figuras"
FIG.mkdir(exist_ok=True)
CLASES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
ES = {"daisy": "margarita", "dandelion": "diente de león", "roses": "rosa",
      "sunflowers": "girasol", "tulips": "tulipán"}
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
AZUL, GRIS, NARANJA, VERDE = "#2b5c8a", "#9aa5b1", "#d9822b", "#3a8d5c"

# ---------- Figura 1: composición y fuga ------------------------------------------
p = pd.read_csv(OUT / "particion.csv")
rep = json.loads((OUT / "particion_reporte.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.1), gridspec_kw={"width_ratios": [1.35, 1]})
tab = pd.crosstab(p.label, p.split)[["train", "val", "test"]].loc[CLASES]
x = np.arange(len(CLASES))
for i, (s, c) in enumerate(zip(["train", "val", "test"], [AZUL, NARANJA, VERDE])):
    ax[0].bar(x + (i - 1) * 0.27, tab[s], 0.27, color=c,
              label={"train": "entrenamiento", "val": "validación", "test": "prueba"}[s])
ax[0].set_xticks(x, [ES[c] for c in CLASES], rotation=20, ha="right")
ax[0].set_ylabel("imágenes")
ax[0].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper left")
ax[0].set_ylim(0, 790)
ax[0].set_title("(a) Composición por clase y partición", fontsize=8, loc="left")
ca, cg = rep["contaminacion_aleatoria"]["test"], rep["contaminacion_por_grupos"]["test"]
etq = ["autor visto\nen entrenamiento", "casi duplicado\nen entrenamiento"]
va = [100 * ca["autor_en_train"] / ca["n"], 100 * ca["casi_duplicado_en_train"] / ca["n"]]
vg = [100 * cg["autor_en_train"] / cg["n"], 100 * cg["casi_duplicado_en_train"] / cg["n"]]
ax[1].bar(np.arange(2) - 0.2, va, 0.4, color=GRIS, label="aleatoria estratificada")
ax[1].bar(np.arange(2) + 0.2, vg, 0.4, color=AZUL, label="estratificada por grupos")
for i, v in enumerate(va):
    ax[1].text(i - 0.2, v + 1.5, f"{v:.1f} %", ha="center", fontsize=7)
for i, v in enumerate(vg):
    ax[1].text(i + 0.2, v + 1.5, f"{v:.1f} %", ha="center", fontsize=7)
ax[1].set_xticks(range(2), etq)
ax[1].set_ylabel("% del conjunto de prueba")
ax[1].set_ylim(0, 90)
ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("(b) Contaminación de prueba según el esquema", fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig1_particion.png", bbox_inches="tight")

# ---------- Figura 2: líneas base -------------------------------------------------
orden = [("E00", "E00 mayoritaria"), ("E01", "E01 HOG+color+RL"),
         ("E02_s11", "E02 CNN s11"), ("E02_s22", "E02 CNN s22"),
         ("E02_s33", "E02 CNN s33"), ("E03", "E03 DINOv2 sonda")]
lb = {k: json.loads((OUT / f"{k}_resultados.json").read_text()) for k, _ in orden}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
f1 = [lb[k]["test"]["macro_f1"] for k, _ in orden]
lo = [lb[k]["test"]["macro_f1_ic95"][0] for k, _ in orden]
hi = [lb[k]["test"]["macro_f1_ic95"][1] for k, _ in orden]
col = [GRIS, NARANJA, AZUL, AZUL, AZUL, VERDE]
y = np.arange(len(orden))[::-1]
ax[0].barh(y, f1, color=col, height=0.6)
ax[0].errorbar(f1, y, xerr=[np.array(f1) - lo, np.array(hi) - f1], fmt="none",
               ecolor="black", capsize=2, lw=0.8)
for yi, v, h in zip(y, f1, hi):
    ax[0].text(h + 0.015, yi, f"{v:.3f}", va="center", fontsize=7)
ax[0].set_yticks(y, [n for _, n in orden])
ax[0].set_xlim(0, 1.12)
ax[0].set_xlabel("macro-F1 en prueba (IC 95 % bootstrap por autor)")
ax[0].set_title("(a) Escalera de líneas base", fontsize=8, loc="left")
yt = p[p.split == "test"].label.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
pred = np.load(OUT / "E03_proba_test.npy").argmax(1)
cm = confusion_matrix(yt, pred, labels=range(5), normalize="true")
im = ax[1].imshow(cm, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax[1].text(j, i, "0" if cm[i, j] == 0 else f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9, fontweight="bold",
                   color="white" if cm[i, j] > 0.6 else "black")
ax[1].set_xticks(range(5), [ES[c] for c in CLASES], rotation=35, ha="right")
ax[1].set_yticks(range(5), [ES[c] for c in CLASES])
ax[1].set_xlabel("predicha")
ax[1].set_ylabel("real")
ax[1].set_title("(b) E03: matriz normalizada por fila", fontsize=8, loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig2_lineas_base.png", bbox_inches="tight")

# ---------- Figura 3: curvas de la CNN --------------------------------------------
fig, ax = plt.subplots(figsize=(3.2, 2.0))
for s, c in zip((11, 22, 33), ["#2b5c8a", "#5d88b3", "#8fb2d4"]):
    d = pd.read_csv(OUT / f"E02_s{s}_curvas.csv")
    ax.plot(d.epoca, d.train_loss, color=c, lw=1, ls="--")
    ax.plot(d.epoca, d.val_loss, color=c, lw=1.2, label=f"semilla {s}")
ax.set_xlabel("época")
ax.set_ylabel("entropía cruzada")
ax.legend(frameon=False, fontsize=6.5, title="continua: val; discontinua: train", title_fontsize=6.5)
fig.tight_layout()
fig.savefig(FIG / "fig3_curvas_cnn.png", bbox_inches="tight")
print("figuras en", FIG)
