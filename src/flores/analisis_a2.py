"""Análisis de la Actividad 2: curvas, gráficos de eje, comparación pareada y análisis de error.

Lee solo artefactos ya guardados en outputs/a2 (no entrena ni abre la prueba por su cuenta).
Uso: python -m flores.analisis_a2  ->  outputs/a2/figuras/*.png y outputs/a2/tablas/*.csv|json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from .ajuste_fino import CLASES  # noqa: E402

OUT = Path("outputs")
A2 = OUT / "a2"
FIG = A2 / "figuras"
TAB = A2 / "tablas"
NOMBRES = {"daisy": "margarita", "dandelion": "diente de león", "roses": "rosa",
           "sunflowers": "girasol", "tulips": "tulipán"}
plt.rcParams.update({"font.size": 9, "axes.formatter.use_locale": False, "figure.dpi": 150})


def coma(x, d=3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def cientifica(v: float) -> str:
    """3e-4 -> '3·10⁻⁴' (notación para figuras en español)."""
    m, e = f"{v:.0e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m}·10{str(int(e)).translate(sup)}"


def _fmt_coma(ax) -> None:
    for eje in (ax.xaxis, ax.yaxis):
        if eje.get_scale() == "linear":
            eje.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


# --------------------------------------------------------------------------------------
def estudio(ide: str) -> pd.DataFrame:
    return pd.read_csv(A2 / f"{ide}_estudio.csv")


def curvas_estudio(ide: str, etiqueta: str = "lr") -> Path:
    """Panel 2 x 2: pérdida y exactitud, entrenamiento (discontinua) y validación (continua)."""
    est = estudio(ide)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colores = plt.cm.viridis(np.linspace(0, 0.9, len(est)))
    for (_, r), c in zip(est.iterrows(), colores):
        h = pd.read_csv(A2 / f"{r.run}_curvas.csv")
        h = h[h.fase == "ft"] if "fase" in h else h[h.epoca > 0]
        etq = f"lr = {cientifica(r[etiqueta])}" if etiqueta == "lr" else f"{etiqueta} = {r[etiqueta]}"
        ax[0].plot(h.epoca, h.train_loss, "--", color=c, lw=1)
        ax[0].plot(h.epoca, h.val_loss, "-o", color=c, ms=3, lw=1.3, label=etq)
        ax[1].plot(h.epoca, h.train_acc, "--", color=c, lw=1)
        ax[1].plot(h.epoca, h.val_acc, "-o", color=c, ms=3, lw=1.3)
        ax[0].plot(r.mejor_epoca, r.val_loss, "*", color=c, ms=10, mec="k", mew=0.5)
    if not ide.startswith("E10"):   # la cola del ViT arranca exactamente en la sonda E03 (época 0)
        ax[0].axhline(0.1496, color="grey", lw=0.8, ls=":", label="E03 (punto de partida)")
    ax[0].set(xlabel="época", ylabel="pérdida (entropía cruzada)", yscale="log", title="Pérdida")
    bajo = min(min(l.get_ydata().min() for l in ax[1].get_lines()), 0.9)
    ax[1].set(xlabel="época", ylabel="exactitud", title="Exactitud", ylim=(max(0.5, bajo - 0.02), 1.002))
    for a in ax:
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].legend(fontsize=7, frameon=False)
    for a in ax:
        _fmt_coma(a)
        a.grid(alpha=0.25)
    fig.suptitle(f"{ide}: entrenamiento en línea discontinua, validación en línea continua; estrella = mejor checkpoint", fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    ruta = FIG / f"curvas_{ide}.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def grafico_eje(ids: dict[str, str]) -> Path:
    """Pérdida de validación del mejor checkpoint frente a la tasa de aprendizaje (gráfico de eje)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for ide, nombre in ids.items():
        if not (A2 / f"{ide}_estudio.csv").exists():
            continue
        e = estudio(ide)
        if ide == "E10" and (A2 / "E10b_estudio.csv").exists():   # punto de ampliación del borde
            e = pd.concat([e, estudio("E10b")])
        e = e.sort_values("lr")
        ax.plot(e.lr, e.val_loss, "-o", ms=4, label=nombre)
    ax.axhline(0.1496, color="grey", ls=":", lw=1, label="E03 sonda congelada")
    ax.set(xscale="log", xlabel="tasa de aprendizaje máxima (OneCycle)", ylabel="pérdida de validación (mejor checkpoint)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    ruta = FIG / "eje_lr.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------------------
def comparacion_pareada(a: str, b: str, metrica: str = "macro_f1") -> dict:
    """Diferencia pareada por pliegue (b - a) en los mismos 5 pliegues agrupados.

    Se informan la media, la desviación, el IC 95 % t de Student, la t pareada, la prueba de
    rangos con signo de Wilcoxon (exacta, unilateral b > a) y el tamaño del efecto d_z de Cohen.
    Con k = 5 la potencia es baja y la prueba de Wilcoxon no puede bajar de p = 1/32.
    """
    da, db = pd.read_csv(A2 / f"{a}_cv.csv"), pd.read_csv(A2 / f"{b}_cv.csv")
    d = (db[metrica] - da[metrica]).to_numpy()
    k = len(d)
    se = d.std(ddof=1) / np.sqrt(k)
    alt = "less" if metrica == "ece" else "greater"     # en la ECE, menor es mejor
    t = stats.ttest_rel(db[metrica], da[metrica], alternative=alt)
    try:
        w = stats.wilcoxon(d, alternative=alt, method="exact").pvalue
    except ValueError:
        w = float("nan")
    return {"metrica": metrica, "A": a, "B": b, "media_A": float(da[metrica].mean()), "de_A": float(da[metrica].std()),
            "media_B": float(db[metrica].mean()), "de_B": float(db[metrica].std()),
            "dif_por_pliegue": d.round(4).tolist(), "dif_media": float(d.mean()), "dif_de": float(d.std(ddof=1)),
            "ic95": [float(d.mean() - stats.t.ppf(0.975, k - 1) * se), float(d.mean() + stats.t.ppf(0.975, k - 1) * se)],
            "t_pareada_p_unilateral": float(t.pvalue), "wilcoxon_p_unilateral": float(w),
            "d_z_cohen": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf"),
            "pliegues_B_mejor": int((d > 0).sum()), "pliegues_empate": int((d == 0).sum())}


def por_clase(y, proba) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y, proba.argmax(1), labels=range(len(CLASES)), zero_division=0)
    return pd.DataFrame({"clase": [NOMBRES[c] for c in CLASES], "precision": p, "recall": r, "f1": f, "soporte": s})


def analisis_errores(y, p_base, p_nuevo, archivos, autores) -> pd.DataFrame:
    """Errores corregidos e introducidos por el modelo nuevo frente a la línea base."""
    ab, an = p_base.argmax(1) == y, p_nuevo.argmax(1) == y
    filas = []
    for i in np.where((ab != an) | ~an | ~ab)[0]:
        filas.append({"archivo": archivos[i], "autor": autores[i], "real": NOMBRES[CLASES[y[i]]],
                      "pred_E03": NOMBRES[CLASES[p_base[i].argmax()]], "conf_E03": float(p_base[i].max()),
                      "pred_nuevo": NOMBRES[CLASES[p_nuevo[i].argmax()]], "conf_nuevo": float(p_nuevo[i].max()),
                      "tipo": "corregido" if an[i] and not ab[i] else ("introducido" if ab[i] and not an[i] else "persistente")})
    return pd.DataFrame(filas).sort_values(["tipo", "conf_nuevo"], ascending=[True, False])


def matrices_confusion(y, probas: dict[str, np.ndarray]) -> Path:
    fig, ax = plt.subplots(1, len(probas), figsize=(3.3 * len(probas), 3.0))
    for a, (nombre, p) in zip(np.atleast_1d(ax), probas.items()):
        m = confusion_matrix(y, p.argmax(1), labels=range(len(CLASES)))
        a.imshow(m, cmap="Blues")
        for i in range(len(CLASES)):
            for j in range(len(CLASES)):
                a.text(j, i, m[i, j], ha="center", va="center", fontsize=7, color="white" if m[i, j] > m.max() / 2 else "black")
        etq = [NOMBRES[c] for c in CLASES]
        a.set(xticks=range(5), yticks=range(5), title=nombre, xlabel="predicha", ylabel="real")
        a.set_xticklabels(etq, rotation=45, ha="right", fontsize=7)
        a.set_yticklabels(etq, fontsize=7)
    fig.tight_layout()
    ruta = FIG / "confusion_prueba.png"
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def guardar(obj, nombre: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(TAB / f"{nombre}.csv", index=False)
    else:
        (TAB / f"{nombre}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def fase_final(ide: str = "FIN") -> dict:
    """Prueba (abierta una sola vez): E03 frente al finalista por clase, confusión y errores."""
    part = pd.read_csv(OUT / "particion.csv")
    te = np.where(part.split == "test")[0]
    y = part.label.map({c: i for i, c in enumerate(CLASES)}).to_numpy()[te]
    p03 = np.load(OUT / "E03_proba_test.npy")
    res = json.loads((A2 / f"{ide}_final_resumen.json").read_text())
    semillas = sorted(res)
    probas = {s: np.load(A2 / f"{s}_proba_test.npy") for s in semillas}
    sel = json.loads(Path("modelos/finalista.json").read_text())["origen"]
    tabla = por_clase(y, p03).rename(columns=lambda c: c if c in ("clase", "soporte") else f"{c}_E03")
    pc = [por_clase(y, p) for p in probas.values()]
    for m in ("precision", "recall", "f1"):
        v = np.stack([d[m] for d in pc])
        tabla[f"{m}_fin_media"], tabla[f"{m}_fin_de"] = v.mean(0), v.std(0, ddof=1)
    guardar(tabla, "por_clase_prueba")
    err = analisis_errores(y, p03, probas[sel], part.file.to_numpy()[te], part.owner.to_numpy()[te])
    guardar(err, "errores_prueba")
    matrices_confusion(y, {"E03 (sonda congelada)": p03, f"Finalista ({sel})": probas[sel]})
    detalle = {s: json.loads((A2 / f"{s}_resultados.json").read_text()) for s in semillas}
    resumen = {"semillas": semillas, "publicado": sel,
               "test_macro_f1": [detalle[s]["test"]["macro_f1"] for s in semillas],
               "mcnemar": {s: detalle[s]["mcnemar_vs_E03"] for s in semillas},
               "tipos_error": err.tipo.value_counts().to_dict()}
    guardar(resumen, "final_resumen_analisis")
    return resumen


def costo() -> pd.DataFrame:
    """Parámetros entrenables, tiempo de ajuste y latencia por configuración."""
    ent = pd.read_csv(A2 / "arquitectura_entrenables.csv", index_col=0).parametros
    filas = []
    for ide in ("E04b", "E05", "E06", "E07", "E09a", "E09b", "E09c", "E10"):
        if (A2 / f"{ide}_estudio.csv").exists():
            e = estudio(ide).sort_values("val_loss").iloc[0]
            filas.append({"estudio": ide, "run": e.run, "entrenables": int(e.entrenables), "segundos_ft": float(e.segundos_ft),
                          "val_loss": float(e.val_loss), "val_macro_f1": float(e.val_macro_f1)})
    df = pd.DataFrame(filas)
    df.attrs["E03_entrenables"] = int(ent.get("E03", 1925))
    guardar(df, "costo_y_mejor_por_estudio")
    return df


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ids = {"E04": "LP-FT k = 2, 7 épocas", "E04b": "LP-FT k = 2, 3 épocas", "E05": "LP-FT k = 4, 3 épocas",
           "E06": "LoRA r = 8, 3 épocas", "E07": "k = 2 + aumentación, 7 épocas", "E09a": "SGD Nesterov",
           "E09b": "decaimiento 0 y 0,2", "E09c": "profundidad estocástica 0,1",
           "E10": "EfficientNet-B0 (completo)", "E10b": "EfficientNet-B0, lr 3e-3"}
    tablas = []
    for ide in ids:
        if (A2 / f"{ide}_estudio.csv").exists():
            curvas_estudio(ide)
            tablas.append(estudio(ide).assign(estudio=ide))
    grafico_eje({k: v for k, v in ids.items() if k in ("E04", "E04b", "E10")})
    if tablas:
        guardar(pd.concat(tablas, ignore_index=True), "estudios")
    costo()
    if (A2 / "CV_FIN_cv.csv").exists():
        comp = {m: comparacion_pareada("CV_E03", "CV_FIN", m) for m in ("macro_f1", "macro_f1_baja_res", "ece", "cobertura")}
        guardar(comp, "comparacion_pareada_cv")
    if (A2 / "FIN_final_resumen.json").exists():
        fase_final()
    print("figuras:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
