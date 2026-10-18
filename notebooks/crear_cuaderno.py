"""Genera notebooks/01_plan_experimental_lineas_base.ipynb (se ejecuta con nbconvert)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
c = []
c.append(md("""# Proyecto integrador · Actividad 1 (U4): diseño experimental y línea base

**Técnicas Avanzadas de Modelado en IA, MIA 2026-5, Universidad de La Sabana.**
Equipo: Leonar Socarrás Molina, PhD, y John Jairo Serrano Cifuentes, Esp. Docente: PhD. Oscar Alberto Rodríguez Meléndez.

Este cuaderno comunica la evidencia del plan experimental; la lógica reutilizable vive en `src/flores/`, las decisiones en `configs/` y la evidencia generada en `outputs/` (estructura recomendada en la sesión U4S2). Orden de ejecución:

1. Verificar el archivo de datos (SHA-256).
2. Auditoría de calidad e inventario.
3. Grupos de fuga y partición estratificada por grupos.
4. Pruebas automáticas de la partición y de las métricas.
5. Líneas base E00 a E03 con registro en MLflow.
6. Comparación, incertidumbre y figuras."""))
c.append(code("""import hashlib, json, subprocess, sys
from pathlib import Path
import pandas as pd
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
OUT, DATOS = RAIZ / "outputs", RAIZ / "data"
RECALCULAR = False   # True: reconstruye inventario, partición y líneas base desde cero
pd.set_option("display.precision", 4)
print(RAIZ)"""))
c.append(md("## 1. Integridad del archivo de datos\nDescarga oficial: `https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz`. El SHA-256 debe coincidir con el registrado en la hoja de datos."))
c.append(code("""ESPERADO = "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c"
tgz = DATOS / "flower_photos.tgz"
if tgz.exists():
    h = hashlib.sha256(tgz.read_bytes()).hexdigest()
    print("SHA-256 coincide:", h == ESPERADO)
else:
    print("Archivo .tgz no presente; se usa la carpeta extraída data/flower_photos")"""))
c.append(md("## 2. Auditoría inicial de calidad"))
c.append(code("""if RECALCULAR:
    subprocess.run([sys.executable, str(RAIZ / "src/flores/auditoria.py"), str(DATOS / "flower_photos"), str(OUT)], check=True)
inv = pd.read_csv(OUT / "inventario_grupos.csv")
k = inv[inv.exclusion.fillna("") == ""]
lado = k[["width", "height"]].min(axis=1)
auditoria = pd.Series({
    "imágenes totales": len(inv),
    "sin línea de licencia": int((inv.exclusion == "sin_licencia").sum()),
    "etiqueta contradictoria (mismos píxeles, 2 clases)": int((inv.exclusion == "etiqueta_contradictoria").sum()),
    "imágenes usadas": len(k),
    "autores de Flickr": k.owner.nunique(),
    "copias exactas de píxeles": int(k.md5_pixels.duplicated().sum()),
    "casi monocromas (saturación < 0,08)": int((k.saturation_mean < 0.08).sum()),
    "oscuras (brillo < 50)": int((k.brightness_mean < 50).sum()),
    "sobreexpuestas (brillo > 200)": int((k.brightness_mean > 200).sum()),
    "borrosas (var. laplaciano < 50)": int((k.laplacian_var < 50).sum()),
    "ruido alto (sigma Immerkær > 15)": int((k.noise_sigma > 15).sum()),
    "lado menor < 200 px": int((lado < 200).sum()),
})
auditoria.to_frame("valor")"""))
c.append(code("""pd.crosstab(k.label, (k.saturation_mean < 0.08).map({True: "casi monocroma", False: "color"}))"""))
c.append(md("## 3. Grupos de fuga y partición\nUnidad de observación independiente: el **autor de Flickr** (misma cámara, jardín o planta). Un grupo une imágenes con el mismo autor, los mismos píxeles o pHash a ≤ 6 bits; cada grupo cae completo en una sola partición."))
c.append(code("""if RECALCULAR:
    subprocess.run([sys.executable, str(RAIZ / "src/flores/particion.py"), str(OUT)], check=True)
rep = json.loads((OUT / "particion_reporte.json").read_text())
p = pd.read_csv(OUT / "particion.csv")
display(pd.crosstab(p.label, p.split, margins=True)[["train", "val", "test", "All"]])
pd.DataFrame({"estratificada por grupos": rep["contaminacion_por_grupos"]["test"],
              "aleatoria estratificada": rep["contaminacion_aleatoria"]["test"]})"""))
c.append(md("## 4. Pruebas automáticas"))
c.append(code("""r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:warnings", str(RAIZ / "tests")], capture_output=True, text=True, cwd=RAIZ)
print(r.stdout[-600:])"""))
c.append(md("## 5. Líneas base E00 a E03\nCada experimento hereda de `BaseExperimento` (contrato común), ajusta solo con entrenamiento, elige hiperparámetros con validación y evalúa una sola vez en prueba. Los runs quedan en `mlruns/`."))
c.append(code("""if RECALCULAR:
    cfgs = ["configs/e00_mayoritaria.yaml", "configs/e01_hog_color_lr.yaml", "configs/e03_dinov2_sonda.yaml",
            "configs/e02_cnn_s11.yaml", "configs/e02_cnn_s22.yaml", "configs/e02_cnn_s33.yaml"]
    subprocess.run([sys.executable, "scripts/ejecutar.py", *cfgs], check=True, cwd=RAIZ)
filas = {}
for f in sorted(OUT.glob("E0*_resultados.json")):
    r = json.loads(f.read_text()); t = r["test"]
    filas[f.name.split("_resultados")[0]] = {
        "macro-F1": t["macro_f1"], "IC95 inf": t["macro_f1_ic95"][0], "IC95 sup": t["macro_f1_ic95"][1],
        "exactitud": t["accuracy"], "exact. balanceada": t["balanced_acc"], "log-loss": t["log_loss"],
        "ECE": t["ece"], "macro-F1 val": r["val"]["macro_f1"], "P95 ms": r["extra"].get("p95_ms")}
tabla = pd.DataFrame(filas).T
tabla"""))
c.append(code("""cnn = tabla.loc[[i for i in tabla.index if i.startswith("E02")], ["macro-F1", "exactitud", "ECE"]]
print("E02 media ± DE entre semillas:")
print((cnn.mean().round(4).astype(str) + " ± " + cnn.std(ddof=1).round(4).astype(str)).to_string())"""))
c.append(code("""rec = pd.DataFrame({k: {c: json.loads((OUT / f"{k}_resultados.json").read_text())["test"][f"recall_{c}"]
                        for c in ["daisy", "dandelion", "roses", "sunflowers", "tulips"]} for k in tabla.index})
rec.T  # recall por clase en prueba"""))
c.append(md("## 6. Comparación pareada, control de fuga y figuras"))
c.append(code("""import numpy as np
from flores.metricas import mcnemar_exacto
yt = p[p.split == "test"].label.map({c: i for i, c in enumerate(["daisy", "dandelion", "roses", "sunflowers", "tulips"])}).to_numpy()
ok = {k: np.load(OUT / f"{k}_proba_test.npy").argmax(1) == yt for k in tabla.index}
print("McNemar E03 vs E02_s33:", mcnemar_exacto(ok["E03"], ok["E02_s33"]))
print("McNemar E03 vs E01:   ", mcnemar_exacto(ok["E03"], ok["E01"]))
e03 = json.loads((OUT / "E03_resultados.json").read_text())["extra"]
print("Control con etiquetas permutadas (macro-F1 val):", round(e03["val_macro_f1_etiquetas_permutadas"], 4))"""))
c.append(md("### Validación cruzada agrupada por autor y robustez de E03\nLa prueba se toca una sola vez; la CV agrupada (k = 5) usa solo entrenamiento y validación. Las perturbaciones imitan fotos de celular."))
c.append(code("""if RECALCULAR:
    subprocess.run([sys.executable, "src/flores/diagnostico.py", "data/flower_photos", "outputs"], check=True, cwd=RAIZ)
dg = json.loads((OUT / "E03_diagnostico.json").read_text())
cv = dg["cv_agrupada_k5"]
print(f"CV agrupada k=5: {cv['media']:.4f} ± {cv['de']:.4f}  pliegues {[round(v, 4) for v in cv['macro_f1_pliegues']]}")
pd.Series({k: v for k, v in dg["robustez_prueba_E03"].items()}).to_frame("macro-F1 / %")"""))
c.append(md("### Predicción selectiva y comparaciones pareadas de E03\nUmbral de confianza fijado en validación para precisión ≥ 99 % en lo aceptado; McNemar exacto frente a cada línea base."))
c.append(code("""subprocess.run([sys.executable, "src/flores/seleccion.py", "outputs"], check=True, cwd=RAIZ, capture_output=True)
sc = json.loads((OUT / "E03_seleccion_y_comparaciones.json").read_text())
display(pd.Series(sc["seleccion"]).to_frame("E03"))
pd.DataFrame(sc["mcnemar_E03_vs"]).T"""))
c.append(code("""subprocess.run([sys.executable, "scripts/figuras.py", "outputs"], check=True, cwd=RAIZ)
from IPython.display import Image as Img
for f in ["fig1_particion.png", "fig2_lineas_base.png", "fig3_curvas_cnn.png"]:
    display(Img(filename=str(OUT / "figuras" / f), width=760))"""))
c.append(md("## 7. Lectura de los resultados\nLa interpretación completa está en el plan experimental (PDF). Este cuaderno no selecciona modelo final con prueba: la prueba solo se usa para reportar la línea base de referencia que la Actividad 2 debe superar con el mismo protocolo."))
nb = nbf.v4.new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, Path(__file__).with_name("01_plan_experimental_lineas_base.ipynb"))
print("ok")
