"""Genera notebooks/02_ejecucion_optimizacion_validacion.ipynb (se ejecuta con nbconvert).

El cuaderno comunica la evidencia de la A2 y del modelo publicado (A3). La lógica vive en src/flores,
las decisiones en configs/a2 y los resultados en outputs/a2; con RECALCULAR = True se reentrena todo.
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
c = []
c.append(md("""# Proyecto integrador · Actividades 2 y 3 (U4): ejecución, optimización, validación comparativa y modelo publicado

**Técnicas Avanzadas de Modelado en IA, MIA 2026-5, Universidad de La Sabana.** Grupo 15: Leonar Socarrás Molina, PhD, y John Jairo Serrano Cifuentes, Esp. Docente: PhD. Óscar Alberto Rodríguez Meléndez.

Pregunta: ¿el ajuste fino de una red profunda supera de forma demostrable a la línea base E03 (sonda lineal sobre DINOv2 ViT-S/14 congelado)?

Orden del cuaderno:
1. Entorno, semillas y pruebas automáticas.
2. Arquitecturas capa por capa (ViT y CNN) y parámetros entrenables.
3. Búsqueda de hiperparámetros: estudios E04 a E10, curvas y diagnóstico.
4. Validación cruzada agrupada k = 5 frente a E03, con pruebas pareadas.
5. Prueba única del finalista (3 semillas), por clase, errores, robustez, calibración y latencia.
6. Modelo publicado e inferencia de ejemplo.
7. Consulta de MLflow."""))
c.append(code("""import json, os, subprocess, sys, platform
from pathlib import Path
import pandas as pd
from IPython.display import Image, display
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(RAIZ); sys.path.insert(0, str(RAIZ / "src"))
A2, TAB, FIG = RAIZ / "outputs/a2", RAIZ / "outputs/a2/tablas", RAIZ / "outputs/a2/figuras"
RECALCULAR = False   # True: reconstruye la caché y reentrena estudios, CV y final (varias horas en CPU)
pd.set_option("display.precision", 4)
import torch, timm, sklearn, mlflow
print(platform.python_version(), "torch", torch.__version__, "timm", timm.__version__, "sklearn", sklearn.__version__, "mlflow", mlflow.__version__)"""))
c.append(md("## 1. Pruebas automáticas\nVerifican la partición sin fuga, las métricas, que la cola sobre la caché reproduce el ViT completo, que LoRA se fusiona sin error, el número de parámetros entrenables y que la cabeza inicial equivale a E03."))
c.append(code("""r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
print(r.stdout[-400:])"""))
c.append(md("## 2. Arquitecturas\nDINOv2 ViT-S/14: se congelan los bloques 0 a 9 y se ajustan los dos últimos; la cabeza arranca desde la sonda E03 (LP-FT). EfficientNet-B0: se ajusta completa como contraste convolucional."))
c.append(code("""display(Image(FIG / "diagrama_arquitecturas.png", width=900))
display(pd.read_csv(A2 / "arquitectura_vit.csv"))
display(pd.read_csv(A2 / "arquitectura_cnn.csv"))
pd.read_csv(A2 / "arquitectura_entrenables.csv")"""))
c.append(md("## 3. Búsqueda de hiperparámetros\nCada estudio responde a una pregunta (YAML en `configs/a2`). La tasa de aprendizaje es hiperparámetro de ajuste: se busca dentro de cada condición y la rejilla se amplía si el óptimo cae en un borde (E04 → E04b, E10 → E10b)."))
c.append(code("""if RECALCULAR:
    # E05, E06 y E07 tienen configuración pero no se ejecutaron por el plazo de entrega (trabajo futuro)
    for y in ["e04_lpft_k2", "e04b_lpft_k2_corto", "e09a_sgd", "e09b_decaimiento", "e09c_drop_path"]:
        subprocess.run([sys.executable, "-m", "flores.experimentos_a2", "estudio", f"configs/a2/{y}.yaml"], check=True, env={**os.environ, "PYTHONPATH": "src"})
    for y in ["e10_cnn_effnet", "e10b_cnn_effnet_borde"]:
        subprocess.run([sys.executable, "-m", "flores.cnn_contraste", f"configs/a2/{y}.yaml"], check=True, env={**os.environ, "PYTHONPATH": "src"})
est = pd.read_csv(TAB / "estudios.csv")
est[["estudio", "run", "ajustar", "lora_r", "aumentacion", "optimizador", "lr", "decaimiento", "drop_path", "epocas", "val_loss", "val_macro_f1", "mejor_epoca", "segundos_ft", "entrenables"]]"""))
c.append(code("""display(Image(FIG / "eje_lr.png", width=520))
for f in sorted(FIG.glob("curvas_*.png")):
    print(f.stem); display(Image(f, width=850))"""))
c.append(md("""**Lectura de las curvas.** En E04 (7 épocas) la pérdida de validación con tasa 3e-4 alcanza su mínimo en la época 1 y luego sube mientras la exactitud de entrenamiento llega a 1,0: el ajuste memoriza el entrenamiento (sobreajuste). Con 1e-5 la pérdida no baja del punto de partida (subajuste) y con 3e-5 aparece un pico en la época 2 (inestabilidad transitoria del calentamiento). Por eso E04b reduce el presupuesto a 3 épocas. En la CNN las curvas siguen bajando al final: con 15 minutos de CPU no converge."""))
c.append(md("## 4. Validación cruzada agrupada frente a E03\nMismos 5 pliegues que E03 (StratifiedGroupKFold, semilla 2026). En cada pliegue, 15 % de los grupos de entrenamiento calibran temperatura y umbral; el pliegue retenido solo se evalúa."))
c.append(code("""if RECALCULAR:
    for y in ["cv_e03", "cv_finalista"]:
        subprocess.run([sys.executable, "-m", "flores.experimentos_a2", "cv", f"configs/a2/{y}.yaml"], check=True, env={**os.environ, "PYTHONPATH": "src"})
cv = pd.concat([pd.read_csv(A2 / "CV_E03_cv.csv").assign(modelo="E03"), pd.read_csv(A2 / "CV_FIN_cv.csv").assign(modelo="finalista")])
display(cv.groupby("modelo")[["macro_f1", "macro_f1_baja_res", "ece", "cobertura", "precision_aceptadas", "recall_min"]].agg(["mean", "std"]))
pd.DataFrame(json.loads((TAB / "comparacion_pareada_cv.json").read_text())).T"""))
c.append(md("## 5. Prueba única del finalista\nSemillas 11, 22 y 33. La prueba no interviene en ninguna selección; se publica la semilla con menor pérdida de validación."))
c.append(code("""if RECALCULAR:
    subprocess.run([sys.executable, "-m", "flores.experimentos_a2", "final", "configs/a2/final.yaml"], check=True, env={**os.environ, "PYTHONPATH": "src"})
    subprocess.run([sys.executable, "-m", "flores.analisis_a2"], check=True, env={**os.environ, "PYTHONPATH": "src"})
res = json.loads((A2 / "FIN_final_resumen.json").read_text())
display(pd.DataFrame(res).T)
display(pd.read_csv(TAB / "por_clase_prueba.csv"))
display(Image(FIG / "confusion_prueba.png", width=700))
print(json.loads((TAB / "final_resumen_analisis.json").read_text()))
pd.read_csv(TAB / "errores_prueba.csv")"""))
c.append(md("## 6. Modelo publicado e inferencia\nLos pesos de la cola (float16) y sus metadatos están en `modelos/`. El resto del ViT se descarga de Hugging Face con `timm`."))
c.append(code("""meta = json.loads(Path("modelos/finalista.json").read_text()); print(meta)
import hashlib; print("SHA-256 coincide:", hashlib.sha256(Path("modelos/finalista.pt").read_bytes()).hexdigest() == meta["sha256"])
from flores.predecir import cargar, predecir
fotos = sorted((RAIZ / "data/flower_photos").glob("*/*.jpg"))[:3] if (RAIZ / "data/flower_photos").exists() else []
if fotos:
    m, tf, meta = cargar(Path("modelos/finalista.pt"))
    for r in predecir(m, tf, meta, fotos):
        print(r)"""))
c.append(md("## 7. Consulta de MLflow\nCada run registra parámetros, métricas por época, etiquetas (estudio, pregunta, fase) y artefactos. `mlflow ui --backend-store-uri sqlite:///mlflow.db` abre la interfaz."))
c.append(code("""mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["flores-proyecto-integrador"])
cols = [k for k in ["tags.mlflow.runName", "tags.estudio", "tags.fase", "params.lr", "params.ajustar", "metrics.val_loss", "metrics.val_macro_f1", "metrics.test_macro_f1"] if k in runs]
runs[cols].sort_values("tags.mlflow.runName").to_csv(A2 / "runs_mlflow.csv", index=False)
print(len(runs), "runs registrados"); runs[cols].head(60)"""))
nb = nbf.v4.new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, Path(__file__).with_name("02_ejecucion_optimizacion_validacion.ipynb"))
print("cuaderno escrito")
