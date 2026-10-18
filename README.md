# Proyecto integrador: clasificación de cinco flores con transferencia desde un modelo fundacional

Técnicas Avanzadas de Modelado en IA, MIA 2026-5, Universidad de La Sabana.
Equipo: Leonar Socarrás Molina, PhD (leonarsomo@unisabana.edu.co) y John Jairo Serrano Cifuentes, Esp. (johnseci@unisabana.edu.co).
Docente: PhD. Oscar Alberto Rodríguez Meléndez.

Este repositorio acompaña la **Actividad 1 de la Unidad 4** (diseño experimental, partición sin fuga y línea base) y servirá de base para las Actividades 2 y 3.

## Estructura

```
configs/      decisiones experimentales (una YAML por run; comun.yaml = factores fijos)
src/flores/   auditoria.py · particion.py · modelos.py · metricas.py · experimentos.py
scripts/      ejecutar.py (punto de entrada) · figuras.py
tests/        pruebas de partición sin fuga y de métricas
notebooks/    01_plan_experimental_lineas_base.ipynb (comunica la evidencia)
docs/         datasheet_tf_flowers.md (Gebru et al., 2021)
outputs/      inventario, grupos, partición, resultados por run, figuras
mlflow.db     registro MLflow (SQLite) de parámetros y métricas; mlruns/ guarda los artefactos
```

## Reproducir desde cero (CPU, ~35 min)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data && cd data
curl -O https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
echo "4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c  flower_photos.tgz" | shasum -a 256 -c   # en Linux también sirve: sha256sum -c
tar xzf flower_photos.tgz && cd ..
python src/flores/auditoria.py data/flower_photos outputs     # inventario y calidad
python src/flores/particion.py outputs                         # grupos de fuga y partición
python -m pytest -q tests                                      # 8 pruebas deben pasar
python scripts/ejecutar.py configs/e00_mayoritaria.yaml configs/e01_hog_color_lr.yaml \
       configs/e03_dinov2_sonda.yaml configs/e02_cnn_s11.yaml configs/e02_cnn_s22.yaml configs/e02_cnn_s33.yaml
python src/flores/diagnostico.py data/flower_photos outputs  # CV agrupada k=5 y robustez de E03
python src/flores/seleccion.py outputs                       # abstención calibrada y McNemar
python src/flores/tiempos.py data/flower_photos outputs      # tiempos de entrenamiento separados de extracción y bootstrap
python scripts/figuras.py outputs
mlflow ui --backend-store-uri sqlite:///mlflow.db            # comparar runs
```

Semillas: partición 2026; CNN 11, 22 y 33; control de etiquetas permutadas 2026. `torch.use_deterministic_algorithms(True)` en la CNN. Los pesos de DINOv2 se descargan de Hugging Face mediante `timm` (`vit_small_patch14_dinov2.lvd142m`, licencia Apache 2.0).

## Resultados de la línea base (Actividad 1)

| Run | Modelo | Macro-F1 en prueba (IC 95 % por autor) |
|---|---|---|
| E00 | Clase mayoritaria | 0,079 (0,062–0,100) |
| E01 | HOG + HSV + regresión logística | 0,569 (0,503–0,612) |
| E02 | CNN desde cero (semillas 11, 22, 33; 5,4 min de entrenamiento por semilla) | 0,718 ± 0,017 |
| E03 | DINOv2 ViT-S/14 congelado + RL | 0,987 (0,971–0,997) |

- Validación cruzada agrupada por autor (k = 5, sin tocar prueba) de E03: 0,961 ± 0,009. La prueba resultó más fácil que todos los pliegues; la Actividad 2 seleccionará modelos con esta CV.
- Robustez de E03: desenfoque 0,977; JPEG 20 0,988; subexposición 0,987; resolución 1/4 0,954.
- La partición agrupada elimina la contaminación que una partición aleatoria introduciría en 74,3 % de las fotos de prueba (autor visto en entrenamiento).

## Documentos

- `docs/U4A1_Plan_experimental_Grupo15.pdf`: plan experimental entregado (6 páginas).
- `docs/U4A1_Fundamentacion_teorica_Grupo15.pdf`: fundamentación teórica y técnica (APA 7).
- `docs/datasheet_tf_flowers.md`: hoja de datos.
- `docs/ficha_modelo_E03.md`: ficha del modelo de la línea base E03 (borrador; versión final en la Actividad 2).

## Datos y licencia

Las imágenes **no** se incluyen. Cada foto conserva su licencia CC-BY 2.0 y la atribución de `data/flower_photos/LICENSE.txt`. Ver `docs/datasheet_tf_flowers.md`.

## Uso previsto y límites

Prototipo académico para clasificar cinco flores comunes en fotos de aficionados y abstenerse cuando la confianza calibrada es baja. No apto para decisiones sobre toxicidad, comestibilidad, uso medicinal o conservación, ni para especies fuera de las cinco clases.
