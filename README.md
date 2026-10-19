# Proyecto integrador: clasificación de cinco flores con transferencia desde un modelo fundacional

Técnicas Avanzadas de Modelado en IA, MIA 2026-5, Universidad de La Sabana.
Equipo: Leonar Socarrás Molina, PhD (leonarsomo@unisabana.edu.co) y John Jairo Serrano Cifuentes, Esp. (johnseci@unisabana.edu.co).
Docente: PhD. Oscar Alberto Rodríguez Meléndez.

Este repositorio acompaña la **Actividad 1 de la Unidad 4** (diseño experimental, partición sin fuga y línea base) y servirá de base para las Actividades 2 y 3. El conjunto es tf_flowers (TensorFlow Team, 2019) y el modelo fundacional es DINOv2 (Oquab et al., 2024).

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
- La partición agrupada elimina la contaminación que una partición aleatoria introduciría en 74,3 % de las fotos de prueba (autor visto en entrenamiento), el problema que describen Scikit-learn Developers (s. f.) para datos agrupados.

## Documentos

- `docs/U4A1_Plan_experimental_Grupo15.pdf`: plan experimental entregado (6 páginas).
- `docs/U4A1_Fundamentacion_teorica_Grupo15.pdf`: fundamentación teórica y técnica (APA 7).
- `docs/datasheet_tf_flowers.md`: hoja de datos según Gebru et al. (2021).
- `docs/ficha_modelo_E03.md`: ficha del modelo de la línea base E03 según Mitchell et al. (2019) (borrador; versión final en la Actividad 2).

## Datos y licencia

Las imágenes **no** se incluyen. Cada foto conserva su licencia CC-BY 2.0 y la atribución de `data/flower_photos/LICENSE.txt`. Ver `docs/datasheet_tf_flowers.md`.

## Uso previsto y límites

Prototipo académico para clasificar cinco flores comunes en fotos de aficionados y abstenerse cuando la confianza calibrada es baja (Guo et al., 2017). No apto para decisiones sobre toxicidad, comestibilidad, uso medicinal o conservación, ni para especies fuera de las cinco clases.

## Actividad 2 y 3: ajuste fino, validación y modelo publicado

La Actividad 2 compara estrategias de ajuste fino de DINOv2 ViT-S/14 frente a la sonda E03 y un contraste con EfficientNet-B0; la Actividad 3 publica el modelo final con su ficha. El finalista es un LP-FT de los bloques 10 y 11 más cabeza lineal, iniciado desde E03 (AdamW, tasa máxima 3e-4, OneCycle, 3 épocas), con temperatura y umbral de abstención fijados en validación. Sus cifras están en [`docs/ficha_modelo.md`](docs/ficha_modelo.md).

### Estructura nueva

```
configs/a2/                    una YAML por estudio (e04 a e10), CV (cv_e03, cv_finalista) y evaluación final (final.yaml)
src/flores/ajuste_fino.py      caché de tokens del bloque de corte, cola ajustable del ViT (LP-FT y LoRA), degradaciones
src/flores/experimentos_a2.py  subcomandos estudio, cv y final; registro en MLflow y exportación del modelo
src/flores/cnn_contraste.py    E10: contraste con EfficientNet-B0 (ImageNet) ajustada por completo, mismo protocolo
src/flores/arquitectura.py     inventario capa por capa de ViT y CNN: formas, parámetros y partes entrenadas
src/flores/analisis_a2.py      curvas, comparación pareada, análisis de error, tablas y figuras (lee outputs/a2)
src/flores/predecir.py         inferencia: foto -> clase calibrada o abstención
tests/test_ajuste_fino.py      LoRA, parámetros entrenables, cola equivalente al ViT y sonda E03 plegada
modelos/finalista.pt           pesos de la cola (bloques 10 y 11 y cabeza) en float16
modelos/finalista.json         bloque de corte, bloques ajustados, temperatura, umbral, criterio y SHA-256
docs/ficha_modelo.md           ficha del modelo final (Mitchell et al., 2019)
```

### Reproducir la Actividad 2 (en orden)

Requiere haber ejecutado antes la auditoría y la partición de la Actividad 1.

```bash
# 1. caché de tokens a partir del bloque 8, con 3 vistas por imagen (~12 min)
PYTHONPATH=src python -m flores.ajuste_fino cache data/flower_photos outputs --desde 8 --vistas 3
# 2. estudios de ajuste fino, uno por archivo (e04_lpft_k2, e05_lpft_k4, e06_lora_r8, e07_aumentacion, e09a_sgd, ...)
HILOS=2 PYTHONPATH=src python -m flores.experimentos_a2 estudio configs/a2/<archivo>.yaml
# 3. contraste con CNN preentrenada
PYTHONPATH=src python -m flores.cnn_contraste configs/a2/e10_cnn_effnet.yaml
# 4. validación cruzada agrupada k = 5 de la línea base y del finalista
HILOS=2 PYTHONPATH=src python -m flores.experimentos_a2 cv configs/a2/cv_e03.yaml
HILOS=2 PYTHONPATH=src python -m flores.experimentos_a2 cv configs/a2/cv_finalista.yaml
# 5. evaluación única en prueba con 3 semillas y exportación a modelos/
HILOS=2 PYTHONPATH=src python -m flores.experimentos_a2 final configs/a2/final.yaml
# 6. curvas, comparación pareada, análisis de error, tablas y figuras
PYTHONPATH=src python -m flores.analisis_a2
# 7. inferencia con el modelo publicado
PYTHONPATH=src python -m flores.predecir modelos/finalista.pt foto.jpg
# 8. pruebas (14 deben pasar)
PYTHONPATH=src python -m pytest -q tests
```

**Semillas.** Partición 2026; estudios 2026; finalista 11, 22 y 33 (se publica la de menor pérdida de validación, nunca la de mejor prueba); pliegues de la CV con `random_state` 2026, los mismos para E03 y el finalista.

**Tiempos aproximados en CPU de 2 hilos.** Caché, unos 12 min; cada run de 3 épocas con 2 bloques ajustados, unos 6 min; cada run de la CNN, unos 15 min.

### Verificar los pesos publicados

El SHA-256 de `modelos/finalista.pt` es `{{SHA256_PESOS}}` y también consta en `modelos/finalista.json`.

```bash
shasum -a 256 modelos/finalista.pt     # en Linux: sha256sum modelos/finalista.pt
```

El valor debe coincidir exactamente; si no coincide, el archivo está incompleto o es otra versión.

### Licencia y cómo citar

- **Código y pesos ajustados:** licencia MIT ([`LICENSE`](LICENSE)).
- **Fotografías:** CC-BY 2.0 por imagen, con la atribución de `data/flower_photos/LICENSE.txt`; no se redistribuyen.
- **Pesos base de DINOv2:** Apache 2.0, descargados de Hugging Face mediante timm.
- **Cita:** usar [`CITATION.cff`](CITATION.cff); GitHub muestra el enlace "Cite this repository" con formatos APA y BibTeX. Contacto: leonarsomo@unisabana.edu.co, johnseci@unisabana.edu.co o issues del repositorio.

## Referencias

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86-92. https://doi.org/10.1145/3458723

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. En *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321-1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220-229). ACM. https://doi.org/10.1145/3287560.3287596

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., ... Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2304.07193

Scikit-learn Developers. (s. f.). *Cross-validation: Evaluating estimator performance*. https://scikit-learn.org/stable/modules/cross_validation.html

TensorFlow Team. (2019). *Flowers* [Conjunto de datos]. TensorFlow Datasets. https://www.tensorflow.org/datasets/catalog/tf_flowers
