# Ficha del modelo (borrador): E03, sonda lineal sobre DINOv2 ViT-S/14 congelado

Estructura de Mitchell et al. (2019). Es un borrador de la Actividad 1. La versión final, con el modelo que gane la Actividad 2, se publica el 22 de octubre de 2026.

## Detalles del modelo
- **Equipo:** Grupo 15, Técnicas Avanzadas de Modelado en IA (MIA 2026-5, Universidad de La Sabana). Integrantes: Leonar Socarrás Molina, PhD, y John Jairo Serrano Cifuentes, Esp.
- **Arquitectura:**
  - Extractor: `vit_small_patch14_dinov2.lvd142m` de timm, congelado. Tiene 21,6 M de parámetros y licencia Apache 2.0 (Oquab et al., 2024).
  - Entrada: token de clase de 384 dimensiones.
  - Clasificador: estandarización seguida de regresión logística multinomial, con C = 0,1 elegido en validación. Tiene 1 925 parámetros entrenables.
- **Preprocesamiento:** redimensión bicúbica a 256 px, recorte central de 224 px y normalización con la media y la desviación de ImageNet.
- **Configuración y trazabilidad:** `configs/e03_dinov2_sonda.yaml`, con el run registrado en `mlflow.db`. La semilla del control de fuga es 2026.

## Uso previsto
- **Para qué sirve:** sugerir la clase de una foto de aficionado entre cinco flores ornamentales (margarita, diente de león, rosa, girasol y tulipán), para bitácoras escolares de observación botánica.
- **Abstención:** el sistema se abstiene cuando la probabilidad máxima es menor que 0,90. Ese umbral se fijó en validación para lograr una precisión de 99 % en las fotos aceptadas.
- **Usuarios:** estudiantes y docentes de ciencias, en un celular o un portátil sin GPU.
- **Usos fuera de alcance:**
  - Especies distintas de las cinco clases.
  - Decisiones sobre toxicidad, comestibilidad, uso medicinal o conservación.
  - Cualquier análisis de personas.
  - Registrar datos automáticamente sin revisión docente.

## Factores
- **Subgrupos analizados en la Actividad 2:** fotos casi monocromas (88, de las cuales 49 son dientes de león), oscuras (149), sobreexpuestas (32) y de baja resolución (397 con el lado menor por debajo de 200 px).
- **Condiciones de captura evaluadas:** desenfoque, compresión JPEG, subexposición y reducción de resolución (Hendrycks y Dietterich, 2019).

## Métricas
- **Primaria:** macro-F1, con IC 95 % por bootstrap que remuestrea autores, porque el autor es la unidad agrupada de la partición (Scikit-learn Developers, s. f.).
- **Secundarias:** exactitud balanceada, recall por clase, ECE con 15 contenedores (Guo et al., 2017), cobertura a precisión fija y latencia P95 en CPU. La selección de métricas por subgrupo sigue a Mitchell et al. (2019).

## Datos de evaluación
- **Partición:** prueba agrupada por autor de Flickr, con 547 fotos y ningún autor ni casi duplicado compartido con entrenamiento (`outputs/particion.csv`).
- **Validación cruzada:** 5 pliegues agrupados sobre entrenamiento y validación.

## Datos de entrenamiento
- **Conjunto:** tf_flowers (TensorFlow Team, 2019), con 2 564 fotos de entrenamiento y 548 de validación. La procedencia, la licencia y las limitaciones están en `docs/datasheet_tf_flowers.md`.

## Resultados cuantitativos (Actividad 1)
| Medida | Valor |
|---|---|
| Macro-F1 en prueba (IC 95 % por autor) | 0,987 (0,971 a 0,997) |
| Macro-F1 en validación | 0,950 |
| Macro-F1, CV agrupada k = 5 | 0,961 ± 0,009 |
| Recall mínimo por clase (prueba) | 0,968 |
| ECE en prueba / en validación | 0,009 / 0,023 |
| Cobertura a precisión ≥ 99 % (prueba / validación) | 96,0 % / 91,2 % |
| Macro-F1 con resolución a 1/4 (prueba) | 0,954 |
| Latencia P95 por imagen, CPU de 2 hilos | 141 ms |

## Consideraciones éticas
- **Solapamiento con el preentrenamiento:** el corpus LVD-142M se armó por recuperación desde conjuntos públicos (Oquab et al., 2024) y pudo incluir fotos parecidas a las de Flickr, así que las cifras con datos limpios son una cota optimista.
- **Privacidad:**
  - Algunas fotos muestran personas o viviendas de forma incidental.
  - Los nombres de usuario de Flickr se conservan solo como atribución, porque la licencia CC-BY 2.0 la exige.
  - Las fotos de campo que tomen los estudiantes requieren el consentimiento de sus acudientes (Ley 1581 de 2012) y se guardan sin rostros ni metadatos EXIF o GPS.

## Advertencias y recomendaciones
- **Dominio:** el modelo fue entrenado con flora templada y cultivada, así que no representa la flora del Caribe colombiano.
- **Uso recomendado:** como sugerencia que revisa el docente, nunca como identificación definitiva. La revisión humana es la medida de gestión del riesgo de uso indebido que adopta el equipo (National Institute of Standards and Technology [NIST], 2023).

## Referencias

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. En *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321-1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. En *International Conference on Learning Representations*. https://arxiv.org/abs/1903.12261

Ley 1581 de 2012. Por la cual se dictan disposiciones generales para la protección de datos personales. 17 de octubre de 2012. *Diario Oficial* No. 48.587.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220-229). ACM. https://doi.org/10.1145/3287560.3287596

National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., ... Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2304.07193

Scikit-learn Developers. (s. f.). *Cross-validation: Evaluating estimator performance*. https://scikit-learn.org/stable/modules/cross_validation.html

TensorFlow Team. (2019). *Flowers* [Conjunto de datos]. TensorFlow Datasets. https://www.tensorflow.org/datasets/catalog/tf_flowers
