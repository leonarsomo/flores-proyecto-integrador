# Ficha del modelo: LP-FT de DINOv2 ViT-S/14 con abstención calibrada (v1.0.0)

Estructura de Mitchell et al. (2019), nueve secciones. Sustituye al borrador `docs/ficha_modelo_E03.md`, que se conserva como registro de la línea base. Las cifras del modelo final se calculan con `scripts/a2/rellenar_fichas.py` a partir de las probabilidades guardadas por la fase final; ninguna se escribió a mano.

## 1. Detalles del modelo

- **Desarrolladores:** Grupo 15, Técnicas Avanzadas de Modelado en IA (MIA 2026-5, Maestría en Inteligencia Artificial, Universidad de La Sabana): Leonar Socarrás Molina, PhD, y John Jairo Serrano Cifuentes, Esp. Docente: PhD. Óscar Alberto Rodríguez Meléndez.
- **Fecha y versión:** 18 de octubre de 2026; versión 1.0.0, igual al tag `v1.0.0` del repositorio. Identificador MLflow: run `FIN_s33` del experimento flores-proyecto-integrador (mlflow.db).
- **Arquitectura:** `vit_small_patch14_dinov2.lvd142m` de timm (21,6 M de parámetros; Oquab et al., 2024). Se ajustan los dos últimos bloques del transformador (10 y 11) y una cabeza lineal sobre el token de clase de 384 dimensiones; los bloques 0 a 9 quedan congelados. Parámetros entrenables: 3 553 157 de 21,6 M.
- **Entrenamiento (LP-FT):** la cabeza se inicia desde la sonda lineal E03 y después se ajustan juntos cabeza y bloques. AdamW, tasa máxima 3e-4 con calendario OneCycle, decaimiento de pesos 0,05, 3 épocas, lote 32, sin aumentación ni drop path. Se guarda el estado con menor pérdida de validación. Semillas 11, 22 y 33; se publica la semilla 33, elegida por menor log-loss de validación y nunca por su resultado en prueba.
- **Postprocesamiento:** escalado de temperatura (T = 1,158; Guo et al., 2017) y umbral de abstención sobre la probabilidad máxima calibrada (u = 0,730), ambos fijados en validación para una precisión de 99 % en las fotos aceptadas.
- **Artefactos:** `modelos/finalista.pt` (pesos de la cola en float16, 14,2 MB, SHA-256 0f55c4596dcc68c5f0eb5fb23c3649dff4ac12b05d9ae1a8e6d4b40ecf2c3d1e) y `modelos/finalista.json` (bloque de corte, bloques ajustados, temperatura, umbral, criterio de selección y hash). El resto del transformador se descarga de Hugging Face mediante timm.
- **Licencias:** código y pesos ajustados bajo MIT (archivo `LICENSE`); los pesos base de DINOv2 conservan su licencia Apache 2.0 y sus avisos; fotos bajo CC-BY 2.0 por imagen.
- **Cómo citar:** ver `CITATION.cff` en la raíz del repositorio. **Contacto:** leonarsomo@unisabana.edu.co, johnseci@unisabana.edu.co o issues en https://github.com/leonarsomo/flores-proyecto-integrador.

## 2. Uso previsto

- **Uso primario:** sugerir la clase de una foto de aficionado entre cinco flores ornamentales (margarita, diente de león, rosa, girasol y tulipán) para bitácoras escolares de observación botánica. La salida es "clase sugerida" con su probabilidad calibrada o "no sé, revisar" cuando la probabilidad cae bajo el umbral.
- **Usuarios primarios:** estudiantes y docentes de ciencias. El entorno de despliegue es un portátil escolar sin GPU, lo que justifica reportar la latencia P95 en una CPU de 2 hilos.
- **Usos fuera de alcance:** decidir sobre toxicidad, comestibilidad, uso medicinal o estado de conservación de una planta; clasificar especies fuera de las cinco clases (el modelo siempre responde con una de ellas o se abstiene, y no detecta especies nuevas); cualquier análisis de personas o lugares; registrar datos de forma automática sin revisión docente.

## 3. Factores

Factores relevantes no evaluables con tf_flowers: flora local del Cesar, tipo de celular de los estudiantes e iluminación tropical cenital. Por eso los factores de evaluación son sustitutos parciales dentro del conjunto. La tabla siguiente se calcula sobre la prueba (547 fotos). La métrica es el recall en las filas de clase, la exactitud en los subgrupos y la macro-F1 en las degradaciones, que se aplican a las 547 fotos.

| Familia | Nivel | n | Final (IC 95 %) | Cobertura final | E03 |
|---|---|---|---|---|---|
| Clase | margarita | 93 | 0,946 (0,880 a 0,977) | 96,8 % | 0,968 |
| Clase | diente de león | 134 | 1,000 (0,972 a 1,000) | 100,0 % | 1,000 |
| Clase | rosa | 95 | 0,958 (0,897 a 0,984) | 94,7 % | 0,968 |
| Clase | girasol | 105 | 0,990 (0,948 a 0,998) | 98,1 % | 1,000 |
| Clase | tulipán | 120 | 1,000 (0,969 a 1,000) | 99,2 % | 0,992 |
| Autor | con 1 foto | 146 | 0,973 (0,932 a 0,989) | 96,6 % | 0,993 |
| Autor | con varias fotos | 401 | 0,985 (0,968 a 0,993) | 98,5 % | 0,985 |
| Resolución nativa | lado menor < 200 px | 66 | 1,000 (0,945 a 1,000) | 100,0 % | 0,985 |
| Resolución nativa | lado menor ≥ 200 px | 481 | 0,979 (0,962 a 0,989) | 97,7 % | 0,988 |
| Foto casi monocroma | saturación < 0,08 | 15 | 1,000 (0,796 a 1,000) | 100,0 % | 1,000 |
| Foto casi monocroma | resto | 532 | 0,981 (0,966 a 0,990) | 97,9 % | 0,987 |
| Iluminación | oscura (brillo < 50) | 24 | 1,000 (0,862 a 1,000) | 95,8 % | 1,000 |
| Iluminación | sobreexpuesta (> 200) | 4 | 1,000 (0,510 a 1,000) | 100,0 % | 1,000 |
| Iluminación | normal | 519 | 0,981 (0,965 a 0,990) | 98,1 % | 0,987 |
| Degradación | desenfoque gaussiano r = 2 | 547 | 0,974 | 96,0 % | 0,977 |
| Degradación | JPEG calidad 20 | 547 | 0,982 | 97,3 % | 0,988 |
| Degradación | subexposición × 0,5 | 547 | 0,985 | 98,4 % | 0,987 |
| Degradación | resolución × 1/4 | 547 | 0,957 | 93,4 % | 0,954 |

Cruces interseccionales mínimos: diente de león casi monocromo (n = 11, exactitud 1,000); resolución nativa baja con reducción a 1/4 (n = 66, exactitud 0,955); confusiones rosa y tulipán en la matriz de confusión (2 casos). Con n de decenas, los intervalos son anchos; para exactitudes se reporta también el intervalo de Wilson.

## 4. Métricas

Primaria: macro-F1 con IC 95 % por bootstrap que remuestrea autores, porque el autor es la unidad agrupada de la partición. Secundarias: recall por clase, exactitud balanceada, ECE con 15 contenedores antes y después de la temperatura, cobertura a precisión ≥ 99 %, latencia P95 y tamaño en MB. Aceptar una clasificación errónea cuesta más que abstenerse, porque el estudiante registraría un dato falso (Mitchell et al., 2019, sección 4.4.1); por eso se fija la precisión y se reporta la cobertura, junto con la curva cobertura-riesgo completa y el punto de operación marcado. La variación se resume con el IC por autor en prueba, la media y la desviación de la CV agrupada k = 5 y la dispersión entre las tres semillas.

## 5. Datos de evaluación

Prueba: 547 fotos de autores disjuntos de entrenamiento y validación y sin casi duplicados compartidos (`outputs/particion.csv`), única partición nunca usada para seleccionar. CV agrupada: los mismos 5 pliegues de E03 sobre entrenamiento y validación, con calibración anidada. Degradaciones: definidas en `src/flores/ajuste_fino.py` con los parámetros de la tabla anterior.

## 6. Datos de entrenamiento

tf_flowers (TensorFlow Team, 2019): 2 564 fotos de entrenamiento y 548 de validación, de las 3 659 que quedan tras las exclusiones. Procedencia, licencias, distribución por clase y limitaciones en `docs/datasheet_tf_flowers.md`.

## 7. Análisis cuantitativos

| Medida | E03 (línea base) | Modelo final |
|---|---|---|
| Macro-F1 en prueba (IC 95 % por autor) | 0,987 (0,971 a 0,997) | 0,981 (0,964 a 0,993) |
| Macro-F1 en prueba, media ± DE entre semillas | no aplica | 0,983 ± 0,003 |
| Macro-F1, CV agrupada k = 5 | 0,961 ± 0,009 | 0,967 ± 0,008 |
| ECE en prueba (tras temperatura) | 0,009 | 0,013 |
| Cobertura a precisión ≥ 99 % (prueba) | 96,0 % (precisión 99,8 %) | 98,0 % (precisión 99,3 %) |
| Latencia P95 por imagen, CPU de 2 hilos | 140,5 ms | 78,1 ms |
| Prueba de McNemar frente a E03 | no aplica | p = 0,453 |

La prueba resultó más fácil que todos los pliegues de la CV en la línea base; por eso la selección se hizo con la CV y la prueba se usa una sola vez.

## 8. Consideraciones éticas

- **Datos sensibles:** algunas fotos muestran personas o viviendas de forma incidental; los nombres de usuario de Flickr se conservan solo como atribución exigida por CC-BY 2.0 y como variable de agrupamiento.
- **Solapamiento con el preentrenamiento:** LVD-142M se armó por recuperación desde conjuntos públicos (Oquab et al., 2024) y pudo incluir fotos parecidas a las de Flickr; las cifras con datos limpios son una cota optimista.
- **Riesgos, destinatario y magnitud:** un estudiante que acepta una clasificación errónea de una planta fuera de las cinco clases (probabilidad media, porque el modelo no detecta especies nuevas; daño bajo mientras el uso sea de bitácora, alto si se extrapola a comestibilidad o toxicidad); servicio desigual si la abstención rechaza sobre todo fotos de baja resolución o casi monocromas, típicas de celulares de gama baja (probabilidad por confirmar con la cobertura desagregada; daño bajo, pero inequitativo); exposición de terceros si se suben fotos con personas sin su consentimiento.
- **Mitigaciones:** abstención calibrada, revisión docente obligatoria, lista explícita de usos fuera de alcance, consentimiento de acudientes para fotos de campo (Ley 1581 de 2012) y almacenamiento sin rostros ni metadatos EXIF o GPS.
- **Casos delicados:** uso fuera del aula, en salidas de campo sin docente o con fotos de terceros.

## 9. Advertencias y recomendaciones

El modelo se entrenó con flora templada y cultivada y no representa la flora del Caribe colombiano; debe usarse como sugerencia que revisa el docente, nunca como identificación definitiva (National Institute of Standards and Technology [NIST], 2023). Pruebas adicionales sugeridas: evaluación con fotos locales del Cesar, una clase de rechazo con otras flores para medir la detección fuera de distribución y medición de latencia en el equipo objetivo. Grupos no representados: flora tropical, fotos de celulares de gama baja y luz cenital intensa. Un conjunto de evaluación ideal tendría muestreo por sitio y fecha, etiquetas verificadas por un botánico y metadatos de dispositivo. La utilidad de esta ficha depende de la integridad de quienes la escriben (Mitchell et al., 2019, sección 6) y no sustituye una auditoría externa.

## Referencias

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. En *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321-1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

Ley 1581 de 2012. Por la cual se dictan disposiciones generales para la protección de datos personales. 17 de octubre de 2012. *Diario Oficial* No. 48.587.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220-229). ACM. https://doi.org/10.1145/3287560.3287596

National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., ... Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2304.07193

TensorFlow Team. (2019). *Flowers* [Conjunto de datos]. TensorFlow Datasets. https://www.tensorflow.org/datasets/catalog/tf_flowers
