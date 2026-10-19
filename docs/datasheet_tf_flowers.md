# Hoja de datos (datasheet) de tf_flowers para el proyecto integrador

Formato: Gebru et al. (2021), siete secciones. El equipo es consumidor del conjunto, no su creador. Cuando un dato no consta en la documentación de origen se escribe "desconocido para el equipo consumidor"; los autores del formato recomiendan responder el mayor número posible de preguntas "rather than skipping" (Gebru et al., 2021, sección 3), de modo que la ausencia de información quede documentada y no oculta.

Autores de la hoja: Leonar Socarrás Molina, PhD, y John Jairo Serrano Cifuentes, Esp. Fecha: 18 de octubre de 2026. Versión de la capa derivada: v1.0, congelada en el tag `v1.0.0` del repositorio (https://github.com/leonarsomo/flores-proyecto-integrador).

## 1. Motivación

- Propósito declarado: archivo `flower_photos.tgz` publicado por el equipo de TensorFlow como material de tutoriales de clasificación de imágenes (TensorFlow Team, 2019).
- Tarea o vacío que atendía: ninguna tarea científica declarada. El conjunto se creó con fines didácticos; no se documenta una pregunta de investigación, un protocolo de muestreo ni un estándar de etiquetado.
- Creadores y entidad: el archivo se distribuye desde un depósito de Google Cloud Storage del proyecto TensorFlow. Las personas concretas que reunieron las fotos son desconocidas para el equipo consumidor.
- Financiación: desconocida para el equipo consumidor. Se buscó en la ficha del catálogo de TensorFlow Datasets y en el `LICENSE.txt` del archivo; ninguno de los dos la menciona.
- Consecuencia para el proyecto: el conjunto no fue diseñado como benchmark balanceado ni como muestra de una flora regional. Sirve para estudiar transferencia de representaciones (Oquab et al., 2024). Para validar un identificador botánico de campo haría falta una muestra local con su propio protocolo de recolección (Gebru et al., 2021).

## 2. Composición

| Elemento | Valor verificado en el cuaderno |
|---|---|
| Instancias | 3 670 fotografías JPEG RGB en origen, una flor o macizo por imagen como motivo principal; 3 659 tras las exclusiones de la sección 4 (cifra que usan la ficha del modelo y el informe) |
| Clases | margarita 633, diente de león 898, rosa 641, girasol 699, tulipán 799 (razón máx./mín. 1,42) |
| Resolución | mediana 320 × 240 px; lado menor entre 143 y 442 px; 397 imágenes con lado menor < 200 px |
| Etiqueta | carpeta de origen (una etiqueta por imagen); sin cajas, máscaras ni metadatos de especie |
| Partición oficial | ninguna (TFDS expone un único split `train`) |
| Integridad | SHA-256 del archivo `4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c` |
| Atribución | `LICENSE.txt` con 3 661 líneas: archivo, autor y URL de Flickr |
| Autores de Flickr | 1 531 cuentas distintas; mediana 1 foto por cuenta, máximo 63; 282 cuentas aportan fotos a más de una clase |

Muestra o población. Es una muestra de conveniencia de fotos de Flickr con licencia CC-BY. No es representativa de ninguna flora (predominan variedades ornamentales y cultivadas de clima templado), de ningún tipo de fotógrafo (usuarios de Flickr con cámara o celular y disposición a publicar con licencia abierta) ni de ninguna región. Tampoco es exhaustiva respecto de las cinco clases: "rosa" o "tulipán" agrupan cultivares que el conjunto no distingue.

Contenido de cada instancia. Un archivo JPEG sin otra información que su nombre de archivo y la carpeta que hace de etiqueta. El nombre de archivo conserva el identificador de la foto en Flickr, que enlaza con la línea correspondiente de `LICENSE.txt`.

Información faltante por instancia. No hay fecha de captura ni de carga, modelo de cámara, metadatos EXIF, coordenadas, especie, cultivar ni verificación botánica de la etiqueta. Nueve imágenes carecen además de línea de atribución. Ninguno de estos datos se reconstruyó; el equipo no consultó la API de Flickr.

Relaciones entre instancias. Gebru et al. (2021) piden declarar las relaciones explícitas entre instancias. Aquí hay dependencia por autor (misma cámara, jardín, estilo y a veces la misma planta) y por casi duplicados: 22 pares con pHash a distancia de Hamming ≤ 6 que involucran 43 imágenes, y 2 copias exactas de píxeles. Con datos agrupados, una partición que ignora los grupos sobrestima la generalización (Scikit-learn Developers, s. f.).

Particiones recomendadas. Se recomienda a terceros usar la partición publicada en `outputs/particion.csv` (o regenerarla con la semilla 2026) o, como mínimo, cualquier partición agrupada por autor y casi duplicado. La razón es empírica: una partición aleatoria dejaría en prueba un 74,3 % de fotos cuyo autor también aparece en entrenamiento. La partición del equipo reserva 3 de 20 pliegues para prueba, 3 para validación y 14 para entrenamiento, estratificados por clase.

Errores y redundancias. La misma foto (`15922772266_1167a06620.jpg`) aparece en las carpetas de rosas y de tulipanes: etiqueta contradictoria. Nueve imágenes no tienen línea de licencia. La exactitud de las etiquetas restantes no se auditó de forma sistemática; la revisión manual de errores de la Actividad 2 sirve como muestreo parcial del ruido de etiqueta.

Calidad técnica (umbrales declarados en el cuaderno): 88 imágenes casi monocromas (saturación media < 0,08; 49 de ellas dientes de león), 149 oscuras (brillo medio < 50), 32 sobreexpuestas (> 200), 21 borrosas (varianza del laplaciano < 50) y 135 con ruido alto (sigma de Immerkær > 15; Immerkær, 1996).

Autocontención y dependencia externa. El conjunto no es autocontenido. Las imágenes dependen de la URL https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz, sin DOI ni compromiso público de permanencia conocido por el equipo. La atribución depende de URLs de Flickr que pueden desaparecer si el autor borra la foto o la cuenta. El repositorio no guarda copia archivada de las imágenes; la persistencia se protege solo de forma parcial con el SHA-256 publicado, que permite verificar cualquier copia futura, y con la conservación del `LICENSE.txt` original en la capa derivada. Si la URL deja de responder, la partición sigue siendo verificable pero no reconstruible sin una copia íntegra del archivo.

Contenido confidencial. No se identificó información confidencial; las fotos ya eran públicas en Flickr.

Contenido ofensivo o perturbador. No se detectó en la auditoría visual realizada por el equipo. Esa auditoría no fue exhaustiva, pues no se revisaron las 3 670 imágenes una por una. La respuesta debe leerse como "no detectado", no como "ausente".

Personas y datos sensibles. Algunas fotos incluyen personas o entornos domésticos de forma incidental. No hay etiquetas de subpoblación ni datos sensibles declarados.

Identificabilidad. Los individuos pueden identificarse de forma indirecta: el `LICENSE.txt` contiene nombres de usuario de Flickr y URLs de cada foto, que la licencia CC-BY obliga a conservar como atribución y que llevan al perfil público del autor. Las personas que aparecen incidentalmente en las fotos podrían ser reconocibles por su rostro. El proyecto no usa los nombres de usuario como variable predictiva, sino solo como variable de agrupamiento para evitar fuga, y no publica figuras con rostros.

## 3. Proceso de recolección

- Origen: fotografías publicadas en Flickr bajo CC-BY 2.0 y reunidas por terceros.
- Mecanismo de adquisición y validación: desconocido para el equipo consumidor. No consta si las fotos se eligieron por búsqueda de texto, por etiqueta de Flickr o a mano, ni cómo se verificó la clase.
- Estrategia de muestreo: desconocida para el equipo consumidor. La distribución desigual por clase (633 a 898) y por autor (mediana 1, máximo 63) sugiere que no hubo cuotas, pero el equipo no puede confirmarlo.
- Participantes y remuneración: desconocidos para el equipo consumidor.
- Periodo de recolección y de creación de las fotos: desconocidos para el equipo consumidor. El archivo acompaña tutoriales de TensorFlow desde al menos la ficha de TFDS citada (TensorFlow Team, 2019); las fechas de carga en Flickr podrían recuperarse con su API, pero no se consultaron.
- Revisión ética: desconocida para el equipo consumidor.
- Notificación, consentimiento y revocación de las personas fotografiadas: no consta. La licencia CC-BY autoriza el reuso de la obra por decisión de su autor, pero no equivale al consentimiento de las personas que aparecen en ella, que además no tienen un mecanismo conocido para revocarlo dentro de este conjunto. Por eso la privacidad se gestiona como riesgo (National Institute of Standards and Technology [NIST], 2023).
- Análisis de impacto sobre esas personas: el equipo no tiene noticia de que exista. Su propio análisis se limita a no publicar rostros, no inferir nada sobre las personas y no usar los nombres de usuario fuera del agrupamiento.

## 4. Preprocesamiento, limpieza y etiquetado (capa del equipo)

1. Verificación del SHA-256 del archivo descargado.
2. Inventario (`src/flores/auditoria.py`): dimensiones, MD5 de bytes y de píxeles, pHash de 64 bits, saturación, brillo, contraste, varianza del laplaciano y ruido.
3. Exclusiones (11 imágenes): 9 sin licencia verificable y 2 con etiqueta contradictoria. Quedan 3 659.
4. Grupos de fuga: componentes conexas del grafo que une imágenes con el mismo autor, los mismos píxeles o pHash ≤ 6 bits. Resultado: 1 531 grupos.
5. Partición `StratifiedGroupKFold` (20 pliegues, semilla 2026): 3 pliegues prueba, 3 validación, 14 entrenamiento.

Los datos crudos no se modifican. Todo lo derivado se reconstruye con `scripts/` y `src/flores/` a partir del archivo original. El software de preprocesamiento está disponible en el mismo repositorio bajo licencia MIT (archivo `LICENSE`), y el tag `v1.0.0` congela la versión que produce las cifras del informe.

## 5. Usos

- Usos previos: los tutoriales oficiales de TensorFlow de clasificación de imágenes y de transferencia de aprendizaje usan este archivo, y TensorFlow Datasets lo expone como `tf_flowers` (TensorFlow Team, 2019). El equipo no conoce un repositorio que enlace de forma sistemática los trabajos académicos que lo emplean.
- Uso previsto en el proyecto: docencia y comparación de estrategias de transferencia para clasificación cerrada de cinco flores en fotografías de aficionados.
- Otras tareas posibles: estudios de calibración y abstención, de robustez ante degradaciones sintéticas, de detección de casi duplicados y de efecto de la partición agrupada sobre la estimación del error.
- Aspectos de la composición que condicionan usos futuros: sesgo hacia flora templada y cultivada y hacia el estilo de los fotógrafos de Flickr; concentración de dientes de león entre las fotos casi monocromas (49 de 88), que abre un posible atajo por color; dependencia por autor que exige particiones agrupadas. Mitigación: partición agrupada, evaluación desagregada por factor en la ficha del modelo y advertencia explícita de dominio.
- Usos no recomendados: decidir sobre comestibilidad, toxicidad, valor medicinal o estado de conservación de plantas; identificar especies fuera de las cinco clases; inferir algo sobre las personas o los lugares fotografiados; reidentificar autores o personas a partir de las fotos; publicar desempeños como válidos para flora tropical sin evaluación local; usar particiones aleatorias para reportar generalización. Los riesgos de uso indebido se documentan también en la ficha del modelo (Mitchell et al., 2019), `docs/ficha_modelo.md`.

## 6. Distribución

- Qué se distribuye: el repositorio no redistribuye imágenes. Publica código, inventario, grupos, lista de identificadores por partición, exclusiones y el `LICENSE.txt` original, y remite a la URL de descarga oficial con su SHA-256.
- Cómo y cuándo: por GitHub, en https://github.com/leonarsomo/flores-proyecto-integrador; la versión v1.0.0 de la capa derivada se fecha el 18 de octubre de 2026 y se liga al release del mismo nombre.
- DOI: la capa derivada no tiene DOI asignado a la fecha de esta hoja.
- Licencias: cada foto conserva su licencia CC-BY 2.0 de origen, que exige atribución por imagen (autor y URL según `LICENSE.txt`). La capa derivada del equipo (código, inventario, grupos, partición y documentación) se distribuye bajo licencia MIT; esa licencia no se extiende a las imágenes. Los pesos base de DINOv2 que usa el modelo tienen licencia Apache 2.0.
- Restricciones de terceros: la obligación de atribución de CC-BY 2.0. Cualquier figura con fotos del conjunto debe citar al autor de Flickr.
- Controles de exportación o regulatorios: no aplica, según el conocimiento del equipo.

## 7. Mantenimiento

- Quién mantiene: el equipo mantiene solo la capa derivada (inventario, grupos, partición, exclusiones). El archivo de imágenes lo mantiene su distribuidor original, sobre el que el equipo no tiene control.
- Contacto: Leonar Socarrás Molina (leonarsomo@unisabana.edu.co) y John Jairo Serrano Cifuentes (johnseci@unisabana.edu.co), o la sección de issues del repositorio.
- Fe de erratas: se registra en esta sección y en las notas del release. Erratas conocidas del origen: la foto `15922772266_1167a06620.jpg` duplicada en rosas y tulipanes, y las nueve imágenes sin línea de licencia; ambas quedan excluidas en la capa derivada.
- Política de actualización: cada cambio de la capa derivada crea una versión nueva con fecha, tag de Git y hash de `particion.csv`, registrado como parámetro en MLflow. La versión vigente es v1.0, ligada al tag `v1.0.0`.
- Versiones anteriores: permanecen accesibles por su tag en el historial de Git; no se garantiza soporte activo más allá del curso.
- Contribuciones de terceros: por issues o pull requests en el repositorio; las correcciones de etiqueta se aceptan con la evidencia que las sustente y generan una versión nueva.

## Referencias

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86-92. https://doi.org/10.1145/3458723

Immerkær, J. (1996). Fast noise variance estimation. *Computer Vision and Image Understanding, 64*(2), 300-302. https://doi.org/10.1006/cviu.1996.0060

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220-229). ACM. https://doi.org/10.1145/3287560.3287596

National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., ... Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2304.07193

Scikit-learn Developers. (s. f.). *Cross-validation: Evaluating estimator performance*. https://scikit-learn.org/stable/modules/cross_validation.html

TensorFlow Team. (2019). *Flowers* [Conjunto de datos]. TensorFlow Datasets. https://www.tensorflow.org/datasets/catalog/tf_flowers
