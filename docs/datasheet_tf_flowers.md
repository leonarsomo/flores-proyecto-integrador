# Hoja de datos (datasheet) de tf_flowers para el proyecto integrador

Formato: Gebru et al. (2021), siete secciones. El equipo es consumidor del conjunto, no su creador; cuando un dato no consta en la documentación de origen se escribe "desconocido para el equipo".

Autores de la hoja: Leonar Socarrás Molina, PhD, y John Jairo Serrano Cifuentes, Esp. Fecha: 18 de octubre de 2026. Versión de la capa derivada: v1.0.

## 1. Motivación

- Propósito declarado: archivo `flower_photos.tgz` publicado por el equipo de TensorFlow como material de tutoriales de clasificación de imágenes (TensorFlow Team, 2019). No se documenta una pregunta científica ni un protocolo de muestreo.
- Financiación y entidad: desconocido para el equipo más allá de su distribución por Google/TensorFlow.
- Consecuencia para el proyecto: el conjunto no fue diseñado como benchmark balanceado ni como muestra de una flora regional. Sirve para estudiar transferencia de representaciones, no para validar un identificador botánico de campo.

## 2. Composición

| Elemento | Valor verificado en el cuaderno |
|---|---|
| Instancias | 3 670 fotografías JPEG RGB, una flor o macizo por imagen como motivo principal |
| Clases | margarita 633, diente de león 898, rosa 641, girasol 699, tulipán 799 (razón máx./mín. 1,42) |
| Resolución | mediana 320 × 240 px; lado menor entre 143 y 442 px; 397 imágenes con lado menor < 200 px |
| Etiqueta | carpeta de origen (una etiqueta por imagen); sin cajas, máscaras ni metadatos de especie |
| Partición oficial | ninguna (TFDS expone un único split `train`) |
| Integridad | SHA-256 del archivo `4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c` |
| Atribución | `LICENSE.txt` con 3 661 líneas: archivo, autor y URL de Flickr |
| Autores de Flickr | 1 531 cuentas distintas; mediana 1 foto por cuenta, máximo 63; 282 cuentas aportan fotos a más de una clase |

Relaciones entre instancias. Hay dependencia por autor (misma cámara, jardín, estilo y a veces la misma planta) y por casi duplicados: 22 pares con pHash a distancia de Hamming ≤ 6 que involucran 43 imágenes, y 2 copias exactas de píxeles.

Errores y redundancias. La misma foto (`15922772266_1167a06620.jpg`) aparece en las carpetas de rosas y de tulipanes: etiqueta contradictoria. Nueve imágenes no tienen línea de licencia. No se auditó aún la exactitud de las etiquetas restantes; la revisión manual de errores de la Actividad 2 servirá como muestreo de ruido de etiqueta.

Calidad técnica (umbrales declarados en el cuaderno): 88 imágenes casi monocromas (saturación media < 0,08; 49 de ellas dientes de león), 149 oscuras (brillo medio < 50), 32 sobreexpuestas (> 200), 21 borrosas (varianza del laplaciano < 50) y 135 con ruido alto (sigma de Immerkær > 15).

Personas y datos sensibles. Algunas fotos incluyen personas o entornos domésticos de forma incidental. El archivo de licencias contiene nombres de usuario de Flickr, que la licencia CC-BY obliga a conservar como atribución. El proyecto no los usa como variable predictiva; solo como clave de agrupamiento para evitar fuga.

## 3. Proceso de recolección

Fotografías publicadas en Flickr bajo CC-BY 2.0 y reunidas por terceros. Criterio de selección, periodo y revisión ética: desconocido para el equipo. La licencia autoriza el reuso de la obra, pero no equivale al consentimiento de las personas que puedan aparecer en ella.

## 4. Preprocesamiento, limpieza y etiquetado (capa del equipo)

1. Verificación del SHA-256 del archivo descargado.
2. Inventario (`src/flores/auditoria.py`): dimensiones, MD5 de bytes y de píxeles, pHash de 64 bits, saturación, brillo, contraste, varianza del laplaciano y ruido.
3. Exclusiones (11 imágenes): 9 sin licencia verificable y 2 con etiqueta contradictoria. Quedan 3 659.
4. Grupos de fuga: componentes conexas del grafo que une imágenes con el mismo autor, los mismos píxeles o pHash ≤ 6 bits. Resultado: 1 531 grupos.
5. Partición `StratifiedGroupKFold` (20 pliegues, semilla 2026): 3 pliegues prueba, 3 validación, 14 entrenamiento.

Los datos crudos no se modifican. Todo lo derivado se reconstruye con `scripts/` a partir del archivo original.

## 5. Usos

- Uso previsto: docencia y comparación de estrategias de transferencia para clasificación cerrada de cinco flores en fotografías de aficionados.
- Usos desaconsejados: decidir sobre comestibilidad, toxicidad, valor medicinal o estado de conservación de plantas; identificar especies fuera de las cinco clases; inferir algo sobre las personas o los lugares fotografiados; publicar desempeños como válidos para flora tropical sin evaluación local.

## 6. Distribución

El repositorio no redistribuye imágenes. Publica código, lista de identificadores por partición, exclusiones y `LICENSE.txt` original, y remite a la URL de descarga oficial con su SHA-256. Cualquier figura con fotos del conjunto debe citar al autor de Flickr.

## 7. Mantenimiento

El equipo mantiene solo la capa derivada (inventario, grupos, partición, exclusiones). Cada cambio crea una versión nueva con fecha y hash de `particion.csv`, registrado como parámetro en MLflow.

## Referencias

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86-92. https://doi.org/10.1145/3458723

TensorFlow Team. (2019). *Flowers* [Conjunto de datos]. TensorFlow Datasets. https://www.tensorflow.org/datasets/catalog/tf_flowers
