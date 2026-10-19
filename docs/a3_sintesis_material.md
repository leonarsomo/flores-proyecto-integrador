# Síntesis del material de estudio para la Actividad 3 (Unidad 4)

**Curso:** Técnicas Avanzadas de Modelado en IA, Maestría en Inteligencia Artificial, Universidad de La Sabana
**Actividad:** U4 A3, "Sustentación y comunicación técnica del proyecto final de IA"
**Fecha de consulta de las fuentes:** 18 de octubre de 2026

Nota de uso. Todo el contenido está parafraseado. Las frases entre comillas dobles son citas literales breves (menos de 15 palabras, en el idioma original) con su sección, para poder citarlas con localizador. Mitchell et al. (2019) y Gebru et al. (2021) se leyeron en la versión de arXiv (ar5iv), cuya numeración de secciones coincide con la publicada; si se cita con página, debe verificarse contra la versión ACM. Las páginas de GitHub Docs y MLflow no tienen paginación y se citan por título de sección.

---

## A. Ideas centrales por fuente

### A.1 Mitchell et al. (2019), fichas de modelo

**Referencia APA 7:**
Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220-229). ACM. https://doi.org/10.1145/3287560.3287596

**Ideas centrales:**

1. **Qué es una ficha.** Es un documento breve que acompaña a un modelo entrenado y reporta su evaluación bajo condiciones variadas, no un único número global; el resumen la presenta como evaluación comparativa "in a variety of conditions" (Resumen). La extensión propuesta es de una a dos páginas, "short (one to two page) records" (sección 1).
2. **Motivación.** Los autores observan que no existía una práctica estandarizada para documentar el desempeño de modelos y que los sesgos suelen aparecer después del despliegue; otras industrias, como la electrónica o la farmacéutica, ya exigen pruebas desagregadas (secciones 1 y 2).
3. **Las nueve secciones.** Detalles del modelo, uso previsto, factores, métricas, datos de evaluación, datos de entrenamiento, análisis cuantitativos, consideraciones éticas, y advertencias y recomendaciones (sección 4, figura 1). Los autores advierten que la lista no es exhaustiva y puede adaptarse al dominio (sección 4).
4. **Detalles del modelo.** Incluye quién lo desarrolló, fecha, versión, tipo de arquitectura, algoritmo de entrenamiento, parámetros y restricciones de equidad, referencia para ampliar, forma de citarlo, licencia y un contacto para comentarios (sección 4.1).
5. **Uso previsto.** Separa usos primarios, usuarios primarios y usos fuera de alcance, para que el lector no extrapole el modelo a contextos que no fueron evaluados (sección 4.2).
6. **Factores.** Distingue los factores relevantes (lo que previsiblemente altera el desempeño) de los factores de evaluación (lo que efectivamente se reporta) y pide explicar por qué difieren. Agrupa los factores en tres familias: grupos, instrumentación y entorno; los ejemplos de instrumentación y entorno son el hardware de la cámara y la iluminación baja (secciones 4.3.1 a 4.3.3). El apartado "Card Prompts" (4.3.4) concentra las preguntas guía.
7. **Métricas y umbrales.** Hay que justificar las medidas elegidas, los umbrales de decisión usados y el enfoque de variación. En formato digital proponen que "a threshold slider should ideally be available" (sección 4.4), es decir, mostrar el desempeño a lo largo de los umbrales y no en uno solo.
8. **Costos asimétricos del error.** En clasificación, la importancia relativa de falsos positivos y falsos negativos depende del contexto y de quién soporta el error; por eso la elección de métricas derivadas de la matriz de confusión es una decisión explícita (sección 4.4.1).
9. **Variación e incertidumbre.** Se pide reportar desviación estándar, varianza o intervalos de confianza, e indicar cómo se aproximaron (por ejemplo, promediando corridas o con validación cruzada); en el ejemplo de sonrisas los autores usan IC 95 % por bootstrap (secciones 4.4, 4.4.3 y 5.1).
10. **Datos de evaluación y de entrenamiento.** Deben declararse los conjuntos, la razón de su elección y el preprocesamiento; la sección de entrenamiento debería reflejar la de evaluación y, cuando no puede detallarse, al menos dar la distribución por factores (secciones 4.5 y 4.6).
11. **Desagregación.** Los análisis cuantitativos deben "broken down by the chosen factors" (sección 4.7), con resultados unitarios (un factor a la vez) e interseccionales (combinaciones de factores), mostrando la variación, por ejemplo con barras de error.
12. **Interseccionalidad.** El análisis por un solo eje puede ocultar fallas que se concentran en la combinación de dos características; la sección 2 lo ilustra con el caso jurídico DeGraffenreid, y la sección 5.2 aplica una versión interseccional del conjunto de prueba sintético.
13. **Consideraciones éticas.** Preguntas sobre uso de datos sensibles, impacto en decisiones centrales para la vida humana, estrategias de mitigación, riesgos y daños (para quién, con qué probabilidad y magnitud) y casos de uso especialmente delicados (sección 4.8).
14. **Advertencias y recomendaciones.** Reúne lo no cubierto antes: pruebas adicionales sugeridas, grupos no representados en la evaluación y las características que tendría un conjunto de evaluación ideal (sección 4.9).
15. **Límites de la propia herramienta.** La utilidad de la ficha "relies on the integrity of the creator(s) of the card itself" (sección 6), y no sustituye auditorías externas ni otros mecanismos de transparencia.

### A.2 Gebru et al. (2021), hojas de datos

**Referencia APA 7:**
Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86-92. https://doi.org/10.1145/3458723

**Ideas centrales:**

1. **Dos públicos.** La hoja busca que los creadores reflexionen sobre supuestos, riesgos e implicaciones de su conjunto, y que los consumidores tengan información suficiente para decidir y evitar usos indebidos (sección 1.1).
2. **Reproducibilidad como objetivo secundario.** Una hoja completa permite a terceros construir conjuntos alternativos con características similares (sección 1.1).
3. **No automatizable.** El proceso exige reflexión y, según los autores, "is not intended to be automated" (sección 1.1).
4. **Flujo de trabajo.** Las preguntas deben leerse antes de recolectar, preprocesar o distribuir, y responderse al cerrar cada etapa; se recomienda contestar el máximo posible, "rather than skipping" (sección 3).
5. **Motivación.** Para qué se creó el conjunto, qué tarea o vacío atendía, quién lo creó y en nombre de qué entidad, y quién lo financió (sección 3.1).
6. **Composición.** Qué representan las instancias, cuántas hay, si son muestra o población y si la muestra es representativa, qué contiene cada instancia, etiquetas, información faltante, relaciones entre instancias, particiones recomendadas, errores y redundancias, dependencia de recursos externos y su persistencia, datos confidenciales u ofensivos, y, cuando hay personas, subpoblaciones, identificabilidad y datos sensibles (sección 3.2).
7. **Recolección.** Cómo se obtuvo cada dato (observado, reportado o inferido) y cómo se validó, mecanismos de captura, estrategia de muestreo, quiénes participaron y cómo se les remuneró, periodo de recolección, revisión ética y, si hay personas, notificación, consentimiento, posibilidad de revocarlo y análisis de impacto (sección 3.3).
8. **Preprocesamiento.** Qué limpieza, etiquetado o transformación se aplicó, si se conservaron los datos crudos y si el software de preprocesamiento está disponible (sección 3.4).
9. **Usos.** Usos previos, repositorio que enlace trabajos que lo usan, otras tareas posibles, aspectos de la composición o del proceso que condicionan usos futuros, y tareas para las que no debería usarse (sección 3.5).
10. **Distribución.** A quién, cómo y cuándo se distribuye, si tiene DOI, bajo qué licencia, restricciones de propiedad intelectual de terceros y controles regulatorios (sección 3.6).
11. **Mantenimiento.** Quién lo aloja y mantiene, cómo contactarlo, fe de erratas, política de actualización, soporte a versiones antiguas y vía para contribuciones (sección 3.7).
12. **Preguntas condicionadas a personas.** Varias preguntas solo aplican si el conjunto se relaciona con personas; los autores sugieren omitirlas cuando no corresponda (sección 3), lo que en nuestro caso exige justificar por qué aplican parcialmente (fotos con personas incidentales y nombres de usuario).
13. **Conjuntos dinámicos.** Cuando el conjunto cambia, la hoja debe versionarse con él (sección 4).
14. **Límites.** La hoja no ofrece una solución completa a los sesgos sociales indeseados, y detectar sesgos puede requerir etiquetas demográficas que entran en tensión con la privacidad (sección 4).

### A.3 MLflow Tracking (complemento a la síntesis A2, sección A.2)

**Referencia APA 7:**
MLflow Project. (s. f.). *MLflow Tracking*. MLflow Documentation. Recuperado el 18 de octubre de 2026, de https://mlflow.org/docs/latest/ml/tracking/

La síntesis previa (`docs/a2_sintesis_material.md`, A.2) ya cubre runs, experimentos, parámetros, métricas, etiquetas, artefactos, datasets, runs anidados, consulta programática y almacenes. Aquí solo se añade lo que sirve para comunicar y reproducir.

1. **Registro de modelos.** Requiere un almacén con base de datos; un modelo se registra al guardarlo con el `log_model` de su sabor (por ejemplo, PyTorch) y después se gestiona por interfaz o API, con referencia del tipo `models:/<nombre>/<versión>` (sección "Model Registry" y "FAQ"). Nuestro `mlflow.db` en SQLite cumple el requisito sin servidor.
2. **Identidad del modelo.** En MLflow 3 la referencia `models:/<model_id>` reemplaza la antigua `runs:/<run_id>/<ruta>` (sección "Tracking Models"), lo que permite citar en la ficha un identificador estable del modelo final.
3. **Interfaz de comparación.** La interfaz lista runs por experimento, permite comparar entre experimentos, buscar por valor de parámetro o métrica, graficar métricas y descargar artefactos y metadatos (sección "Tracking UI"). La página no detalla la vista de comparación; conviene capturar sus gráficos para el anexo, no para las diapositivas principales.
4. **Servir la interfaz.** La página documenta `mlflow server --port 5000`; el comando `mlflow ui --backend-store-uri sqlite:///mlflow.db` que usa el README es la forma local equivalente y no aparece en esa página, por lo que debe citarse la referencia de la CLI si se menciona en el informe.
5. **Exportar resultados.** `MlflowClient.search_runs` y `mlflow.search_logged_models` filtran y ordenan con sintaxis tipo SQL (sección "Querying Runs Programmatically"). La función `mlflow.search_runs` devuelve un DataFrame de pandas según la referencia de la API de Python (no según esta página); exportarlo a `outputs/a2/runs_mlflow.csv` deja una tabla versionada en el repositorio que no depende de abrir la interfaz.
6. **Nota de run.** La etiqueta de sistema `mlflow.note.content` aparece en la sección "Notes" del run (sección "FAQ"); sirve para escribir en una línea la conclusión de cada estudio y enlazarla desde el informe.
7. **Datasets vinculados a métricas.** `mlflow.log_input()` registra metadatos del conjunto, y las métricas pueden asociarse a un dataset concreto (sección "Tracking Datasets"); registrar el hash de `particion.csv` como dataset hace verificable que todas las cifras salen de la misma partición.

### A.4 GitHub Docs: repositorio reproducible

**Referencias APA 7:**
GitHub. (s. f.-a). *About CITATION files*. GitHub Docs. Recuperado el 18 de octubre de 2026, de https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files

GitHub. (s. f.-b). *About Git Large File Storage*. GitHub Docs. Recuperado el 18 de octubre de 2026, de https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

GitHub. (s. f.-c). *About large files on GitHub*. GitHub Docs. Recuperado el 18 de octubre de 2026, de https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

GitHub. (s. f.-d). *About READMEs*. GitHub Docs. Recuperado el 18 de octubre de 2026, de https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes

GitHub. (s. f.-e). *About releases*. GitHub Docs. Recuperado el 18 de octubre de 2026, de https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases

GitHub. (s. f.-f). *Licensing a repository*. GitHub Docs. Recuperado el 18 de octubre de 2026, de https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository

(Las letras a-f siguen el orden alfabético del título, como pide APA 7 para un mismo autor sin fecha.)

**Ideas centrales:**

1. **Función del README.** Junto con la licencia, el archivo de citación, la guía de contribución y el código de conducta, fija las expectativas del proyecto; suele ser lo primero que ve el visitante y debe explicar qué hace el proyecto, por qué es útil, cómo empezar, dónde pedir ayuda y quién lo mantiene (GitHub, s. f.-d, introducción).
2. **Foco del README.** Solo lo esencial para empezar a usar y contribuir; la documentación larga va a una wiki o a `docs/` (GitHub, s. f.-d, sección "Wikis").
3. **Ubicación y tamaño.** GitHub busca el README en `.github`, luego en la raíz y luego en `docs`, y muestra el primero que encuentra; el contenido que pasa de 500 KiB se trunca (GitHub, s. f.-d).
4. **Enlaces relativos.** Para archivos internos se recomiendan enlaces relativos, que funcionan en cualquier rama y en clones locales; GitHub genera además un índice automático a partir de los encabezados (GitHub, s. f.-d, secciones "Relative links and image paths" y "Auto-generated table of contents").
5. **Releases y tags.** Un release es una iteración empaquetada del software, con notas y archivos binarios adjuntos, y siempre se basa en un tag de Git que marca un punto de la historia (GitHub, s. f.-e).
6. **Límites de los adjuntos.** Hasta 1 000 archivos por release, cada uno por debajo de 2 GiB, sin límite al tamaño total ni al ancho de banda; GitHub añade automáticamente un ZIP y un tarball del código en el punto del tag (GitHub, s. f.-e, secciones sobre activos y archivos de código fuente).
7. **Notas del release.** Se escriben a mano o se generan automáticamente con una plantilla (GitHub, s. f.-e).
8. **Umbrales de archivos grandes.** Git advierte a partir de 50 MiB; la carga por navegador admite como máximo 25 MiB; GitHub "blocks files larger than 100 MiB" y por encima de ese tamaño hay que usar Git LFS (GitHub, s. f.-c, sección "File size limits").
9. **Tamaño del repositorio.** Se recomienda mantenerlo idealmente por debajo de 1 GB y con fuerza por debajo de 5 GB (GitHub, s. f.-c, sección "Repository size limits").
10. **Borrar archivos ya confirmados.** Si el archivo grande entró en un commit no publicado, basta `git rm --cached` y `git commit --amend`; si ya está en la historia, se recomienda `git filter-repo`, con copia de seguridad previa (GitHub, s. f.-c, sección "Removing files from a repository's history").
11. **Git LFS.** Guarda en el repositorio un puntero con versión, `oid` SHA-256 y tamaño, y el archivo real en un almacén aparte; el máximo por archivo es 2 GB en los planes Free y Pro, 4 GB en Team y 5 GB en Enterprise Cloud; el administrador decide si los objetos LFS entran en los archivos ZIP del release (GitHub, s. f.-b).
12. **CITATION.cff.** Archivo de texto plano en la raíz que indica cómo citar el trabajo; al subirlo a la rama por defecto aparece el enlace "Cite this repository", que genera APA y BibTeX. Campos del ejemplo: `cff-version`, `message`, `authors` (con ORCID opcional), `title`, `version`, `doi`, `date-released`, `url` y el bloque opcional `preferred-citation` (GitHub, s. f.-a).
13. **Citar otra obra.** Con `preferred-citation` y un `type` (por ejemplo, `report` o `conference-paper`) la citación apunta a un informe o artículo en lugar del software (GitHub, s. f.-a, sección sobre recursos distintos del software).
14. **Licencia.** Sin licencia rigen los derechos de autor por defecto y nadie puede reproducir, distribuir ni crear obras derivadas, aunque el repositorio sea público y pueda bifurcarse; la práctica recomendada es un archivo `LICENSE` en la raíz, que GitHub detecta con Licensee (GitHub, s. f.-f, secciones "Choosing the right license" y "Determining the location of your license"). GitHub aclara que no presta asesoría legal (sección "Disclaimer").

### A.5 Presentación oral y figuras

**Referencias APA 7:**
Bourne, P. E. (2007). Ten simple rules for making good oral presentations. *PLoS Computational Biology, 3*(4), Artículo e77. https://doi.org/10.1371/journal.pcbi.0030077

Rougier, N. P., Droettboom, M., & Bourne, P. E. (2014). Ten simple rules for better figures. *PLoS Computational Biology, 10*(9), Artículo e1003833. https://doi.org/10.1371/journal.pcbi.1003833

**Ideas centrales (Bourne, 2007):**

1. **Hablar al público.** La charla se diseña según el conocimiento y las expectativas de la audiencia, "address the target audience" (regla 1).
2. **Menos es más.** Quien intenta demostrar todo lo que sabe satura la charla y el mensaje central "is often lost" (regla 2); una charla concisa deja tiempo a las preguntas.
3. **Hablar solo con resultados.** No prometer resultados que no estarán listos ni mostrar material preliminar sin interés (regla 3).
4. **Mensaje persistente.** La prueba es que la audiencia recuerde tres puntos una semana después (regla 4).
5. **Lógica narrativa.** Planteamiento, desarrollo y cierre que remata el mensaje principal (regla 5).
6. **Escenario propio.** La puesta en escena debe ajustarse a la personalidad del expositor, sin humor forzado (regla 6).
7. **Ensayar y cronometrar.** Ensayar ante colegas y no salirse del guion hacia temas que el público conoce mejor (regla 7).
8. **Visuales con mesura.** Del orden de una diapositiva por minuto, simples, y sin leerlas en voz alta salvo para enfatizar (regla 8).
9. **Revisar grabaciones.** Ver la propia grabación revela los hábitos que rompen las demás reglas (regla 9).
10. **Reconocimientos pertinentes.** Dar crédito a quien contribuyó, en el momento en que aparece su aporte, sin diluirlo en listas largas (regla 10).

**Ideas centrales (Rougier et al., 2014):**

11. **Mensaje antes que figura.** Decidir el mensaje antes de construir la figura y adaptarla al soporte, porque una diapositiva no se lee como un artículo (reglas 2 y 3).
12. **Leyendas obligatorias y valores por defecto revisados.** La leyenda explica cómo leer la figura, y los valores por defecto del software deben ajustarse (reglas 4 y 5).
13. **Color con propósito.** Elegir mapas de color adecuados y apto para daltonismo (regla 6).
14. **No engañar.** Vigilar escalas automáticas y codificaciones por área; usar "the simplest type of plots that can convey your message" (regla 7).
15. **Sin adornos.** Siguiendo a Tufte, se eliminan los elementos decorativos que no aportan información (regla 8), y la legibilidad prevalece sobre la estética (regla 9).

---

## B. Plantilla de ficha del modelo adaptada al proyecto

Se toma como base `docs/ficha_modelo_E03.md` y se reorganiza para el modelo que gane la A2 (candidatos: E03, LP-FT, LoRA o EfficientNet-B0). En cada sección se indica qué debe ir, qué ya existe y qué falta (marcado **[FALTA]**).

### B.1 Detalles del modelo (Mitchell et al., 2019, sección 4.1)
- Desarrolladores, institución y curso (ya existe).
- **[FALTA]** Fecha del modelo, número de versión (por ejemplo, `v1.0.0`, el mismo del tag del release), identificador MLflow (`models:/<nombre>/<versión>` o `model_id`) y SHA-256 del archivo de pesos.
- Arquitectura: base (DINOv2 ViT-S/14 o EfficientNet-B0), qué bloques se ajustaron, rango de LoRA si aplica, parámetros entrenables y totales.
- Algoritmo de entrenamiento: optimizador, tasa, calendario, épocas, aumentación, regularización, semillas y criterio de selección del checkpoint.
- Postprocesamiento: escalado de temperatura (valor de T) y umbral de abstención.
- **[FALTA]** Licencia del código y de los pesos, y nota sobre las licencias heredadas (pesos de DINOv2 bajo Apache 2.0; para los pesos de EfficientNet-B0 de timm debe verificarse la licencia de su fuente).
- **[FALTA]** Forma de citar (remitir a `CITATION.cff`) y contacto para comentarios (correo o issues del repositorio).

### B.2 Uso previsto (sección 4.2)
- Uso primario, usuarios y usos fuera de alcance: ya están bien formulados en la ficha E03.
- **[FALTA]** Declarar el entorno de despliegue que justifica la métrica de latencia (CPU de portátil escolar sin GPU) y que el sistema opera con abstención: la salida es "clase sugerida" o "no sé, revisar".

### B.3 Factores (sección 4.3)
Factores relevantes y de evaluación para nuestro caso, organizados según las tres familias del artículo:

| Familia (Mitchell) | Factor del proyecto | Operacionalización | Justificación |
|---|---|---|---|
| Grupos | Clase | 5 clases | El error puede concentrarse en pares confusos (rosa y tulipán). |
| Grupos | Autor de Flickr | Autores con 1 foto frente a autores con varias; autores que fotografían varias clases | Unidad de agrupamiento de la partición; el estilo del autor es la principal fuente de dependencia. |
| Instrumentación | Resolución nativa | Lado menor < 200 px frente a ≥ 200 px | 397 imágenes del conjunto bajo 200 px; cámaras y compresión de Flickr. |
| Instrumentación | Degradación sintética | Desenfoque, JPEG 20, subexposición, resolución 1/4 | Simulan cámara de baja calidad y transmisión (Hendrycks y Dietterich, 2019). |
| Entorno | Foto casi monocroma | Saturación media < 0,08 | 88 imágenes, 49 de diente de león: posible atajo por color. |
| Entorno | Iluminación | Oscuras (brillo < 50) y sobreexpuestas (> 200) | 149 y 32 imágenes. |

- **[FALTA]** Explicar por qué los factores de evaluación difieren de los relevantes: la flora local, el tipo de celular de los estudiantes y la iluminación tropical son relevantes pero no evaluables con tf_flowers.

### B.4 Métricas (sección 4.4)
- Primaria: macro-F1. Secundarias: recall por clase, exactitud balanceada, ECE (15 contenedores) antes y después de la temperatura, cobertura a precisión ≥ 99 %, riesgo selectivo, latencia P95 en CPU, tamaño del modelo en MB.
- Umbral de decisión: el umbral de abstención fijado en validación. Siguiendo la idea del deslizador de umbral (sección 4.4), **[FALTA]** incluir la curva cobertura-riesgo completa con el punto de operación marcado.
- Costos asimétricos (sección 4.4.1): aceptar una clasificación errónea es más costoso que abstenerse, porque el estudiante registra un dato falso; por eso se fija la precisión y se reporta la cobertura.
- Variación: IC 95 % por bootstrap que remuestrea autores en prueba; media y desviación de la CV agrupada k = 5; dispersión entre semillas en los modelos ajustados.

### B.5 Datos de evaluación (sección 4.5)
- Prueba: 547 fotos, autores disjuntos de entrenamiento y validación. **[FALTA]** Distribución de la prueba por factor (cuántas casi monocromas, cuántas de baja resolución, por clase), porque los IC por subgrupo dependen de esos n.
- Versiones degradadas de la prueba: parámetros exactos de cada corrupción.
- Motivación de la elección: es la única partición nunca usada para seleccionar.

### B.6 Datos de entrenamiento (sección 4.6)
- 2 564 entrenamiento y 548 validación; remitir a la hoja de datos.
- **[FALTA]** Tabla de distribución por clase y por factor en entrenamiento, espejo de B.5, como pide el artículo.

### B.7 Análisis cuantitativos (sección 4.7)
- Resultados unitarios: tabla con una fila por nivel de cada factor (clase, grupo de autor, resolución, monocroma, iluminación, cada degradación) con n, macro-F1 o exactitud, IC 95 % y cobertura bajo abstención.
- Resultados interseccionales mínimos: clase × monocroma (en especial diente de león monocromo), clase × degradación, resolución nativa baja × resolución 1/4.
- Comparación con E03 en las mismas filas, para mostrar qué compró el ajuste fino.
- **[FALTA]** Todo B.7 con el modelo final. La ficha E03 solo reporta cifras globales más una degradación. Con subgrupos pequeños (por ejemplo, decenas de monocromas en prueba) el IC por bootstrap será ancho; conviene reportar también intervalos de Wilson para exactitudes y declarar n en cada celda. Un punto delicado es la cobertura desagregada: si la abstención rechaza sobre todo fotos monocromas o de baja resolución, el 99 % global oculta un servicio desigual por subgrupo.

### B.8 Consideraciones éticas (sección 4.8)
- Datos sensibles: personas incidentales y nombres de usuario (ya existe).
- **[FALTA]** Riesgos y daños con destinatario, probabilidad y magnitud (por ejemplo, estudiante que confía en una clasificación errónea de una planta no incluida: probabilidad media, daño bajo mientras el uso sea de bitácora; alto si se extrapola a comestibilidad).
- **[FALTA]** Mitigaciones adoptadas: abstención calibrada, revisión docente, lista explícita de usos fuera de alcance.
- **[FALTA]** Casos de uso delicados: aplicación por fuera del aula, uso con fotos de terceros sin consentimiento.
- Solapamiento con el preentrenamiento de DINOv2 (ya existe; mantenerlo, también para EfficientNet preentrenada en ImageNet).

### B.9 Advertencias y recomendaciones (sección 4.9)
- Dominio templado y cultivado (ya existe).
- **[FALTA]** Pruebas adicionales sugeridas: evaluación con fotos locales del Cesar, clases de rechazo (otras flores) para medir detección fuera de distribución, prueba en el celular objetivo.
- **[FALTA]** Grupos no representados: flora tropical, fotos de celulares de gama baja, condiciones de luz cenital intensa.
- **[FALTA]** Características de un conjunto de evaluación ideal: muestreo por sitio y fecha, etiqueta verificada por botánico, metadatos de dispositivo.

### B.10 Datos que faltan por producir (resumen)
1. Tabla de factores unitarios con n, métrica, IC 95 % y cobertura bajo abstención, para el modelo final y para E03.
2. Al menos tres cruces interseccionales con n.
3. Curva cobertura-riesgo y diagrama de fiabilidad antes y después de la temperatura.
4. Matriz de confusión del modelo final en prueba limpia y en la peor degradación.
5. Distribución de factores en entrenamiento, validación y prueba.
6. Latencia P95 y tamaño en MB del modelo final en el mismo hardware que E03.
7. Versión, fecha, licencia, hash de pesos, identificador MLflow y contacto.

---

## C. Lista de verificación de la hoja de datos (Gebru et al., 2021) frente a `docs/datasheet_tf_flowers.md`

La hoja actual es sólida en composición, relaciones entre instancias, calidad técnica y preprocesamiento, y adopta con acierto la posición de consumidor. Las preguntas siguientes no están respondidas o lo están solo de forma implícita.

**Motivación (3.1)**
- [ ] Vacío o tarea específica para la que se creó: se dice que es material de tutoriales, pero no se declara explícitamente que no hubo tarea científica (basta una línea).
- [ ] Financiación: marcada como desconocida; correcto, pero falta indicar dónde se buscó.

**Composición (3.2)**
- [ ] ¿Muestra o población, y es representativa? Falta afirmar que es una muestra de conveniencia de Flickr no representativa de ninguna flora ni de ningún tipo de fotógrafo.
- [ ] Información faltante por instancia: no se mencionan metadatos ausentes (fecha, cámara, EXIF, especie o cultivar).
- [ ] Particiones recomendadas y su justificación: se describe la nuestra, pero no se dice explícitamente que se recomienda a terceros y por qué.
- [ ] Autocontención y dependencia externa: las fotos dependen de una URL de Google Storage y la atribución de URLs de Flickr que pueden desaparecer; falta indicar persistencia y si existe copia archivada.
- [ ] Contenido ofensivo o perturbador: no se responde (probablemente "no detectado", debe decirse con el método de revisión).
- [ ] Identificabilidad de individuos: los nombres de usuario de Flickr los identifican indirectamente; la hoja lo menciona pero no lo responde como pregunta.

**Recolección (3.3)**
- [ ] Mecanismo de adquisición y su validación (cómo se eligieron las fotos, por búsqueda, por etiqueta).
- [ ] Estrategia de muestreo.
- [ ] Quiénes participaron y cómo se les remuneró.
- [ ] Periodo de recolección frente a periodo de creación de las fotos (fechas de carga en Flickr).
- [ ] Notificación, consentimiento y revocación para las personas que aparecen incidentalmente; análisis de impacto sobre ellas.
- Nota: varias respuestas serán "desconocido para el equipo"; Gebru et al. (2021, sección 3) recomiendan responder igual, de modo que la ausencia quede documentada.

**Preprocesamiento (3.4)**
- [ ] Disponibilidad y licencia del software de preprocesamiento: los scripts existen, pero falta remitir a la licencia del repositorio y al tag del release que los congela.

**Usos (3.5)**
- [ ] Usos previos del conjunto (tutoriales de TensorFlow, trabajos académicos) y repositorio que los enlace.
- [ ] Otras tareas posibles.
- [ ] Aspectos de composición que condicionan usos futuros (sesgo hacia flora templada y fotógrafos de Flickr; atajo por color en dientes de león) y cómo mitigarlos.

**Distribución (3.6)**
- [ ] Licencia de la capa derivada (inventario, grupos, partición): no se indica.
- [ ] DOI de la capa derivada (por ejemplo, vía archivo del release).
- [ ] Fecha de distribución.
- [ ] Restricciones de terceros: falta citar la licencia de TFDS y precisar que CC-BY 2.0 exige atribución por imagen.
- [ ] Controles de exportación o regulatorios: responder "no aplica" de forma explícita.

**Mantenimiento (3.7)**
- [ ] Contacto del mantenedor.
- [ ] Fe de erratas (por ejemplo, la foto duplicada en rosas y tulipanes es una errata del origen).
- [ ] Política de actualización y comunicación de cambios.
- [ ] Soporte a versiones anteriores de la partición.
- [ ] Mecanismo para contribuciones de terceros.

Observación menor de coherencia: la hoja reporta 3 670 instancias en composición y 3 659 tras exclusiones; conviene que la ficha del modelo y el informe usen la segunda cifra y remitan a la primera solo como origen.

---

## D. Lista de verificación de repositorio reproducible

Estado actual inspeccionado: hay `README.md`, `requirements.txt` con versiones exactas, `.gitignore` que excluye datos, cachés y pesos, y semillas documentadas. No hay `LICENSE` ni `CITATION.cff`. Los pesos de la A2 (`outputs/a2/*.pt`) pesan entre 16 y 28 MB; las cachés de `outputs/cache/` llegan a 722 MB por archivo; el directorio de trabajo suma unos 4 GB.

**README (GitHub, s. f.-d)**
- [ ] Qué hace el proyecto y por qué es útil (existe).
- [ ] Cómo empezar: instalación, descarga verificada por SHA-256 y comandos de reproducción (existe; actualizar con los pasos de la A2 y la descarga de pesos del release).
- [ ] Resultados finales de la A2 con IC y enlace a la ficha del modelo.
- [ ] Dónde pedir ayuda y quién mantiene (añadir contacto o issues).
- [ ] Enlaces relativos a `docs/`, `configs/` y `outputs/`.
- [ ] Mantenerlo breve; lo extenso va en `docs/`. Tamaño muy por debajo de 500 KiB.
- [ ] Sustituir los rangos con guion en la tabla de resultados por "a" para coherencia con el estilo del informe.

**Licencia (GitHub, s. f.-f)**
- [ ] Añadir `LICENSE` en la raíz (MIT o Apache-2.0 para el código). Sin él, rigen los derechos de autor por defecto y nadie puede reutilizar el código legalmente.
- [ ] Declarar en el README que la licencia del código no se extiende a las imágenes (CC-BY 2.0 por foto) ni a los pesos base (Apache 2.0 de DINOv2; verificar la de EfficientNet-B0).

**CITATION.cff (GitHub, s. f.-a)**
- [ ] Crear en la raíz con `cff-version: 1.2.0`, `message`, `title`, `authors` (con ORCID cuando exista; el de Leonar Socarrás es 0000-0003-0601-9151), `version` igual al tag, `date-released`, `url` del repositorio y `license`.
- [ ] Opcional: `preferred-citation` de tipo `report` que apunte al informe de la A3.
- [ ] Verificar que aparece "Cite this repository" en la rama por defecto.

**Release y tag (GitHub, s. f.-e)**
- [ ] Crear el tag `v1.0.0` sobre el commit que produce exactamente las cifras del informe.
- [ ] Publicar el release con notas: modelo ganador, métricas principales, comando de reproducción y hash de `particion.csv`.
- [ ] Adjuntar como activos: pesos del modelo final, `runs_mlflow.csv` exportado y, si se desea, una copia de `mlflow.db` (1,3 MB). Cada activo debe ser menor de 2 GiB.
- [ ] Publicar en el README el SHA-256 de cada activo.

**Pesos y archivos grandes (GitHub, s. f.-b; s. f.-c)**
- Pesos de 16 a 28 MB: quedan por debajo de la advertencia de 50 MiB y del bloqueo de 100 MiB, de modo que técnicamente podrían confirmarse con Git; sin embargo, los de 28 MB superan el límite de 25 MiB de carga por navegador (habría que subirlos por línea de comandos) y cada versión engrosaría la historia de forma permanente. Recomendación: no versionarlos en Git y adjuntar solo el checkpoint final al release.
- Cachés de 722 MB: superan el bloqueo de 100 MiB; son regenerables y deben seguir excluidas. Git LFS (2 GB por archivo en el plan gratuito) solo se justificaría si fueran insumo no regenerable, lo cual no ocurre.
- [ ] Mantener el repositorio por debajo de 1 GB.
- [ ] Antes del primer `push`, comprobar con `git count-objects -vH` y una búsqueda de archivos de más de 50 MB que nada pesado entró en la historia; si entró, limpiar con `git filter-repo`.

**Entorno y aleatoriedad**
- [ ] `requirements.txt` con versiones exactas (existe) y versión de Python declarada (existe en comentario; conviene añadir `python_requires` o una línea explícita en el README).
- [ ] Semillas documentadas por experimento, incluidas las de la A2 (LP-FT, LoRA, EfficientNet) y la de bootstrap.
- [ ] `torch.use_deterministic_algorithms(True)` donde aplique, y declarar los casos en que no fue posible.
- [ ] Prueba automatizada que verifique que el modelo publicado reproduce la macro-F1 de prueba dentro de una tolerancia.
- [ ] Registrar en MLflow el commit de Git (etiqueta de sistema automática si el código está en un repositorio) y el tag del release como etiqueta del run final.

---

## E. Diez reglas para la presentación (8 a 12 diapositivas) y la sustentación

1. **Una pregunta, una respuesta, tres mensajes.** Abrir con la pregunta experimental de la A2 y cerrar con tres conclusiones que el jurado pueda recordar; por ejemplo, cuánto aporta el ajuste fino frente a la sonda, cuánto cuesta en latencia y dónde falla. Respaldo: Bourne (2007, reglas 4 y 5).
2. **Calibrar el contenido al jurado.** El docente domina el aprendizaje profundo; sobra explicar qué es un ViT y falta explicar las decisiones propias (partición por autor, abstención). Respaldo: Bourne (2007, regla 1); Rougier et al. (2014, regla 1).
3. **Una diapositiva por minuto.** Con 10 a 12 minutos de exposición, 8 a 12 diapositivas. Estructura sugerida: problema y uso previsto; datos y partición sin fuga; línea base E03; diseño de la A2; resultado principal con IC; costo y latencia; desagregación por factores; abstención y calibración; limitaciones y ficha; repositorio y reproducción; conclusiones. Respaldo: Bourne (2007, regla 8).
4. **Una figura, un mensaje, título afirmativo.** El título de cada diapositiva enuncia la conclusión (ejemplo de formato, con cifras por confirmar en la A2: "LoRA iguala a LP-FT con muchos menos parámetros entrenables") y la figura la demuestra. Respaldo: Rougier et al. (2014, reglas 2 y 9).
5. **Siempre la incertidumbre.** Toda cifra comparativa va con su IC o su desviación entre semillas o pliegues; si los intervalos se solapan, decirlo. Respaldo: Mitchell et al. (2019, secciones 4.4 y 4.7).
6. **Mostrar dónde falla el modelo.** Incluir una diapositiva desagregada por factores (monocromas, baja resolución, degradaciones) y la cobertura por subgrupo bajo abstención. Respaldo: Mitchell et al. (2019, sección 4.7); la honestidad sobre límites anticipa las preguntas del jurado.
7. **Gráficos simples y honestos.** Barras o puntos con barras de error en lugar de tablas densas; ejes que no exageren diferencias de milésimas; sin 3D ni adornos; paleta apta para daltonismo; valores por defecto revisados. Respaldo: Rougier et al. (2014, reglas 5 a 8).
8. **No leer las diapositivas.** El texto de la diapositiva es mínimo; el discurso lo completa. Las tablas completas, las curvas de MLflow y la ficha van a diapositivas de respaldo después del cierre. Respaldo: Bourne (2007, reglas 2 y 8).
9. **Ensayar con cronómetro y preparar el banco de preguntas.** Ensayo cronometrado entre los dos integrantes, con reparto explícito de diapositivas y una grabación revisada. Preparar respuestas a preguntas previsibles: por qué la prueba es más fácil que la CV, posible solapamiento con el preentrenamiento, por qué precisión 99 %, qué pasaría con flora local. Respaldo: Bourne (2007, reglas 7 y 9).
10. **Cerrar con trazabilidad y créditos.** Última diapositiva con el enlace al repositorio, el tag del release, la ficha y la hoja de datos, y el reconocimiento de fuentes (autores de Flickr bajo CC-BY, DINOv2, docente). Respaldo: Bourne (2007, regla 10); GitHub (s. f.-a; s. f.-e).

---

## F. Citas candidatas para el informe APA 7

1. **Ficha del modelo con análisis desagregado.**
   "Siguiendo a Mitchell et al. (2019), la evaluación del modelo final se desagregó por los factores elegidos, pues los análisis cuantitativos deben presentarse 'broken down by the chosen factors' (sección 4.7), con resultados unitarios e interseccionales acompañados de su variación."
   Referencia: Mitchell et al. (2019), en A.1.

2. **Umbral de decisión.**
   "Dado que la importancia relativa de los errores depende del contexto de uso (Mitchell et al., 2019, sección 4.4.1), se fijó la precisión en 99 % sobre las fotos aceptadas y se reporta la cobertura resultante junto con la curva cobertura-riesgo completa."

3. **Hoja de datos.**
   "La documentación del conjunto sigue las siete secciones propuestas por Gebru et al. (2021), que recomiendan responder el mayor número posible de preguntas 'rather than skipping' (sección 3), incluso cuando la respuesta es que la información es desconocida para el equipo consumidor."
   Referencia: Gebru et al. (2021), en A.2.

4. **Distribución de pesos.**
   "Los pesos del modelo final se distribuyen como activo del release v1.0.0 y no dentro de la historia de Git; GitHub admite activos de hasta 2 GiB por archivo (GitHub, s. f.-e) y bloquea en el repositorio los archivos superiores a 100 MiB (GitHub, s. f.-c, sección 'File size limits')."

5. **Citación y licencia del repositorio.**
   "El repositorio incluye un archivo CITATION.cff, a partir del cual GitHub genera la citación en APA y BibTeX (GitHub, s. f.-a), y una licencia explícita, ya que sin ella rigen los derechos de autor por defecto y se impide la reutilización del código (GitHub, s. f.-f)."

6. **Trazabilidad de experimentos.**
   "Todos los runs se registraron en MLflow con un almacén SQLite; el modelo final se registró en el registro de modelos, que exige un almacén respaldado por base de datos (MLflow Project, s. f., sección 'FAQ'), y la tabla de runs se exportó al repositorio para su consulta sin servidor."
   Referencia: MLflow Project (s. f.), en A.3.

7. **Diseño de la sustentación.**
   "La presentación se limitó a una diapositiva por minuto y se organizó en torno a tres mensajes centrales, según las recomendaciones de Bourne (2007, reglas 4 y 8)."
   Referencia: Bourne (2007), en A.5.

8. **Figuras.**
   "Las figuras emplean el tipo de gráfico más simple capaz de transmitir el mensaje y omiten elementos decorativos, conforme a Rougier et al. (2014, reglas 7 y 8)."
   Referencia: Rougier et al. (2014), en A.5.

### Limitaciones de esta síntesis
- Las dos publicaciones académicas se leyeron en la versión de arXiv; las citas con página deben verificarse contra la versión ACM.
- La página de MLflow Tracking no describe en detalle la vista de comparación ni `mlflow.search_runs`; lo dicho sobre ello proviene de la referencia de la API y debe citarse como tal.
- La página "About large files on GitHub" remite el límite por archivo de los releases al máximo de Git LFS del plan; la cifra de 2 GiB proviene de "About releases".
- La guía de presentación es bibliográfica; si el docente publica una rúbrica de sustentación, esta prevalece sobre la sección E.
