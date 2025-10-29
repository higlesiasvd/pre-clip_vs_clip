# Memoria Técnica: Análisis Comparativo Pre-CLIP y CLIP

# Link Repositorio:

## Práctica 2 - Sistemas Interactivos Inteligentes

# Estructura del Proyecto

```
.
├── Dockerfile           # Configuración del contenedor Docker
├── docker-compose.yml   # Orquestación con Docker Compose
├── Makefile            # Recetas para construir y ejecutar
├── requirements.txt    # Dependencias de Python
├── preclip.py          # Tarea 2: Pre-CLIP (ResNet + Sentence-Transformers)
├── clip_task.py        # Tarea 3: CLIP multimodal
├── compare.py          # Comparación de resultados
├── README.md           # Este archivo
├── dataset/            # Tu dataset (debes crearlo)
│   ├── metadata.json   # Metadatos con imágenes, captions, categorías
│   ├── categoria1/     # 5 imágenes de la categoría 1
│   ├── categoria2/     # 5 imágenes de la categoría 2
│   ├── categoria3/     # 5 imágenes de la categoría 3
│   └── categoria4/     # 5 imágenes de la categoría 4
└── results/            # Resultados (se genera automáticamente)
    ├── preclip.json
    ├── preclip_sim.png
    ├── clip.json
    ├── clip_sim.png
    ├── comparison.json
    └── comparison.png
```

## Requisitos

* **20 imágenes** organizadas en **4 categorías** (5 imágenes cada una)
* Cada imagen debe tener un **caption descriptivo en inglés**
* Archivo  `dataset/metadata.json` con la estructura:

```json
[
  {
    "image": "categoria1/imagen1.jpg",
    "caption": "Descripción de la imagen en inglés",
    "category": "Nombre Categoría"
  },
  ...
]
```

### Uso en Local

```bash
make local-all
```

### Uso con Docker

### Ver todos los comandos disponibles

```bash
make help
```

Salida:

```
==========================================
Práctica 2: Pre-CLIP y CLIP
==========================================

Comandos disponibles:
  make build          - Construir imagen Docker
  make run-preclip    - Ejecutar tarea Pre-CLIP
  make run-clip       - Ejecutar tarea CLIP
  make run-compare    - Comparar resultados
  make run-all        - Ejecutar todo en secuencia
  make shell          - Acceder a shell interactiva
  make clean          - Limpiar imágenes y contenedores

Requisitos:
  - Carpeta 'dataset/' con imágenes y metadata.json
  - Carpeta 'results/' se crea automáticamente
```

### Construir la imagen Docker

```bash
make build
```

### Ejecutar las tareas

#### Opción A: Ejecutar todo automáticamente (recomendado)

```bash
make run-all
```

Ejecuta Pre-CLIP → CLIP → Comparación en secuencia.

#### Opción B: Ejecutar paso a paso

```bash
# Paso 1: Pre-CLIP
make run-preclip

# Paso 2: CLIP
make run-clip

# Paso 3: Comparar
make run-compare
```

### 4. Acceder a shell interactiva (opcional)

```bash
make shell
```

Útil para debugging o exploración del contenedor.

### 5. Limpiar todo

```bash
make clean
```

## Uso sin Docker (Local)

Si prefieres ejecutar directamente en tu sistema:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar tareas
python3 preclip.py      # Tarea Pre-CLIP
python3 clip_task.py    # Tarea CLIP
python3 compare.py      # Comparación
```

## Solución de Problemas

### Error: "No se encontró metadata.json"

```bash
# Asegúrate de tener esta estructura:
dataset/
  metadata.json
  categoria1/
    imagen1.jpg
    ...
```

### Error: "Dataset cargado: 0 imágenes"

* Verifica que las rutas en  `metadata.json` sean correctas
* Las rutas deben ser relativas a la carpeta  `dataset/`

### Error al construir Docker

```bash
# Limpia todo y reconstruye
make clean
make build
```

---

## 1. Introducción

Esta práctica evalúa el rendimiento de dos aproximaciones distintas para el matching multimodal imagen-texto: un enfoque tradicional usando modelos separados (Pre-CLIP) y el modelo multimodal CLIP. El objetivo es demostrar empíricamente las ventajas del entrenamiento conjunto en espacios de embeddings compartidos frente al uso de arquitecturas independientes. A través de este análisis comparativo, se busca evidenciar cómo la evolución hacia modelos multimodales unificados ha revolucionado las capacidades de matching imagen-texto, superando las limitaciones inherentes a las aproximaciones basadas en arquitecturas desacopladas.

---

## 2. Dataset

### 2.1 Composición

El dataset creado para este estudio contiene 20 imágenes distribuidas uniformemente en 4 categorías temáticas, con 5 imágenes por categoría. Las categorías seleccionadas fueron Urban Scenes (escenas urbanas y arquitectura), Nature (paisajes naturales), World Cuisine (comidas de diferentes culturas) y Wildlife (animales en su hábitat natural). Esta distribución equilibrada permite evaluar el rendimiento de ambos modelos a través de diversos tipos de contenido visual, desde escenas estructuradas hasta elementos naturales con mayor variabilidad.

### 2.2 Características del Dataset

Todas las imágenes del dataset tienen una resolución y calidad suficiente para su procesamiento por redes neuronales convolucionales y transformers visuales. Cada imagen cuenta con una descripción textual en inglés (caption) que captura sus características visuales principales de manera concisa pero descriptiva. En la selección de las imágenes se priorizó la variabilidad intra-categoría para evaluar la capacidad de generalización de los modelos, asegurando que dentro de cada categoría existieran diferencias significativas en composición, iluminación y elementos visuales.

---

## 3. Metodología

### 3.1 Pre-CLIP: Aproximación con Modelos Separados

La aproximación Pre-CLIP implementada utiliza dos arquitecturas completamente independientes para procesar imágenes y texto. Para el procesamiento de imágenes se empleó ResNet-50, una red neuronal convolucional pre-entrenada en ImageNet que ha sido durante años el estándar de facto en visión por computador. De esta red se extraen embeddings de 2048 dimensiones desde la penúltima capa, eliminando la capa de clasificación final para obtener representaciones genéricas de las características visuales. El preprocesamiento de las imágenes sigue el protocolo estándar de ImageNet, con un resize a 256 píxeles, un crop central a 224x224 píxeles, y normalización usando la media y desviación estándar de ImageNet.

Para el procesamiento del texto se utilizó el modelo all-MiniLM-L6-v2 de la librería Sentence-Transformers. Este modelo, basado en arquitectura transformer, produce embeddings de 384 dimensiones optimizados para tareas de similitud semántica. Se seleccionó este modelo específico por su balance entre eficiencia computacional y capacidad de captura semántica, además de su tamaño reducido que facilita la reproducibilidad del experimento.

El desafío fundamental de esta aproximación surge de que ResNet-50 y all-MiniLM-L6-v2 fueron entrenados completamente por separado, en diferentes datasets y para diferentes tareas. Esto resulta en espacios de embeddings con dimensionalidades distintas (2048D vs 384D) y, más importante aún, sin ninguna relación semántica entre ellos. Para poder comparar estos embeddings se implementó una proyección lineal aleatoria, definida por una matriz W ∈ ℝ^(2048×384) cuyos elementos se inicializan según una distribución normal con media cero y desviación estándar 1/√2048. Esta proyección transforma el embedding de imagen al espacio de 384 dimensiones.

Es importante destacar que esta proyección aleatoria no es una solución óptima, sino precisamente una demostración del problema fundamental del enfoque Pre-CLIP: la necesidad de alineación post-hoc de espacios de representación no relacionados. En la práctica, métodos más sofisticados como Canonical Correlation Analysis (CCA) o linear probes entrenadas podrían mejorar parcialmente los resultados, pero no resolverían la incompatibilidad fundamental de espacios no co-entrenados.

### 3.2 CLIP: Modelo Multimodal

CLIP (Contrastive Language-Image Pre-training) representa un cambio paradigmático en el procesamiento multimodal. Para este estudio se utilizó la implementación openai/clip-vit-base-patch32 de Hugging Face, que emplea un Vision Transformer (ViT-B/32) como encoder de imágenes y un Transformer estándar como encoder de texto. La característica fundamental de CLIP es que ambos encoders proyectan sus respectivas modalidades a un espacio de embeddings compartido de 512 dimensiones.

El entrenamiento de CLIP se realizó mediante aprendizaje contrastivo sobre aproximadamente 400 millones de pares imagen-texto recolectados de internet. Durante el entrenamiento, el modelo aprende a maximizar la similitud coseno entre las representaciones de pares correctos imagen-texto mientras simultáneamente minimiza la similitud para pares incorrectos. Este proceso de optimización conjunta es lo que permite que ambas modalidades habiten el mismo espacio semántico de manera natural, eliminando la necesidad de proyecciones o alineaciones posteriores.

La ventaja fundamental de CLIP radica en que los embeddings de imagen y texto son directamente comparables mediante similitud coseno sin necesidad de transformaciones adicionales. Además, el modelo ha aprendido conceptos visuales y lingüísticos de manera conjunta, lo que le permite capturar correspondencias semánticas complejas que serían imposibles de descubrir con modelos entrenados independientemente.

### 3.3 Métricas de Evaluación

Para evaluar el rendimiento de ambas aproximaciones se definieron dos métricas principales. La primera es la precisión de matching, calculada como el porcentaje de imágenes para las cuales su caption correcto obtiene la similitud más alta al comparar contra todos los captions del dataset. Esta métrica se calcula simplemente como el número de matches correctos dividido entre el número total de imágenes, proporcionando una medida directa de la capacidad del sistema para identificar correspondencias imagen-texto.

La segunda métrica es la similitud diagonal, que considera únicamente las puntuaciones de similitud coseno para los pares correctos imagen-caption. Se calcula la media y desviación estándar de estos valores diagonales de la matriz de similitud. Esta métrica es complementaria a la precisión porque proporciona información sobre la confianza del modelo en los matches correctos, incluso cuando estos son identificados correctamente. Una similitud diagonal alta indica que el modelo no solo identifica los pares correctos, sino que lo hace con alta confianza.

---

## 4. Resultados

### 4.1 Resultados Pre-CLIP

Los resultados obtenidos con la aproximación Pre-CLIP revelan un fracaso casi total en la tarea de matching imagen-texto. La precisión de matching fue del 0.0%, lo que significa que ninguna de las 20 imágenes fue correctamente asociada con su caption correspondiente. La similitud diagonal media fue de apenas 0.001, con una desviación estándar de 0.046, valores que indican que las representaciones de imágenes y textos son prácticamente ortogonales en el espacio proyectado.

Este resultado, lejos de ser una deficiencia experimental, constituye una demostración perfecta del problema que CLIP fue diseñado para resolver. Las similitudes extremadamente bajas evidencian que la proyección lineal aleatoria no logra capturar ninguna relación semántica significativa entre las modalidades. Esto se debe a múltiples factores fundamentales en el diseño de estos modelos.

En primer lugar, ResNet-50 fue entrenado exclusivamente para clasificación de objetos en ImageNet, una tarea que no requiere ningún tipo de alineación con lenguaje natural. Las representaciones que aprende están optimizadas para discriminar entre 1000 categorías de objetos, pero no contienen información sobre cómo estos objetos se describen verbalmente. Por otro lado, el modelo de texto all-MiniLM-L6-v2 opera en un espacio semántico completamente independiente, optimizado para capturar relaciones entre oraciones pero sin ninguna conexión con información visual.

La proyección lineal sin aprendizaje supervisado no puede descubrir las correspondencias complejas y no lineales que existen entre el espacio visual y el lingüístico. Finalmente, la ausencia total de supervisión conjunta durante el entrenamiento de ambos modelos hace imposible que compartan una estructura semántica común. En conjunto, estos factores explican por qué la aproximación Pre-CLIP es fundamentalmente inadecuada para tareas de matching multimodal.

### 4.2 Resultados CLIP

En contraste absoluto con Pre-CLIP, los resultados obtenidos con CLIP demuestran capacidades excepcionales de matching imagen-texto. El modelo logró una precisión del 100%, identificando correctamente el caption correspondiente para todas las 20 imágenes del dataset. La similitud diagonal media fue de 0.307, significativamente superior a Pre-CLIP, con una desviación estándar muy baja de 0.022 que indica consistencia en la calidad del matching a través de todo el dataset.

El éxito de CLIP se fundamenta en su diseño y entrenamiento. El entrenamiento contrastivo con cientos de millones de pares imagen-texto permite al modelo aprender asociaciones visuales-lingüísticas a escala masiva. El espacio de embeddings compartido de 512 dimensiones está optimizado simultáneamente para ambas modalidades, lo que garantiza que conceptos semánticamente similares en imagen y texto estén próximos en este espacio. Las arquitecturas de los encoders, tanto el Vision Transformer para imágenes como el Transformer para texto, están específicamente diseñadas para capturar relaciones multimodales de manera efectiva. Además, la normalización implícita en el proceso de entrenamiento facilita las comparaciones directas por similitud coseno.

La baja desviación estándar en las similitudes diagonales es particularmente significativa, ya que indica que CLIP no solo identifica correctamente los matches, sino que lo hace con un nivel de confianza uniforme a través de diferentes tipos de contenido visual y textual. Este comportamiento robusto es característico de modelos bien generalizados que han capturado verdaderas relaciones semánticas en lugar de patrones superficiales.

### 4.3 Comparación Directa

La comparación cuantitativa entre ambas aproximaciones revela diferencias dramáticas. En términos de precisión, Pre-CLIP obtuvo 0.0% mientras CLIP alcanzó 100.0%, representando una mejora absoluta de 100 puntos porcentuales. Dado que Pre-CLIP no logró ningún match correcto, la mejora relativa es técnicamente infinita. En cuanto a similitud, Pre-CLIP obtuvo 0.001 versus 0.307 de CLIP, lo que representa una mejora absoluta de 0.306 o un incremento relativo de aproximadamente 30,600%. En términos de matches correctos, Pre-CLIP identificó 0 de 20 mientras CLIP identificó los 20 de 20.

La observación más significativa de esta comparación es que CLIP mejoró el matching en el 100% de las imágenes, sin excepciones. No hubo ningún caso en el que Pre-CLIP tuviera un rendimiento comparable o superior, lo que demuestra una superioridad absoluta y sistemática del enfoque multimodal sobre la aproximación basada en modelos independientes.

---

## 5. Análisis por Categorías

El análisis desagregado por categorías proporciona información adicional sobre el comportamiento de ambos modelos. En el caso de Pre-CLIP, todas las categorías mostraron similitudes cercanas a cero, con un rango entre -0.08 y +0.10. No se observaron diferencias significativas entre tipos de contenido, lo que confirma que el fracaso es sistemático y no específico de categorías particularmente complejas. Esto indica que el problema no radica en la dificultad inherente de ciertas categorías visuales, sino en la incompatibilidad fundamental de los espacios de representación.

Para CLIP, el rendimiento fue consistentemente alto en todas las categorías, aunque con ligeras variaciones. World Cuisine obtuvo la similitud media más alta con 0.32, seguida por Urban Scenes con 0.31, Nature con 0.30, y Wildlife con 0.29. Estas diferencias son mínimas (rango de apenas 0.03), lo que indica que CLIP generaliza efectivamente a través de diversos tipos de contenido visual.

Es interesante observar que World Cuisine, que típicamente involucra objetos bien definidos con composiciones relativamente estándar, obtuvo las similitudes más altas. Por otro lado, Wildlife, que presenta mayor variabilidad visual debido a los diferentes entornos, poses y especies animales, obtuvo puntuaciones ligeramente inferiores aunque aún excelentes. Urban Scenes, con su estructura arquitectónica clara, también muestra alto rendimiento. Nature, que puede incluir desde paisajes amplios hasta detalles naturales, mantiene consistencia en el nivel de las demás categorías. Estas observaciones sugieren que, aunque CLIP es robusto, existe una ligera ventaja en escenas con objetos claramente delimitados frente a escenas con mayor complejidad contextual.

---

## 6. Visualización de Resultados

Las matrices de similitud generadas proporcionan una representación visual intuitiva del comportamiento de ambos modelos. La matriz Pre-CLIP muestra valores uniformemente distribuidos cerca de cero, sin ninguna estructura diagonal visible. Los colores en el mapa de calor indican que las similitudes son efectivamente aleatorias, sin que exista preferencia del modelo por los pares correctos. Esta ausencia de estructura confirma visualmente que el modelo no ha capturado ninguna correspondencia significativa entre imágenes y textos.

En contraste, la matriz CLIP presenta una diagonal claramente dominante con valores alrededor de 0.30, significativamente superiores a los elementos off-diagonal que típicamente se sitúan entre 0.15 y 0.20. Esta estructura diagonal pronunciada indica que CLIP distingue consistentemente entre pares correctos e incorrectos. Los valores off-diagonal no son aleatorios sino sistemáticamente inferiores, lo que sugiere que el modelo comprende las diferencias semánticas entre distintos contenidos.

Las gráficas comparativas complementan este análisis. El contraste de precisión entre 0% y 100% es visualmente impactante y representa una diferencia cualitativa, no meramente cuantitativa. La gráfica de similitud por imagen muestra que CLIP mantiene valores consistentemente altos a través de todas las imágenes, mientras que Pre-CLIP oscila erráticamente alrededor de cero. La mejora es universal, sin excepciones, lo que refuerza la conclusión de superioridad absoluta del enfoque multimodal.

---

## 7. Discusión

### 7.1 Implicaciones del Fracaso Pre-CLIP

El resultado de 0% de precisión obtenido con Pre-CLIP no debe interpretarse como un fallo experimental, sino como una demostración empírica perfecta del problema que motivó el desarrollo de CLIP. Este resultado ilustra las limitaciones fundamentales de la era Pre-CLIP en el procesamiento multimodal.

Históricamente, los sistemas de visión por computador y procesamiento de lenguaje natural evolucionaron como disciplinas separadas, con arquitecturas, datasets y objetivos de entrenamiento completamente independientes. La integración de estas modalidades requería ingeniería manual de características y métodos ad-hoc de alineación que raramente capturaban las complejidades de las relaciones multimodales reales. La falta de supervisión conjunta durante el entrenamiento impedía que los modelos aprendieran correspondencias naturales entre conceptos visuales y lingüísticos.

Las proyecciones lineales simples, como la implementada en este estudio, son matemáticamente insuficientes para mapear espacios de alta dimensión con estructuras semánticas complejas y potencialmente no lineales. Métodos más sofisticados como Canonical Correlation Analysis podrían mejorar parcialmente los resultados, pero seguirían enfrentando el problema fundamental de intentar alinear post-hoc espacios que nunca fueron diseñados para ser compatibles.

### 7.2 El Avance de CLIP

CLIP representa un cambio de paradigma en el procesamiento multimodal que resuelve estos problemas de raíz. Al unificar el entrenamiento de los encoders de imagen y texto, CLIP permite que ambas modalidades aprendan conjuntamente en lugar de requerir alineación posterior. Esta co-evolución de las representaciones es fundamental para capturar verdaderas correspondencias semánticas.

El escalado masivo a 400 millones de pares imagen-texto contrasta dramáticamente con los datasets tradicionales de tamaño mucho menor. Esta escala permite al modelo encontrar patrones estadísticos robustos y aprender asociaciones visuales-lingüísticas que cubren una enorme variedad de conceptos. El aprendizaje auto-supervisado mediante entrenamiento contrastivo elimina la necesidad de anotaciones costosas, permitiendo aprovechar datos disponibles masivamente en internet.

Una de las capacidades más importantes de CLIP es su transferencia zero-shot, es decir, su habilidad para generalizar a nuevos conceptos sin necesidad de reentrenamiento o fine-tuning. Esta capacidad surge naturalmente del entrenamiento conjunto en lenguaje natural, que le permite al modelo entender nuevas descripciones textuales y asociarlas con patrones visuales aprendidos.

### 7.3 Limitaciones del Estudio

A pesar de los resultados concluyentes, este estudio presenta limitaciones que deben considerarse. El dataset de 20 imágenes, aunque suficiente para demostración educativa, es limitado para análisis estadístico robusto. Con una muestra mayor se podrían identificar patrones más sutiles sobre qué tipos de contenido son más desafiantes incluso para CLIP.

La proyección aleatoria utilizada para Pre-CLIP, aunque representativa del problema conceptual, podría mejorarse con métodos supervisados. Una comparación más completa incluiría proyecciones aprendidas mediante CCA o linear probes entrenadas, aunque anticipamos que seguirían siendo significativamente inferiores a CLIP. La comparación es inherentemente desigual dado que CLIP se entrenó con aproximadamente 400 millones de imágenes mientras ResNet solo vio 1.2 millones en ImageNet, aunque esta desigualdad es precisamente parte de lo que hace superior al enfoque CLIP.

Los captions utilizados son descripciones relativamente simples y directas. Textos más complejos con lenguaje figurativo, referencias culturales o descripciones abstractas podrían revelar limitaciones adicionales en ambos modelos. Finalmente, el estudio se enfoca en matching directo pero no explora otras capacidades importantes como la recuperación de imágenes por texto o la generación de descripciones.

### 7.4 Trabajo Futuro

Este estudio abre varias direcciones para trabajo futuro. Sería interesante implementar proyecciones aprendidas para Pre-CLIP usando un conjunto de entrenamiento, lo que permitiría cuantificar cuánto mejora un método de alineación supervisado versus la proyección aleatoria, aunque probablemente seguiría siendo inferior a CLIP.

Probar variantes de CLIP como CLIP-Large, que usa arquitecturas más grandes, o modelos alternativos, permitiría estudiar cómo el tamaño del modelo y variaciones arquitectónicas impactan el rendimiento. Finalmente, extender el análisis a tareas relacionadas como image retrieval, zero-shot classification o visual question answering proporcionaría una visión más completa de las capacidades multimodales.

---

## 8. Conclusiones

Este estudio demuestra empíricamente la superioridad fundamental del entrenamiento multimodal conjunto sobre aproximaciones basadas en modelos independientes. Pre-CLIP fracasó completamente con 0% de precisión, evidenciando la incompatibilidad fundamental de espacios de embeddings no relacionados. Ninguna imagen fue correctamente asociada a su caption, confirmando que modelos entrenados independientemente no pueden alinearse efectivamente mediante proyecciones simples.

CLIP alcanzó perfección con 100% de precisión gracias a su entrenamiento contrastivo en un espacio compartido. Todas las imágenes fueron correctamente identificadas con sus captions, demostrando la efectividad del aprendizaje multimodal conjunto. La mejora no es incremental sino cualitativa: CLIP resuelve un problema que Pre-CLIP fundamentalmente no puede abordar. La diferencia de 100 puntos porcentuales en precisión y más de 30,000% en similitud representa un salto paradigmático, no una mejora gradual.

La mejora es universal, beneficiando todas las imágenes y categorías sin excepción. No hubo ningún caso donde la aproximación tradicional fuera competitiva, validando completamente el diseño multimodal. El experimento valida el diseño de CLIP y justifica la transición de la comunidad científica hacia modelos de fundación multimodales. Los resultados extremos obtenidos (0% vs 100%) no son atípicos sino representativos del salto cualitativo que supone el entrenamiento conjunto en visión y lenguaje.

En conclusión, este trabajo demuestra que el matching multimodal efectivo requiere diseño y entrenamiento específicamente orientado a capturar correspondencias entre modalidades. Las aproximaciones basadas en composición de modelos independientes, aunque útiles históricamente, son fundamentalmente inadecuadas para estas tareas. CLIP y modelos similares representan el futuro de la inteligencia artificial multimodal.

---

## 9. Referencias Técnicas

Los modelos utilizados en este estudio incluyen ResNet-50, una arquitectura convolucional profunda ampliamente utilizada en visión por computador; Sentence-Transformers, específicamente el modelo all-MiniLM-L6-v2, optimizado para embeddings de texto semánticos; y CLIP en su variante ViT-B/32, el modelo multimodal de OpenAI basado en entrenamiento contrastivo.

La implementación se realizó utilizando PyTorch 2.1.0 como framework principal y Transformers 4.35.0 de Hugging Face para acceso a los modelos pre-entrenados. Los experimentos se ejecutaron en CPU, demostrando que los resultados son reproducibles sin necesidad de hardware especializado. El código completo está disponible en el repositorio adjunto con Dockerfile para garantizar reproducibilidad completa del entorno.

---

## Apéndice: Decisiones de Implementación

La selección de ResNet-50 para la aproximación Pre-CLIP se justifica por ser el estándar de facto en visión por computador durante la era pre-Transformers, ofreciendo un balance óptimo entre rendimiento y eficiencia computacional. Su amplio uso en la comunidad científica lo hace representativo de las capacidades disponibles en la era Pre-CLIP, permitiendo una comparación justa con el estado del arte de esa época.

Para el modelo de texto en Pre-CLIP se eligió all-MiniLM-L6-v2 por su tamaño reducido de aproximadamente 90MB, que facilita la descarga y reproducibilidad del experimento. A pesar de su tamaño compacto, ofrece rendimiento comparable a modelos más grandes en tareas de similitud semántica. Además, su entrenamiento en datos multilingües mejora la robustez del modelo, aunque para este estudio se utilizaron exclusivamente captions en inglés.

La versión CLIP-ViT-B/32 se seleccionó por ser la variante más eficiente de CLIP con aproximadamente 600MB de tamaño, suficiente para demostrar las capacidades multimodales sin requerir recursos computacionales excesivos. Esta versión es ampliamente utilizada en investigación como baseline, facilitando la comparación con otros estudios. Aunque existen versiones más grandes como CLIP-Large, la versión base es suficiente para los objetivos demostrativos de esta práctica.

Todos los experimentos utilizan seeds fijos para garantizar reproducibilidad. Específicamente, la proyección aleatoria usa numpy.random.seed(42), asegurando que los resultados Pre-CLIP sean consistentes entre ejecuciones. Los experimentos no requieren GPU y producen resultados idénticos en CPU, lo que democratiza la reproducibilidad. Finalmente, se incluye un Docker container que proporciona un entorno completamente aislado y reproducible con todas las dependencias necesarias.

---

**Autor**: Hugo Iglesias Pombo
**Fecha**: 29 de Octubre de 2025
**Asignatura**: Sistemas Interactivos Inteligentes
**Palabras**: ~2,850
