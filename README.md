# IPLN 2020 (G06) - Clasificación de tweets de odio
Laboratorio del curso de Introducción al Procesamiento de Lenguaje Natural 2020, Facultad de Ingeniería (UdelaR), desarrollado por el Grupo 06.

En las siguientes secciones, se cubren los siguientes aspectos:
1. Como ejecutar la solución, que dependencias instalar y otras soluciones auxiliares para facilitar el entendimiento del problema y del modelo.
2. Las decisiones de diseño tomadas a nivel programa y a nivel modelo.
3. Los datos extraídos al realizar experimentación con el conjunto de prueba.
4. Observaciones y conclusiones del laboratorio en general.

## 2. Ejecución
A continuación una breve descripción de componentes necesarios para la ejecución de la solución

### 2.1. Dependencias
Se utilizó Python v3.8.6 como lenguaje de programación y se emplearon las siguientes bibliotecas:
- `sckit-learn v0.21.3`
- `pandas v0.25.2`
- `numpy v1.8.2`
- `pickle v4.0`
- `tokenizer v1.4.0`

### 2.2. Pasos
Los pasos para la ejecución del programa son los siguientes:

1. Instalar las dependencias anteriormente mencionadas:

`python install -r requirements.txt`

2. Ejecutar la firma del programa:

`python es_odio.py <data_path> test_file1.csv … test_fileN.csv`

Donde <data_path> tiene la ruta donde se encuentran los datos impartidos:
- Los conjuntos de entrenamiento, prueba y validación (`train.csv`, `val.csv`, `test.csv`)
- El archivo de embeddings `fasttext.es.300.txt`
- Los archivos de prueba `test_file1.csv` … `test_fileN.csv` (pueden tener salida o no).

Un ejemplo de ejecución:
`python3 es_odio.py data/ test.csv`

### 2.3. Interfaces extra
Además de la interfaz `es_odio.py` se proporcionan otras 3 interfaces con el motivo de facilitar la ejecución de las herramientas, tanto para el equipo de desarrollo como para el cuerpo docente. Las mismas son:

1. `trainer.py` - Genera un modelo con parámetros por defecto, lo entrena utilizando el conjunto de entrenamiento `train.csv` (el cual previamente vectoriza) y guarda dicho modelo como `models/model.sav`. No requiere parámetros
2. `classifier.py` - Carga el modelo guardado como `models/model.sav` y clasifica una entrada pasada por parámetro. Requiere como único parámetro el texto a clasificar (entre comillas).
3. `evaluator.py` - Explora el espacio de hiperparámetros, entrenando distintos modelos, comparando su desempeño y luego entrenando y almacenando como `models/model.sav` aquel que tenga mejor _f1_score_. Requiere como único parámetro 0 o 1 (0 en caso de explorar sólo modelos, 1 en caso de explorar configuraciones paramétricas de cada modelo).

## 3. Solución
A continuación una breve descripción de los aspectos técnicos de la solución, tanto a nivel de modelado como a nivel de implementación.

### 3.1. Vectorización
Para vectorizar los tweets se emplearon los embeddings proporcionados por el cuerpo docente. La forma de utilizar dicha técnica dista de ser la óptima, de hecho, demora aproximadamente 1 minuto en vectorizar todo el conjunto de entrenamiento ya que carga el diccionario en memoria y vectoriza tweet a tweet buscando en dicho diccionario. La vectorización de cada tweet consiste en la media de los vectores de cada una de sus palabras, distando esto de ser una representación óptima. No obstante, por falta de tiempo de desarrollo, se eligió esta forma como idea final debido a su sencillez.

Como potenciales ideas que se tuvieron en cuenta pero se dejaron de lado:
- La implementación de modelos de redes neuronales de la biblioteca `keras` en lugar de modelos de `sklearn`. Dicha biblioteca permite tratar con embeddings de formas más optimizables, cargando el archivo de embeddings como una matriz y vectorizando el conjunto de entrenamiento durante el mismo.
- La implementación de otra forma de vectorizado, modelando cada tweet como un conjunto de características que pudieran estar relacionadas al odio (como lo pueden ser la utilización de mayúsculas, el uso de insultos o palabras clave, entre otros).
- La implementación de nuevas formas de generar embeddings teniendo en cuenta el contexto de cada palabra dentro del tweet, utilizando técnicas como _BERT_ o _ELMO_.

### 3.2. Modelo
Se implementó una lista de modelos genéricos de problemas de aprendizaje automático, siendo los mismos:
- **SVM - Support Vector Machines**
- **TREE - Árboles de Decisión**
- **KNN - K Nearest Neighbors**
- **MLP - Multi Layered Perceptron**

Para los 4 modelos se variaron distintos hiperparámetros con el fin de determinar la mejor configuración paramétrica para cada uno. En el módulo `evaluate.py` pueden consultarse dichos parámetros y al correr el mismo se exploran las métricas resultantes de entrenar el modelo con cada combinación.

Se destaca que hubo que dejar de lado múltiples modelos que se quería explorar, por problemas de diseño del vectorizado así como otras dificultades con bibliotecas. Particularmente, no se pudo implementar **Naive Bayes** ni redes neuronales del tipo **RNN** o **CNN**.

También es importante mencionar que debido a la implementación utilizada de los modelos, no se pudo dar uso al conjunto de validación.

### 3.3. Arquitectura general
La solución se diseño separando el código en 3 grupos:
- Carpeta raíz, donde se encuentra el script `es_odio.py`, así como arhcivos de configuración y carpetas con conjuntos de datos y modelos.
- Carpeta `src` donde se encuentran las 3 interfaces auxiliares anteriormente mencionadas.
- Carpeta `src/modules` donde se encuentran los módulos `const.py` (archivo auxiliar de constantes), `model.py` (archivo que representa un modelo en cuestión) y `vectorizer.py` (archivo encargado de realizar la vectorización).

## 4. Experimentación
Utilizando la salida del módulo `evaluate.py` se corrieron múltiples configuraciones paramétricas para distintos modelos.

Cabe destacar que las métricas resultantes fueron muy similares para la mayoría de los mismos, pero se adjuntan las tablas de cada uno de igual forma. Dichas tablas están ordenadas de mayor a menor según el _F1 Score_ obtenido.

Las métricas son:

**SVM:**
Parámetros | Accuracy | Precision | Recall | F1 Score
--- | --- | --- | --- | ---
Kernel: Linear <br> C: 10 | 0.68 | 0.69 | 0.68 | 0.68
Kernel: Linear <br> C: 1 | 0.65 | 0.65 | 0.65 | 0.64
Kernel: RBF <br> Gamma: auto <br> C: 10 | 0.57 | 0.57 | 0.62 | 0.52
Kernel: RBF <br> Gamma: auto <br> C: 1 | 0.50 | 0.25 | 0.50 | 0.33

<br>

**Decision Tree:**
Parámetros | Accuracy | Precision | Recall | F1 Score
--- | --- | --- | --- | ---
Criterion: Entropy <br> Max Depth: 8 | 0.59 | 0.59 | 0.59 | 0.59
Criterion: Entropy <br> Max Depth: 4 | 0.58 | 0.59 | 0.60 | 0.58
Criterion: Entropy <br> Max Depth: 12 | 0.56 | 0.56 | 0.56 | 0.56
Criterion: Gini <br> Max Depth: 12 | 0.56 | 0.56 | 0.56 | 0.56
Criterion: Gini <br> Max Depth: 4 | 0.56 | 0.56 | 0.56 | 0.55
Criterion: Gini <br> Max Depth: 8 | 0.55 | 0.55 | 0.55 | 0.55

<br>

**KNN:**
Parámetros | Accuracy | Precision | Recall | F1 Score
--- | --- | --- | --- | ---
Neighbors: 3 <br> Weights: Distance <br> Metric: Manhattan | 0.62 | 0.62 | 0.62 | 0.62
Neighbors: 3 <br> Weights: Uniform <br> Metric: Manhattan | 0.62 | 0.62 | 0.61 | 0.61
Neighbors: 3 <br> Weights: Distance <br> Metric: Euclidean | 0.61 | 0.60 | 0.60 | 0.61
Neighbors: 3 <br> Weights: Uniform <br> Metric: Euclidean | 0.61 | 0.61 | 0.59 | 0.60
Neighbors: 5 <br> Weights: Distance <br> Metric: Manhattan | 0.60 | 0.61 | 0.59 | 0.60
Neighbors: 5 <br> Weights: Uniform <br> Metric: Manhattan | 0.60 | 0.61 | 0.59 | 0.60
Neighbors: 5 <br> Weights: Distance <br> Metric: Euclidean | 0.59 | 0.59 | 0.60 | 0.59
Neighbors: 5 <br> Weights: Uniform <br> Metric: Euclidean | 0.59 | 0.59 | 0.59 | 0.59

<br>

**MLP:**
Parámetros | Accuracy | Precision | Recall | F1 Score
--- | --- | --- | --- | ---
Hidden Layers: 1 x 100 <br> Activation: ReLU <br> Solver: Adam <br> Alpha: 0.05 <br> Learning Rate: Adaptive | 0.69 | 0.69 | 0.69 | 0.69
Hidden Layers: 1 x 100 <br> Activation: ReLU <br> Solver: Adam <br> Alpha: 0.05 <br> Learning Rate: Constant | 0.69 | 0.69 | 0.69 | 0.69
Hidden Layers: 2 x 100 <br> Activation: Logistic <br> Solver: Adam <br> Alpha: 0.05 <br> Learning Rate: Adaptive | 0.67 | 0.67 | 0.67 | 0.67
Hidden Layers: 2 x 100 <br> Activation: ReLU <br> Solver: Adam <br> Alpha: 0.05 <br> Learning Rate: Adaptive | 0.67 | 0.67 | 0.67 | 0.67
Hidden Layers: 2 x 100 <br> Activation: ReLU <br> Solver: Adam <br> Alpha: 0.05 <br> Learning Rate: Constant | 0.67 | 0.67 | 0.67 | 0.67
Hidden Layers: 3 x 100 <br> Activation: ReLU <br> Solver: Adam <br> Alpha: 0.05 <br> Learning Rate: Adaptive | 0.66 | 0.66 | 0.66 | 0.66
Hidden Layers: 1 x 100 <br> Activation: Logistic <br> Solver: Adam <br> Alpha: 0.05 <br> Learning Rate: Adaptive | 0.66 | 0.67 | 0.66 | 0.66
Hidden Layers: 3 x 100 <br> Activation: ReLU <br> Solver: Adam <br> Alpha: 0.05 <br> Learning Rate: Constant | 0.64 | 0.64 | 0.64 | 0.64
Hidden Layers: 1 x 100 <br> Activation: Logistic <br> Solver: SGD <br> Alpha: 0.0001 <br> Learning Rate: Constant | 0.50 | 0.51 | 0.50 | 0.39
Hidden Layers: 2 x 100 <br> Activation: ReLu <br> Solver: SGD <br> Alpha: 0.0001 <br> Learning Rate: Constant | 0.50 | 0.50 | 0.50 | 0.34
Hidden Layers: 3 x 100 <br> Activation: ReLu <br> Solver: SGD <br> Alpha: 0.0001 <br> Learning Rate: Constant | 0.50 | 0.25 | 0.50 | 0.33
Hidden Layers: 1 x 100 <br> Activation: Logistic <br> Solver: SGD <br> Alpha: 0.0001 <br> Learning Rate: Constant | 0.50 | 0.25 | 0.50 | 0.33
Hidden Layers: 2 x 100 <br> Activation: Logistic <br> Solver: SGD <br> Alpha: 0.0001 <br> Learning Rate: Constant | 0.50 | 0.25 | 0.50 | 0.33
Hidden Layers: 3 x 100 <br> Activation: Logistic <br> Solver: SGD <br> Alpha: 0.0001 <br> Learning Rate: Constant | 0.50 | 0.25 | 0.50 | 0.33
Hidden Layers: 1 x 100 <br> Activation: Logistic <br> Solver: SGD <br> Alpha: 0.05 <br> Learning Rate: Constant | 0.50 | 0.25 | 0.50 | 0.33
Hidden Layers: 2 x 100 <br> Activation: Logistic <br> Solver: SGD <br> Alpha: 0.05 <br> Learning Rate: Constant | 0.50 | 0.25 | 0.50 | 0.33
Hidden Layers: 3 x 100 <br> Activation: Logistic <br> Solver: SGD <br> Alpha: 0.05 <br> Learning Rate: Constant | 0.50 | 0.25 | 0.50 | 0.33
Hidden Layers: 3 x 100 <br> Activation: Logistic <br> Solver: SGD <br> Alpha: 0.05 <br> Learning Rate: Adaptive | 0.50 | 0.25 | 0.50 | 0.33

<br>

Tomando el mejor resultado de cada modelo, se observa la siguiente tabla:

Modelo | Accuracy | Precision | Recall | F1 Score
--- | --- | --- | --- | ---
MLP | 0.69 | 0.69 | 0.69 | 0.69
SVM | 0.68 | 0.69 | 0.68 | 0.68
KNN | 0.62 | 0.62 | 0.62 | 0.62
Tree | 0.59 | 0.59 | 0.59 | 0.59

<br>

Como apreciación general, se observa que la red neuronal MLP ofrece mejores resultados, particularmente al usar solver _adam_ con un ratio de aprendizaje adaptativo.

## 5. Observaciones y Conclusiones
Respecto a los resultados obtenidos y al trabajo en general, se observa lo siguiente:
- La manipulación de embeddings así como la exploración de otras formas de vectorización fueron un debe.
- La exploración de modelos fue bastante exhaustiva, aunque podría haberse enfocado en la utilización de modelos más actuales (como redes neuronales recurrentes o convolucionales).
- Los resultados fueron de baja calidad en general.

De dichas observaciones y como conclusiones generales, se establece que:
- La calidad de los resultados puede deberse a múltiples factores, como por ejemplo:
    - La pobre manipulación de embeddings y la dificultad en preservar el contexto.
    - No haber encontrado el modelo correcto o la configuración paramétrica necesaria.
    - Que el espacio de entrenamiento o prueba esté sesgado debido a la cantidad de "pocos" ejemplos (se ha observado que en muchas tareas de aprendizaje automático es necesario un espacio muestral más denso).
- En general, se concluye que la tarea de clasificación de texto cuenta con dos grandes pilares: la representación de la información y la utilización de un modelo apropiado para dicha representación. Para trabajos futuros puede resultar coherente repartir esfuerzos en ambas tareas de forma más equitativa

