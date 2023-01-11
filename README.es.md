<!-- hide -->
# Tutorial del proyecto Clasificador de Imágenes
<!-- endhide -->

- Escribirás un algoritmo para clasificar si las imágenes contienen un perro o un gato. Esto es fácil para humanos, perros y gatos. Tu computadora lo encontrará un poco más difícil.

> ¡No te olvides de ser siempre ingenioso!

## 🌱  Cómo iniciar este proyecto

Esta vez no se hará Fork, tómate un tiempo para leer estas instrucciones:

1. Crear un nuevo repositorio basado en el [proyecto de Machine Learing](https://github.com/4GeeksAcademy/machine-learning-python-template/generate) [haciendo clic aquí](https://github.com/4GeeksAcademy/machine-learning-python-template).
2. Abre el repositorio creado recientemente en Gitpod usando la [extensión del botón de Gitpod](https://www.gitpod.io/docs/browser-extension/).
3. Una vez que Gitpod VSCode haya terminado de abrirse, comienza tu proyecto siguiendo las instrucciones a continuación.

## 🚛 Cómo entregar este proyecto

Una vez que hayas terminado de resolver los ejercicios, asegúrate de confirmar tus cambios, hazle "push" a el fork de tu repositorio y ve a 4Geeks.com para subir el enlace del repositorio.

## 📝 Instructions

**Clasificador de imágenes**

El conjunto de datos se compone de fotos de perros y gatos proporcionadas como un subconjunto de fotos de un conjunto de datos mucho más grande de 3 millones de fotos anotadas manualmente. El conjunto de datos se desarrolló como una asociación entre Petfinder.com y Microsoft.

El conjunto de datos se usó originalmente como un CAPTCHA, es decir, una tarea que se cree que un humano encuentra trivial, pero que una máquina no puede resolver, que se usa en sitios web para distinguir entre usuarios humanos y bots. La tarea se denominó "Asirra". Cuando se presentó "Asirra", se mencionó "que los estudios de usuarios indican que los humanos pueden resolverlo el 99,6% de las veces en menos de 30 segundos". A menos que se produzca un gran avance en la visión artificial, esperamos que los ordenadores no tengan más de 1/54.000 posibilidades de resolverlo.

En el momento en que se publicó la competencia, el resultado de última generación se logró con un SVM y se describió en un artículo de 2007 con el título "Ataques de Machine Learning contra el CAPTCHA de Asirra" (PDF) que logró una precisión de clasificación del 80 %. Fue este documento el que demostró que la tarea ya no era una tarea adecuada para un CAPTCHA poco después de que se propusiera la tarea.

El conjunto de datos es fácil de entender y lo suficientemente pequeño como para caber en la memoria y comenzar con la visión artificial y las redes neuronales convolucionales.

Enlaces de conjuntos de datos:

https://www.kaggle.com/c/dogs-vs-cats/data

**Paso 1:**

Descarga la carpeta datatset y descomprime los archivos. Ahora tendrás una carpeta llamada 'tren/' que contiene 25 000 archivos .jpg de perros y gatos. Las fotos están etiquetadas por su nombre de archivo, con la palabra “perro” o “gato”.

**Paso 2:**

Importa las siguientes bibliotecas:

```py
import keras,os
from keras.models import Sequential  #ya que todas las capas del modelo se organizarán en secuencia
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator #ya que importa datos con etiquetas fácilmente al modelo. Tiene funciones para cambiar la escala, rotar, hacer zoom, etc. Esta clase altera los datos sobre la marcha mientras los pasa al modelo.
import numpy as np
```

**Paso 3:**

Carga y traza las primeras nueve fotos de perros en una sola figura. Repite lo mismo para los gatos. Puedes ver que las fotos son a color y tienen diferentes formas y tamaños.

Las fotos deberán remodelarse antes del modelado para que todas las imágenes tengan la misma forma. Esto es a menudo una pequeña imagen cuadrada. Las entradas más pequeñas significan un modelo que es más rápido de entrenar, por lo que elegiremos un tamaño fijo de 200 × 200 píxeles.

Podríamos cargar todas las imágenes, remodelarlas y almacenarlas como un solo array NumPy. Esto podría caber en la memoria RAM en muchas máquinas modernas, pero no en todas, especialmente si solo tienes 8 gigabytes para trabajar.

Podemos escribir código personalizado para cargar las imágenes en la memoria y cambiarles el tamaño como parte del proceso de carga, luego guardarlas listas para el modelado.

1. Si tienes más de 12 gigabytes de RAM, use la API de procesamiento de imágenes de Keras para cargar las 25 000 fotos en el conjunto de datos de entrenamiento y remodelarlas a fotos cuadradas de 200 × 200. La etiqueta también debe determinarse para cada foto en función de los nombres de archivo. Se debe guardar una tupla de fotos y etiquetas.

2. Si no tienes más de 12 gigabytes de RAM, carga las imágenes progresivamente usando la clase Keras ImageDataGenerator y la API flow_from_directory(). Esto será más lento de ejecutar pero se ejecutará en más máquinas. Esta API prefiere que los datos se dividan en directorios train/ y test/ separados, y debajo de cada directorio para tener un subdirectorio para cada clase.

**Paso 4:**

Create an object of ImageDataGenerator for both training and testing data and pass the folder which has train data to the object trdata and similarly pass the folder which has test data to the object tsdata. 

The ImageDataGenerator will automatically label all the data inside cat folder as cat and vis-à-vis for dog folder. In this way data is easily ready to be passed to the neural network.

**Paso 5:**

Cualquier clasificador que se ajuste a este problema tendrá que ser robusto porque algunas imágenes muestran al gato o al perro en una esquina o tal vez a 2 gatos o perros en la misma foto. VGG16 es una arquitectura de red neuronal de convolución (CNN) utilizada para ganar la competencia ILSVR (Imagenet) en 2014. Se considera una de las arquitecturas de modelos de visión excelentes hasta la fecha.

Lo más singular de VGG16 es que, en lugar de tener una gran cantidad de hiperparámetros, se enfocaron en tener capas de convolución de filtro 3x3 con un paso 1 y siempre usaron el mismo relleno y la misma capa maxpool de filtro 2x2 de paso 2. Sigue esto disposición de las capas de convolución y maxpool consistentemente a lo largo de toda la arquitectura. Al final, tiene 2 FC (capas totalmente conectadas) seguidas de un softmax para la salida. El 16 en VGG16 se refiere a que tiene 16 capas que tienen pesos. Esta red es bastante grande y tiene alrededor de 138 millones (aprox.) de parámetros.

Inicializa el modelo especificando que el modelo es un modelo secuencial. Después de inicializar el modelo, agrega:

→ 2 x capa de convolución de 64 canales de 3x3 kernel y mismo relleno.

→ 1 x capa maxpool de tamaño piscina 2x2 y zancada 2x2.

→ 2 x capa de convolución de 128 canales de 3x3 kernel y mismo relleno.

→ 1 x capa maxpool de tamaño piscina 2x2 y zancada 2x2.

→ 3 x capa de convolución de 256 canales de 3x3 kernel y mismo relleno.

→ 1 x capa maxpool de tamaño piscina 2x2 y zancada 2x2.

→ 3 x capa de convolución de 512 canales de 3x3 kernel y mismo relleno.

→ 1 x capa maxpool de tamaño piscina 2x2 y zancada 2x2.

→ 3 x capa de convolución de 512 canales de 3x3 kernel y mismo relleno.

→ 1 x capa maxpool de tamaño piscina 2x2 y zancada 2x2.

Agrega la activación de relu (Unidad lineal rectificada) a cada capa para que todos los valores negativos no pasen a la siguiente capa.

Veamos unas primeras filas para tener una idea, y seguir con todas las capas:

```py
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
```

**Paso 6:**

Después de crear toda la convolución, pasa los datos a la capa densa. Para hacer eso, primero debes aplanar el vector que sale de las circunvoluciones y luego agregar:

→ 1 x Capa densa de 4096 unidades

→ 1 x Capa densa de 4096 unidades

→ 1 x Capa Dense Softmax de 2 unidades

Usa la activación RELU para ambas capas densas para dejar de reenviar valores negativos a través de la red. Use una capa densa de 2 unidades al final con activación softmax ya que tiene 2 clases para predecir. La capa softmax generará el valor entre 0 y 1 en función de la confianza del modelo en la clase a la que pertenecen las imágenes.

**Paso 7:**

Importa el optimizador de Adam y utilízalo para compilar el modelo. Especifica una tasa de aprendizaje para ello.

**Paso 8:**

Consulta el resumen del modelo

**Paso 9:**

Importa el método ModelCheckpoint y EarlyStopping de keras. Crea un objeto de ambos y páselo como funciones de devolución de llamada a fit_generator.

**Paso 10:**

Una vez que hayas entrenado el modelo, visualiza la precisión y la pérdida del entrenamiento/validación.

**Paso 11:**

Carga el mejor modelo guardado y preprocesa la imagen, luego pasa la imagen al modelo y haz predicciones.

**Paso 12:**

Usa tu archivo app.py para crear su clasificador de imágenes.

En tu archivo README escribe un breve resumen.

Guía de soluciones: 

https://github.com/4GeeksAcademy/image-classifier-project-tutorial/blob/main/solution_guide.ipynb
