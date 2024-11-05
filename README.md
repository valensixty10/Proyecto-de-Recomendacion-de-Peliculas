# 🎬 Movie Recommendation API 🎬

¡Bienvenidos a la **Movie Recommendation API**! Este proyecto fue creado como parte del curso de Data Science en Henry, donde desarrollamos un sistema de recomendación de películas y consultas personalizadas basadas en datos de películas. 🚀

---

## 📋 Descripción

La **Movie Recommendation API** permite obtener información detallada sobre películas y realiza recomendaciones basadas en el contenido de cada una. Se construyó un modelo de Machine Learning utilizando técnicas de procesamiento de lenguaje natural y reducción de dimensionalidad, optimizando los recursos para un rendimiento eficiente.

**Nota**: Para una mejor gestión de recursos y garantizar el despliegue en Render, se recortó el dataset a **25,000 películas** y se redujeron componentes del modelo para minimizar el consumo de memoria.

---

## 🚀 Tecnologías Utilizadas

- **Python** 🐍
- **FastAPI** ⚡ - Framework para la API
- **Scikit-learn** 📚 - Algoritmos de ML para el sistema de recomendación
- **TF-IDF Vectorizer** ✍️ - Procesamiento de texto para similitud de contenido
- **TruncatedSVD** 🧩 - Reducción de dimensionalidad
- **pandas** y **numpy** 📊 - Análisis y manipulación de datos
- **Joblib** 💾 - Guardado y carga del modelo
- **Render** 🌐 - Despliegue de la API en la nube

---

## 🛠️ Funcionalidades

- **Consultas específicas sobre películas**: permite obtener datos de puntaje promedio, cantidad de votos y cantidad de estrenos en un día o mes específico.
- **Sistema de recomendación**: sugerencias de películas similares a una película dada.
- **Análisis de actores**: permite analizar el éxito financiero de películas en las que participó un actor específico.

---

## 📂 Estructura del Proyecto

* **📁 app/** : Contiene los archivos principales para la ejecución de la API.
  * `main.py` : Archivo principal que define la API usando FastAPI, con los diferentes endpoints para consultas de películas y recomendaciones.
  * `utils.py` : Archivo con funciones auxiliares que soportan la funcionalidad de la API.
  * `__init__.py` : Marca el directorio como un módulo de Python.

* **📁 Data/** : Carpeta para almacenar los datos de las películas.
  * **processed_data/** : Contiene los archivos en formato `.parquet` resultantes del proceso ETL, listos para ser utilizados en el análisis y el modelo de recomendaciones.
  * **raw/** : Carpeta para los datos originales en caso de que necesites almacenar archivos sin procesar para referencia o futuros análisis.

* **📁 models/** : Carpeta que contiene los archivos del modelo de recomendación.
  * `matriz_reducida.pkl` : Archivo pickle que almacena la matriz TF-IDF con dimensionalidad reducida usando Truncated SVD, utilizado para recomendaciones.
  * `vectorizer.pkl` : Archivo pickle con el vectorizador TF-IDF entrenado, necesario para transformar los textos en vectores.
  * `models_test.ipynb` : Notebook de prueba para verificar el modelo y los archivos de recomendaciones en un entorno controlado.

* **📁 notebooks/** : Jupyter Notebooks que documentan y exploran diferentes aspectos del proyecto:
  * `ETL.ipynb` : Notebook que realiza el proceso de Extracción, Transformación y Carga de datos (ETL), transformando los datos originales en el formato adecuado para análisis y modelado.
  * `EDA.ipynb` : Notebook de Análisis Exploratorio de Datos (EDA), donde se examinan las variables, distribuciones y relaciones entre los datos, identificando patrones e información relevante.
  * `train_model.ipynb` : Notebook para el entrenamiento del modelo de recomendación, incluyendo la creación de la matriz TF-IDF y la reducción de dimensionalidad con Truncated SVD.

* **📁 scripts/** : Carpeta que puede contener scripts adicionales para el procesamiento de datos u otras tareas automatizadas.

* **📝 requirements.txt** : Archivo de texto con las dependencias del proyecto, incluyendo las bibliotecas necesarias para correr el proyecto en un entorno nuevo.

* **📝 Procfile** : Archivo para el despliegue en Render u otros servicios compatibles, especificando cómo iniciar la aplicación.

* **📝 LICENSE** : Archivo de licencia MIT, que define los términos y condiciones bajo los cuales se distribuye el proyecto.

---

## 📊 EDA y ETL

El análisis exploratorio y el proceso de ETL se enfocaron en:

- Limpiar los datos y detectar valores nulos.
- Convertir los datos a un formato **parquet** optimizado.
- Recortar el dataset a **25,000 películas** para mejorar el rendimiento.
- Crear columnas combinadas como `predictor`, que contiene la información clave de cada película.
- Optimizar las consultas y reducir el modelo de recomendaciones para ajustarse a los **512 MB** de límite de memoria en Render.

---

## 🚀 Instalación y Ejecución Local

### 🧰 Requisitos

* Python 3.10 o superior.
* Instalar las dependencias listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

### ⚡Ejecución Local

1. Clonar el repositorio:

```bash
git clone https://github.com/valensixty10/Proyecto-de-Recomendacion-de-Peliculas
```

2. Navegar al directorio del proyecto:

```bash
cd Proyecto-de-Recomendacion-de-Peliculas
```

3. Ejecutar la API localmente usando  **uvicorn** :

```bash
uvicorn main:app --reload
```

4. La API estará disponible en:

```bash
http://127.0.0.1:8000
```
### 🌐 Endpoints de la API

La API cuenta con los siguientes endpoints, diseñados para realizar consultas y obtener recomendaciones de películas. Cada endpoint incluye un ejemplo de uso para facilitar su comprensión.

---

1️⃣ **GET** `/recomendacion/{titulo}`
   - Devuelve una lista de películas similares a la película proporcionada, utilizando un modelo de recomendación basado en la similitud de coseno.
   - **Ejemplo de consulta**:
     ```bash
     GET /recomendacion/Toy Story
     ```
   - **Respuesta esperada**:
     ```json
     {
       "Recomendaciones": ["Toy Story 2", "Hawaiian Vacation", "Toy Story 3", "Small Fry", "Partysaurus Rex"]
     }
     ```
   - **Descripción**: Este endpoint permite obtener recomendaciones de películas similares, basado en características compartidas como el género, el título y el resumen.

---

2️⃣ **GET** `/peliculas_mes/{mes}`
   - Devuelve la cantidad de películas que fueron estrenadas en un mes específico.
   - **Ejemplo de consulta**:
     ```bash
     GET /peliculas_mes/Enero
     ```
   - **Respuesta esperada**:
     ```json
     {
       "cantidad": "120 películas fueron estrenadas en el mes de Enero"
     }
     ```
   - **Descripción**: Permite analizar la cantidad de lanzamientos de películas en un mes particular, independientemente del año. Esto es útil para observar tendencias de estrenos a lo largo del tiempo.

---

3️⃣ **GET** `/peliculas_dia/{dia}`
   - Devuelve la cantidad de películas que fueron estrenadas en un día específico de la semana.
   - **Ejemplo de consulta**:
     ```bash
     GET /peliculas_dia/Lunes
     ```
   - **Respuesta esperada**:
     ```json
     {
       "cantidad": "85 películas fueron estrenadas en el día Lunes"
     }
     ```
   - **Descripción**: Facilita el análisis de estrenos según el día de la semana, ideal para entender si ciertos días son más populares para los lanzamientos de películas.

---

4️⃣ **GET** `/score_titulo/{titulo}`
   - Devuelve el título, el año de estreno y el puntaje promedio de una película específica.
   - **Ejemplo de consulta**:
     ```bash
     GET /score_titulo/Toy Story
     ```
   - **Respuesta esperada**:
     ```json
     {
       "título": "Toy Story",
       "año": 1995,
       "score": 8.3
     }
     ```
   - **Descripción**: Proporciona información sobre el puntaje de una película específica, permitiendo evaluar su popularidad o recepción crítica.

---

5️⃣ **GET** `/votos_titulo/{titulo}`
   - Devuelve el título, la cantidad de votos y el promedio de votos de una película, siempre que tenga más de 2000 votos.
   - **Ejemplo de consulta**:
     ```bash
     GET /votos_titulo/Spider-Man
     ```
   - **Respuesta esperada**:
     ```json
     {
       "título": "Spider-Man",
       "año": 2002,
       "votos_totales": 8500,
       "promedio_votos": 7.4
     }
     ```
   - **Descripción**: Este endpoint asegura un mínimo de votos (2000) para que la información sea representativa, ideal para validar el éxito de una película basado en la participación de los usuarios.

---

6️⃣ **GET** `/get_actor/{nombre_actor}`
   - Devuelve el éxito de un actor medido a través del retorno total y promedio de las películas en las que ha participado, además de la cantidad de películas.
   - **Ejemplo de consulta**:
     ```bash
     GET /get_actor/Tom Hanks
     ```
   - **Respuesta esperada**:
     ```json
     {
       "actor": "Tom Hanks",
       "películas_totales": 35,
       "retorno_total": 1200.5,
       "retorno_promedio": 34.3
     }
     ```
   - **Descripción**: Este endpoint permite evaluar el impacto financiero de un actor, calculando el retorno total y promedio de sus películas. Es útil para medir el "poder de taquilla" de un actor en particular.

---

### 📌 Notas adicionales:
- Cada endpoint puede ser probado desde la documentación interactiva `/docs` si estás ejecutando la API localmente.
- Los ejemplos de consulta pueden adaptarse a los datos específicos que estés analizando. 
- Todos los endpoints han sido optimizados para una experiencia de consulta rápida y eficiente.

## 🚀 Despliegue en Render

El proyecto ha sido desplegado en la plataforma **Render**, lo que permite acceder a la API directamente desde el navegador o integrarla en aplicaciones. Puedes acceder a la API en el siguiente enlace:

- [🔗 API en Render](https://proyecto-de-recomendacion-de-peliculas.onrender.com)

Desde el enlace, podrás explorar todos los endpoints y probar las consultas directamente utilizando la documentación interactiva de Swagger disponible en `/docs`.

## 📞 Contacto

Si tenés alguna pregunta sobre este proyecto, o te gustaría conectarte conmigo para discutir temas relacionados con Ciencia de Datos o MLOps, ¡podés contactarme en LinkedIn!

- [Mi LinkedIn](https://www.linkedin.com/in/valentin-salgado-463332301/)

## 🎥 Video de Presentación

Como parte de la demostración del proyecto, preparé un video donde explico el funcionamiento de la API, las consultas disponibles y el modelo de recomendación de películas. También doy un vistazo al EDA y al proceso ETL, y explico cómo se desplegó la API.

- [🎬 Ver video de presentación](https://www.youtube.com/watch?v=QjHU0XjjWKY)

## 📜 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Esto significa que podés utilizar, modificar y distribuir el proyecto, siempre y cuando se mantenga la atribución original. Para más detalles, consultá el archivo de licencia:

- [LICENSE](https://github.com/valentin-salgado/proyecto-mlops/blob/main/LICENSE)

---

**¡Gracias por visitar mi proyecto!** Si te resultó útil o interesante, no dudes en dar una ⭐ en GitHub o ponerte en contacto para cualquier duda o sugerencia. 😊
