from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# hay que definir las rutas relativas para los archivos .parquet y modelos
movies_path = 'Data/processed_data/movies_dataset.parquet'
credits_path = 'Data/processed_data/credits.parquet'
matriz_reducida_path = 'models/matriz_reducida.pkl'
vectorizer_path = 'models/vectorizer.pkl'

# ahora cargo los datos y el modelo una vez al iniciar la aplicación
try:
    movies = pd.read_parquet(movies_path)
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['release_year'] = movies['release_date'].dt.year
    
    matriz_reducida = joblib.load(matriz_reducida_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    print(f"Error al cargar los datos o el modelo: {e}")
    movies = None
    matriz_reducida = None
    vectorizer = None

@app.get("/")
def read_root():
    return {"message": "API funcionando correctamente"}

@app.get("/peliculas_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    """
    Obtiene la cantidad de películas estrenadas en un mes específico.

    Args:
        mes (str): El nombre del mes en español (por ejemplo, "Enero", "Febrero").

    Returns:
        dict: Un diccionario con la cantidad de películas estrenadas en el mes especificado.

    Raises:
        HTTPException: Si no se puede cargar el dataset de películas.
        dict: Retorna un error si el nombre del mes ingresado no es válido.

    Ejemplo de consulta:
        GET /peliculas_mes/Enero
    """
    
    if movies is None:
        raise HTTPException(status_code=500, detail="Error al cargar los datos de películas.")
    
    meses_espanol = {
        'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5,
        'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 
        'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
    }
    mes_numero = meses_espanol.get(mes.capitalize())
    if mes_numero is None:
        return {"error": f"El mes '{mes}' no es válido. Introduce un mes en español correctamente."}
    
    peliculas_mes = movies[movies['release_date'].dt.month == mes_numero]
    cantidad = len(peliculas_mes)
    
    return {"cantidad": f"{cantidad} películas fueron estrenadas en el mes de {mes}"}


@app.get("/peliculas_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    """
    Obtiene la cantidad de películas estrenadas en un día específico de la semana.

    Args:
        dia (str): El nombre del día de la semana en español (por ejemplo, "Lunes", "Martes").

    Returns:
        dict: Un diccionario con la cantidad de películas estrenadas en el día especificado.

    Raises:
        HTTPException: Si no se puede cargar el dataset de películas.
        dict: Retorna un error si el nombre del día ingresado no es válido.

    Ejemplo de consulta:
        GET /peliculas_dia/Lunes
    """
    
    if movies is None:
        raise HTTPException(status_code=500, detail="Error al cargar los datos de películas.")
    
    dias_semana = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    if dia.capitalize() not in dias_semana:
        return {"error": f"El día '{dia}' no es válido. Introduce un día en español correctamente."}
    
    dia_numero = dias_semana.index(dia.capitalize())
    peliculas_dia = movies[movies['release_date'].dt.dayofweek == dia_numero]
    cantidad = len(peliculas_dia)
    
    return {"cantidad": f"{cantidad} películas fueron estrenadas en el día {dia}"}


@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    """
    Obtiene el puntaje promedio de una película específica junto con su título y año de estreno.

    Args:
        titulo (str): El título de la película que se desea consultar.

    Returns:
        dict: Un diccionario con el título, año de estreno y puntaje promedio de la película.

    Raises:
        HTTPException: Si no se puede cargar el dataset de películas o si la película no se encuentra.

    Ejemplo de consulta:
        GET /score_titulo/Toy Story
    """
    
    if movies is None:
        raise HTTPException(status_code=500, detail="Error al cargar los datos de películas.")
    
    pelicula = movies[movies['title'].str.contains(titulo, case=False, na=False)]
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="No se encontró ninguna película con ese título.")
    
    titulo = str(pelicula.iloc[0]['title'])
    año_estreno = int(pelicula.iloc[0]['release_year'])
    score = float(pelicula.iloc[0]['vote_average'])
    
    return {"título": titulo, "año": año_estreno, "score": score}


@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    """
    Obtiene el número total de votos y el promedio de votos de una película específica,
    junto con su título y año de estreno, siempre que la película tenga al menos 2000 votos.

    Args:
        titulo (str): El título de la película que se desea consultar.

    Returns:
        dict: Un diccionario con el título, año de estreno, cantidad total de votos y promedio de votos
              de la película. Si la película no cumple con los 2000 votos mínimos, se retorna un mensaje de error.

    Raises:
        HTTPException: Si no se puede cargar el dataset de películas o si la película no se encuentra.

    Ejemplo de consulta:
        GET /votos_titulo/Toy Story
    """
    
    if movies is None:
        raise HTTPException(status_code=500, detail="Error al cargar los datos de películas.")
    
    pelicula = movies[movies['title'].str.contains(titulo, case=False, na=False)]
    if pelicula.empty:
        return {"error": "No se encontró ninguna película con ese título."}
    
    votos = int(pelicula.iloc[0]['vote_count'])
    if votos < 2000:
        return {"error": "La película no cumple con los 2000 votos necesarios."}
    
    titulo = str(pelicula.iloc[0]['title'])
    año_estreno = int(pelicula.iloc[0]['release_year'])
    promedio_votos = float(pelicula.iloc[0]['vote_average'])
    
    return {
        "título": titulo,
        "año": año_estreno,
        "votos_totales": votos,
        "promedio_votos": promedio_votos
    }


@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    """
    Obtiene información sobre el éxito de un actor en función del retorno total y promedio de las películas en las que ha participado,
    así como la cantidad total de películas en las que aparece en el dataset.

    Args:
        nombre_actor (str): El nombre del actor a consultar.

    Returns:
        dict: Un diccionario con el nombre del actor, el total de películas en las que ha participado,
              el retorno total acumulado y el retorno promedio de sus películas.

    Raises:
        HTTPException: Si no se pueden cargar los datos de películas o créditos, o si el actor no tiene películas registradas.

    Ejemplo de consulta:
        GET /get_actor/Tom Cruise
    """

    if movies is None:
        raise HTTPException(status_code=500, detail="Error al cargar los datos de películas.")
    
    try:
        credits = pd.read_parquet(credits_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar los créditos: {e}")
    
    # Filtra las películas en las que aparece el actor
    actor_data = credits[credits['cast_names'].str.contains(nombre_actor, case=False, na=False)]
    if actor_data.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el actor {nombre_actor}")
    
    peliculas_actor = movies[movies['id'].isin(actor_data['id'])]
    if peliculas_actor.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos de películas para el actor {nombre_actor}")
    
    # Calculo el retorno total y promedio
    retorno_total = peliculas_actor['return'].sum()
    retorno_promedio = peliculas_actor['return'].mean()
    
    return {
        "actor": nombre_actor,
        "películas_totales": len(peliculas_actor),
        "retorno_total": float(retorno_total) if not pd.isnull(retorno_total) else 0.0,
        "retorno_promedio": float(retorno_promedio) if not pd.isnull(retorno_promedio) else 0.0
    }

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    """
    Genera recomendaciones de películas similares en función del título proporcionado, utilizando un modelo de similitud
    basado en la similitud del coseno. Devuelve una lista de hasta 5 películas similares a la consultada.

    Args:
        titulo (str): El título de la película en inglés sobre la cual se quieren obtener recomendaciones.

    Returns:
        dict: Un diccionario con una lista de recomendaciones de películas similares.

    Raises:
        HTTPException: Si hay un error al cargar los datos de películas o el modelo de recomendación,
                       o si no se encuentra la película en el dataset.

    Ejemplo de consulta:
        GET /recomendacion/Toy Story
    """

    if movies is None or matriz_reducida is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Error al cargar los datos o el modelo.")
    
    # Convertimos el título a minúsculas para facilitar la búsqueda
    titulo = titulo.lower()
    # Encontramos el índice de la película en el dataset
    idx = movies.index[movies['title'].str.lower() == titulo].tolist()
    if not idx:
        return {"error": "Película no encontrada"}
    idx = idx[0]

    # calculo la similitud de coseno entre la película seleccionada y las demás
    sim_scores = cosine_similarity(matriz_reducida[idx].reshape(1, -1), matriz_reducida)
    sim_scores = list(enumerate(sim_scores[0]))
    # ordenamos las películas por similitud y selecciona las 5 más similares
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in sim_scores[1:6]]
    recomendaciones = movies['title'].iloc[movie_indices].tolist()
    return {"recomendaciones": recomendaciones}
