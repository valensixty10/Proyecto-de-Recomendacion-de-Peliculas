from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Rutas relativas a los archivos .parquet
movies_path = 'Data/processed_data/movies_dataset.parquet'
credits_path = 'Data/processed_data/credits.parquet'

# Cargar el dataset de películas y el modelo de similitud una vez
def load_movies_and_model():
    try:
        movies = pd.read_parquet(movies_path)
        movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
        movies['release_year'] = movies['release_date'].dt.year

        # Crear las características combinadas para el modelo de similitud
        movies['genres'] = movies['genres'].astype(str)
        movies['title'] = movies['title'].astype(str)
        movies['overview_clean'] = movies['overview_clean'].astype(str)
        movies['combined_features'] = (
            movies['genres'].fillna('') + ' ' +
            movies['title'].fillna('') + ' ' +
            movies['overview_clean'].fillna('')
        )

        # Generar el modelo de similitud
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, dense_output=False)

        return movies, cosine_sim
    except Exception as e:
        print(f"Error al cargar los datos o el modelo de similitud: {e}")
        return None, None

movies, cosine_sim = load_movies_and_model()

# Función para cargar 'credits' solo cuando se necesite
def load_credits():
    try:
        return pd.read_parquet(credits_path)
    except Exception as e:
        print(f"Error al cargar los datos de créditos: {e}")
        return None

# Endpoints de la API
@app.get("/")
def read_root():
    return {"message": "API funcionando correctamente"}

@app.get("/peliculas_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
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
    dias_semana = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
    if dia.capitalize() not in dias_semana:
        return {"error": f"El día '{dia}' no es válido. Introduce un día en español correctamente."}
    dia_numero = dias_semana.index(dia.capitalize())
    peliculas_dia = movies[movies['release_date'].dt.dayofweek == dia_numero]
    cantidad = len(peliculas_dia)
    return {"cantidad": f"{cantidad} películas fueron estrenadas en el día {dia}"}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    pelicula = movies[movies['title'].str.contains(titulo, case=False, na=False)]
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="No se encontró ninguna película con ese título.")
    titulo = str(pelicula.iloc[0]['title'])
    año_estreno = int(pelicula.iloc[0]['release_year'])
    score = float(pelicula.iloc[0]['vote_average'])
    return {"título": titulo, "año": año_estreno, "score": score}

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
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
    credits = load_credits()
    if credits is None:
        raise HTTPException(status_code=500, detail="Error al cargar los datos de créditos.")
    actor_data = credits[credits['cast_names'].str.contains(nombre_actor, case=False, na=False)]
    if actor_data.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el actor {nombre_actor}")
    peliculas_actor = movies[movies['id'].isin(actor_data['id'])]
    retorno_total = peliculas_actor['return'].sum()
    retorno_promedio = peliculas_actor['return'].mean()
    return {
        "actor": nombre_actor,
        "películas_totales": len(peliculas_actor),
        "retorno_total": float(retorno_total) if not pd.isnull(retorno_total) else 0.0,
        "retorno_promedio": float(retorno_promedio) if not pd.isnull(retorno_promedio) else 0.0
    }

@app.get("/recomendacion/{titulo}")
def obtener_recomendaciones(titulo: str):
    if movies is None or cosine_sim is None:
        return {"error": "Error en el servidor. No se pudieron cargar los datos o el modelo de similitud."}
    idx = movies[movies['title'].str.lower() == titulo.lower()].index
    if idx.empty:
        return {"error": "Película no encontrada"}
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx].toarray().flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    recomendaciones = movies['title'].iloc[movie_indices].tolist()
    return {"recomendaciones": recomendaciones}
