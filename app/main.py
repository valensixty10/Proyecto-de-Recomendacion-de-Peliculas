from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Rutas relativas para los archivos necesarios
movies_path = 'Data/processed_data/movies_dataset.parquet'
credits_path = 'Data/processed_data/credits.parquet'
vectorizer_path = 'Data/processed_data/vectorizer.pkl'
matriz_reducida_path = 'Data/processed_data/matriz_reducida.pkl'

# Cargar los archivos y el modelo entrenado
def load_movies():
    try:
        movies = pd.read_parquet(movies_path)
        movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
        movies['release_year'] = movies['release_date'].dt.year
        return movies
    except Exception as e:
        print(f"Error al cargar los datos de películas: {e}")
        return None

def load_credits():
    try:
        return pd.read_parquet(credits_path)
    except Exception as e:
        print(f"Error al cargar los datos de créditos: {e}")
        return None

# Cargar vectorizer y matriz reducida
try:
    vectorizer = joblib.load(vectorizer_path)
    matriz_reducida = joblib.load(matriz_reducida_path)
except Exception as e:
    print(f"Error al cargar el modelo de recomendación: {e}")
    vectorizer = None
    matriz_reducida = None

movies_df = load_movies()

# Endpoints de la API
@app.get("/")
def read_root():
    return {"message": "API funcionando correctamente"}

@app.get("/recomendacion/{titulo}")
async def recomendacion(titulo: str):
    """
    Genera una lista de 5 películas recomendadas basadas en el título ingresado.
    """
    titulo = titulo.lower()
    idx = movies_df.index[movies_df['title'].str.lower() == titulo].tolist()
    
    if not idx:
        return {"Error": "Película no encontrada"}
    
    idx = idx[0]
    
    # Calcula la similitud del coseno solo para la película seleccionada
    sim_scores = cosine_similarity(matriz_reducida[idx].reshape(1, -1), matriz_reducida)
    sim_scores = list(enumerate(sim_scores[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtener los índices de las 5 películas más similares
    movie_indices = [i[0] for i in sim_scores[1:6]]
    recomendaciones = movies_df['title'].iloc[movie_indices].tolist()
    
    return {"Recomendaciones": recomendaciones}

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
