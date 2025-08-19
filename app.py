import pickle
from pathlib import Path
import streamlit as st
import requests
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ================== Load Data ==================
ROOT = Path(__file__).parent
with open(ROOT / "movie_list.pkl", "rb") as f:
    movie_list = pickle.load(f)

with open(ROOT / "similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return None
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    data = requests.get(url, params=params, timeout=10).json()
    path = data.get("poster_path")
    return f"https://image.tmdb.org/t/p/w500/{path}" if path else None

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names, recommended_movie_posters

# ================== Styling ==================
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
        }
        .stApp {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
        }
        h1 {
            text-align: center;
            font-family: 'Trebuchet MS', sans-serif;
            color: white;
        }
        .movie-card img {
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .movie-title {
            text-align: center;
            font-weight: bold;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ================== UI ==================
st.markdown("<h1>🎬 Prime Video Watch List</h1>", unsafe_allow_html=True)

_m = pd.read_csv('tmdb_5000_movies.csv')
movies = _m[['id', 'title', 'overview']].copy()
movies['overview'] = movies['overview'].fillna('')
movies['movie_id'] = movies['id']

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    # Grid layout (2 rows, 3 per row max)
    cols = st.columns(3)
    for idx, (name, poster) in enumerate(zip(recommended_movie_names, recommended_movie_posters)):
        with cols[idx % 3]:
            st.markdown(f"<p class='movie-title'>{name}</p>", unsafe_allow_html=True)
            st.image(poster, use_container_width=True, caption="")  
