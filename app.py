import pickle
from pathlib import Path
import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).parent
with open(ROOT / "movie_list.pkl", "rb") as f:
    movie_list = pickle.load(f)

with open(ROOT / "similarity.pkl", "rb") as f:
    similarity = pickle.load(f)


# ================== API Key from Secrets ==================
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return None  # or a placeholder image URL
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
        movie_id = movies.iloc[i[0]].movie_id  # unchanged
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names, recommended_movie_posters

st.header('Movie Recommender System')

_m = pd.read_csv('tmdb_5000_movies.csv')
movies = _m[['id', 'title', 'overview']].copy()
movies['overview'] = movies['overview'].fillna('')
movies['movie_id'] = movies['id']

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0]); st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1]); st.image(recommended_movie_posters[1])
    with col3:
        st.text(recommended_movie_names[2]); st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3]); st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4]); st.image(recommended_movie_posters[4])

# TMDB attribution
st.markdown("This product uses the TMDB API but is not endorsed or certified by TMDB.")
