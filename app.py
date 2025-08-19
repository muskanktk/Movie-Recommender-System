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

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return None
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    data = requests.get(url, params=params, timeout=10).json()
    path = data.get("poster_path")
    return f"https://image.tmdb.org/t/p/w500/{path}" if path else None

def tmdb_watch_link(movie_id, region="US"):
    if not TMDB_API_KEY:
        return None
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
    data = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=10).json()
    return data.get("results", {}).get(region, {}).get("link")

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

st.markdown("""
    <style>
        body { background-color: #5a0f0f; }
        .stApp { background-color: #5a0f0f; }
        h1 {
            text-align: center;
            font-family: Cambria, serif;
            color: white;
        }
        .movie-card img {
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .movie-title {
            text-align: center;
            font-weight: bold;
            font-family: Cambria, serif;
            color: white;
            margin-bottom: 0.25rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎬 FIND SIMILAR MOVIES...</h1>", unsafe_allow_html=True)

_m = pd.read_csv('tmdb_5000_movies.csv')
movies = _m[['id', 'title', 'overview']].copy()
movies['overview'] = movies['overview'].fillna('')
movies['movie_id'] = movies['id']

title_to_id = dict(zip(movies['title'], movies['movie_id']))

movie_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    cols = st.columns(3)
    for idx, (name, poster) in enumerate(zip(recommended_movie_names, recommended_movie_posters)):
        with cols[idx % 3]:
            st.markdown(f"<p class='movie-title'>{name}</p>", unsafe_allow_html=True)
            if poster:
                st.image(poster, use_container_width=True, caption="")
            else:
                st.caption("Poster not available.")
            mid = int(title_to_id.get(name))
            link = tmdb_watch_link(mid)
            if link:
                st.link_button("Where to Watch (TMDB)", link)
            else:
                st.caption("Provider info not available.")
