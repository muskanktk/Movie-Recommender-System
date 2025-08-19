

from pathlib import Path


ROOT = Path(__file__).parent
with open(ROOT / "movie_list.pkl", "rb") as f:
    movie_list = pickle.load(f)

with open(ROOT / "similarity.pkl", "rb") as f:
    similarity = pickle.load(f)
import pickle
import streamlit as st
import requests

# NEW imports for CSV-based build
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id  # unchanged
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender System')

# ==== REPLACEMENT FOR PICKLE LOADS: build from CSV ====
# Reads tmdb_5000_movies.csv in your repo and computes similarity
_m = pd.read_csv('tmdb_5000_movies.csv')
movies = _m[['id', 'title', 'overview']].copy()
movies['overview'] = movies['overview'].fillna('')
movies['movie_id'] = movies['id']  # so your existing code using .movie_id still works

_tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
_matrix = _tfidf.fit_transform(movies['overview'])
similarity = linear_kernel(_matrix, _matrix)
# ======================================================

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)  # <-- fixed here
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
