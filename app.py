# app.py
import time
import functools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from requests.exceptions import RequestException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =========================
# Page config + logging
# =========================
st.set_page_config(page_title="🍿 Movie Recommender", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender")

# =========================
# Agile flavor (versioning)
# =========================
st.sidebar.caption("Version: 0.3.0 (Sprint 6)")
if st.sidebar.button("View Sprint Notes"):
    st.sidebar.info(
        "- Added retries & caching for TMDB\n"
        "- Faster first render via cached TF-IDF\n"
        "- Better error messages\n"
        "- CI + tests + Docker"
    )

# =========================
# Constants / Paths
# =========================
ROOT = Path(__file__).parent
DATA_CSV = ROOT / "tmdb_5000_movies.csv"

# =========================
# Secrets / API Keys
# =========================
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

# =========================
# Resiliency helpers
# =========================
def retry(max_tries=3, backoff=0.6):
    """Exponential backoff retry for transient network errors."""
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            tries = 0
            while True:
                try:
                    return fn(*a, **kw)
                except RequestException as e:
                    tries += 1
                    if tries >= max_tries:
                        logger.error(f"{fn.__name__} failed after {tries} tries: {e}")
                        raise
                    sleep = backoff * (2 ** (tries - 1))
                    logger.warning(f"{fn.__name__} failed ({e}), retrying in {sleep:.1f}s…")
                    time.sleep(sleep)
        return wrap
    return deco

# =========================
# Data loading + TF-IDF
# =========================
@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Required data file not found: {DATA_CSV.name}. "
            "Place 'tmdb_5000_movies.csv' in the project root."
        )
    df = pd.read_csv(DATA_CSV)
    # Keep only needed columns; ensure no NaNs
    df = df[['id', 'title', 'overview']].copy()
    df['overview'] = df['overview'].fillna('')
    df['movie_id'] = df['id'].astype(int)
    df['title'] = df['title'].astype(str)
    return df

@st.cache_data(show_spinner=False)
def build_tfidf(df: pd.DataFrame):
    """Compute TF-IDF and cosine similarity (cached)."""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    # Linear kernel is faster than cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, sim

# =========================
# TMDB API (cached + retries)
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
@retry(max_tries=3, backoff=0.7)
def tmdb_get_json(url, params):
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def fetch_poster(movie_id: int):
    if not TMDB_API_KEY:
        return None
    data = tmdb_get_json(
        f"https://api.themoviedb.org/3/movie/{movie_id}",
        {"api_key": TMDB_API_KEY, "language": "en-US"},
    )
    path = data.get("poster_path")
    return f"https://image.tmdb.org/t/p/w500/{path}" if path else None

def tmdb_watch_link(movie_id: int, region="US"):
    if not TMDB_API_KEY:
        return None
    data = tmdb_get_json(
        f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers",
        {"api_key": TMDB_API_KEY},
    )
    return data.get("results", {}).get(region, {}).get("link")

# =========================
# Recommendation core
# =========================
def recommend(df: pd.DataFrame, sim: np.ndarray, title: str, k: int = 5):
    """Return top-k titles and poster URLs for a given movie title."""
    if title not in set(df['title']):
        raise KeyError(f"Movie '{title}' not found in dataset.")
    idx = df.index[df['title'] == title][0]
    scores = list(enumerate(sim[idx]))
    # sort descending by similarity (skip self at index 0)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:k+1]

    names, posters, ids = [], [], []
    for row_idx, _score in scores:
        mid = int(df.iloc[row_idx].movie_id)
        name = str(df.iloc[row_idx].title)
        poster = fetch_poster(mid)  # may be None if no key
        names.append(name)
        posters.append(poster)
        ids.append(mid)
    return names, posters, ids

# =========================
# UI
# =========================
st.markdown("""
    <style>
        .stApp { background-color: #0f172a; } /* slate-900 */
        h1, h2, h3, p, label, .movie-title { color: #e5e7eb; } /* slate-200 */
        .subtle { color:#94a3b8; } /* slate-400 */
        .movie-card img { border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .movie-title { text-align: center; font-weight: 700; margin-bottom: 0.25rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🍿 FIND SIMILAR MOVIES 🎬</h1>", unsafe_allow_html=True)
st.caption("Type a title from the TMDB 5000 dataset and get 5 similar films. Posters/links appear if a TMDB API key is configured.")

# Diagnostics
with st.expander("Diagnostics"):
    st.write("TMDB key present:", "✅" if TMDB_API_KEY else "❌")
    st.write("Data file:", DATA_CSV.name)

# Load data/model
try:
    movies = load_movies()
    _tfidf, _tfidf_matrix, similarity = build_tfidf(movies)
except Exception as e:
    st.error(str(e))
    st.stop()

# Controls
movie_list = movies['title'].tolist()
selected = st.selectbox("Pick a movie title", options=movie_list, index=0, placeholder="Start typing…")
go = st.button("Show Recommendations")

if go:
    t0 = time.time()
    try:
        names, posters, ids = recommend(movies, similarity, selected, k=5)
    except KeyError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Recommendation failed: {e}")
        st.stop()

    cols = st.columns(5)
    for i, (name, poster, mid) in enumerate(zip(names, posters, ids)):
        with cols[i]:
            st.markdown(f"<p class='movie-title'>{name}</p>", unsafe_allow_html=True)
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.caption("Poster not available.")
            link = tmdb_watch_link(mid)
            if link:
                st.link_button("Where to Watch (TMDB)", link, use_container_width=True)
            else:
                st.caption("Provider info not available.")
    st.caption(f"Done in {time.time() - t0:.2f}s")
