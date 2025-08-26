# 🍿 Movie Recommender System

Discover movies similar to your favorites using **content-based recommendations** powered by **TF-IDF** and **cosine similarity**. Posters and streaming availability are fetched from **TMDB**.

**Live demo:** 

**Dataset:** [TMDB 5000 Movies (Kaggle)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

**Browser:**  [https://your-streamlit-app-url.streamlit.app](#)

## 🚀 Features

* **Content-Based Recommendations**: Suggests 5 similar movies based on plot overview.
* **Movie Posters**: Dynamically pulled from TMDB API.
* **Where to Watch**: Get provider links (region-aware, defaults to US).
* **Streamlit UI**: Simple dropdown to search and select any movie.
* **Precomputed Similarities**: Fast startup with prebuilt `movie_list.pkl` and `similarity.pkl`.

## 🛠️ Tech Stack

* **Language:** Python
* **Frameworks/Libraries:** `streamlit`, `pandas`, `scikit-learn`, `requests`
* **Tools:** Git, VS Code

## 📦 Installation

```bash
# 1) Clone the repository
git clone https://github.com/<your-username>/Movie-Recommender-System.git
cd Movie-Recommender-System

# 2) Create & activate a virtual environment
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
```

**Sample `requirements.txt`:**

```
streamlit
pandas
requests
scikit-learn
```

## 🔑 Configuration

Create a `.streamlit/secrets.toml` file:

```toml
TMDB_API_KEY = "your_tmdb_api_key"
```

Without the key, recommendations still work, but posters and provider links will be missing.

---

## ▶ Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

---

## 💡 How to Use

1. **Select a Movie** from the dropdown search box.
2. Click **Show Recommendation**.
3. View:

   * 🎬 Recommended movie titles
   * 🖼️ Posters from TMDB
   * 🔗 “Where to Watch” links (if available)

## 📂 Project Structure

```
Movie-Recommender-System/
├─ app.py                 # Streamlit entry point
├─ requirements.txt       # Python dependencies
├─ README.md              # This file
├─ movie_list.pkl         # Precomputed movie list
├─ similarity.pkl         # Precomputed similarity matrix
├─ tmdb_5000_movies.csv   # Movie dataset
├─ tmdb_5000_credits.csv  # Credits dataset (optional)
└─ JupNote.ipynb          # Notebook to rebuild artifacts
```
## 🛣️ Roadmap

* [ ] Add genre, cast, and crew features for richer recommendations
* [ ] Hybrid model: combine popularity + similarity
* [ ] Support multiple regions for watch-provider links
* [ ] Add caching for API requests

## FAQ

**Q: Can I run it without a TMDB key?**
A: Yes, but posters and watch links won’t show.

**Q: How are recommendations calculated?**
A: By comparing movie plot overviews with **TF-IDF vectors** and **cosine similarity**.

**Q: Can I add more movies?**
A: Yes! Rebuild `movie_list.pkl` and `similarity.pkl` using the notebook with your dataset.

## Troubleshooting

* **KeyError on recommend()** → Ensure `movies` DataFrame is built and the title exists.
* **No posters/links** → Check your TMDB API key.
* **Slow app startup** → Use the precomputed `.pkl` files instead of recalculating similarity.

## License

MIT License – see [LICENSE](LICENSE) for details.

## Acknowledgments

* TMDB for the dataset and API
