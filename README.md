# 🍿 Movie Recommender System

Discover movies similar to your favorites using **content-based recommendations** powered by **TF-IDF** and **cosine similarity**. This application fetches movie posters and streaming availability from **TMDB**.

> **Live Demo:**

> **Browser Link:** https://movie-recommender-system-g8tw8ng3nmffmeajxwjskv.streamlit.app/

-----

## 🚀 Key Features

  * **Content-Based Recommendations**: Generates 5 movie suggestions based on a selected movie's plot overview.
  * **Movie Posters**: Displays dynamic posters fetched via the TMDB API.
  * **Streaming Availability**: Provides "Where to Watch" links for movies.
  * **Intuitive UI**: A simple Streamlit interface with a searchable dropdown menu.
  * **Fast Performance**: Uses pre-computed similarity data for near-instant recommendations.

-----

## 🛠️ Tech Stack

  * **Language:** Python
  * **Libraries:** `streamlit`, `pandas`, `scikit-learn`, `requests`

-----

## 📦 Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/Movie-Recommender-System.git
    cd Movie-Recommender-System
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Mac/Linux
    source venv/bin/activate
    # Windows
    venv\Scripts\Activate.ps1
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Sample `requirements.txt`

```
streamlit
pandas
requests
scikit-learn
```

-----

## 🔑 Configuration

To display movie posters and streaming links, you need a TMDB API key.

1.  Sign up for a free account on [The Movie Database (TMDB)](https://www.themoviedb.org/).
2.  Create a `.streamlit/secrets.toml` file in your project's root directory.
3.  Add your API key to the file:
    ```toml
    TMDB_API_KEY = "your_tmdb_api_key"
    ```

*Note: The app will still function for recommendations without this key, but posters and links will not appear.*

-----

## ▶ Run Locally

After configuration, simply run the Streamlit application:

```bash
streamlit run app.py
```

This will open the app in your web browser.

-----

## 💡 How to Use

1.  Select a movie from the searchable dropdown menu.
2.  Click the **Show Recommendation** button.
3.  The app will display five recommended movies with their posters and "Where to Watch" links.

-----

## 📂 Project Structure

```
Movie-Recommender-System/
├─ app.py                # Main Streamlit application
├─ requirements.txt      # Project dependencies
├─ README.md             # This file
├─ .streamlit/           # API key configuration
│  └─ secrets.toml
├─ movie_list.pkl        # Pre-computed movie data
├─ similarity.pkl        # Pre-computed cosine similarity matrix
└─ data/
   ├─ tmdb_5000_movies.csv
   └─ tmdb_5000_credits.csv
```

-----

## 🛣️ Roadmap

  * [ ] Incorporate genre, cast, and crew data for a richer recommendation model.
  * [ ] Implement a hybrid recommendation model.
  * [ ] Add multi-region support for "Where to Watch" links.

-----

## 🙋 FAQ

**Q: How does the recommendation engine work?**
A: It uses **TF-IDF** to vectorize movie plot summaries and then calculates **cosine similarity** between them to find the most similar movies.

**Q: Can I use a different dataset?**
A: Yes. You can use a different dataset and re-run the data processing script to generate new `.pkl` files.

-----

## 📄 License

This project is licensed under the MIT License.

-----

## 🙏 Acknowledgments

  * **The Movie Database (TMDB)** for providing the movie data and API.
