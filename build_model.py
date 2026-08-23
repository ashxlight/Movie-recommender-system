"""
Builds model/movie_list.pkl and model/vectors.pkl from the TMDB 5000
movies/credits CSVs, following the same cleaning/tagging pipeline as
notebook86c26b4f17.ipynb.

Note: we store the sparse CountVectorizer output (vectors.pkl), not a
precomputed dense similarity matrix. A full 4806x4806 similarity matrix is
~185MB, which exceeds GitHub's 100MB file limit and is unnecessary — cosine
similarity for one selected movie against all others is computed on demand
in app.py, which is fast and keeps the repo small.

Run once locally before `streamlit run app.py`:
    python build_model.py
"""
import ast
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

MOVIES_CSV = "tmdb_5000_movies.csv"
CREDITS_CSV = "tmdb_5000_credits.csv"
OUTPUT_DIR = "model"


def convert(text):
    return [i["name"] for i in ast.literal_eval(text)]


def convert_cast(text):
    return [i["name"] for i in ast.literal_eval(text)][:3]


def fetch_director(text):
    return [i["name"] for i in ast.literal_eval(text) if i["job"] == "Director"]


def collapse(items):
    return [i.replace(" ", "") for i in items]


def main():
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    movies = movies.merge(credits, on="title")
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["cast"] = movies["cast"].apply(collapse)
    movies["crew"] = movies["crew"].apply(collapse)
    movies["genres"] = movies["genres"].apply(collapse)
    movies["keywords"] = movies["keywords"].apply(collapse)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    movies["tags"] = (
        movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    )

    new = movies.drop(columns=["overview", "genres", "keywords", "cast", "crew"])
    new = new.reset_index(drop=True)
    new["tags"] = new["tags"].apply(lambda x: " ".join(x))

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new["tags"])  # sparse matrix, kept sparse (no .toarray())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pickle.dump(new, open(os.path.join(OUTPUT_DIR, "movie_list.pkl"), "wb"))
    pickle.dump(vectors, open(os.path.join(OUTPUT_DIR, "vectors.pkl"), "wb"))

    print(f"Saved {len(new)} movies to {OUTPUT_DIR}/movie_list.pkl")
    print(f"Saved sparse vectors {vectors.shape} to {OUTPUT_DIR}/vectors.pkl")


if __name__ == "__main__":
    main()
