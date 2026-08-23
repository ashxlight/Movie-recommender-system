import pickle
import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity

PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"

# Get a free key at https://www.omdbapi.com/apikey.aspx and either:
#   - set it here, or
#   - put OMDB_API_KEY = "yourkey" in .streamlit/secrets.toml (preferred, keeps it out of git)
OMDB_API_KEY = st.secrets.get("OMDB_API_KEY", "")

def fetch_poster(title):
    if not OMDB_API_KEY:
        return PLACEHOLDER_POSTER
    url = "https://www.omdbapi.com/?t={}&apikey={}".format(requests.utils.quote(title), OMDB_API_KEY)
    try:
        data = requests.get(url, timeout=(3, 5))
        data = data.json()
        poster = data.get('Poster')
        if not poster or poster == 'N/A':
            return PLACEHOLDER_POSTER
        return poster
    except requests.RequestException:
        return PLACEHOLDER_POSTER

def recommend(movie, fetch_posters=True):
    index = movies[movies['title'] == movie].index[0]
    # Compute similarity of this one movie against all others on demand,
    # instead of loading a precomputed full N x N matrix.
    sims = cosine_similarity(vectors[index], vectors).flatten()
    distances = sorted(list(enumerate(sims)), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        title = movies.iloc[i[0]].title
        recommended_movie_posters.append(fetch_poster(title) if fetch_posters else None)
        recommended_movie_names.append(title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender System')
movies = pickle.load(open('model/movie_list.pkl','rb'))
vectors = pickle.load(open('model/vectors.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

fetch_posters = st.checkbox(
    "Fetch posters from OMDb (needs internet access + OMDB_API_KEY in secrets; uncheck for instant offline results)",
    value=True,
)
if fetch_posters and not OMDB_API_KEY:
    st.warning(
        "No OMDB_API_KEY set — posters will show as placeholders. "
        "Get a free key at https://www.omdbapi.com/apikey.aspx and add it to .streamlit/secrets.toml."
    )

if st.button('Show Recommendation'):
    spinner_msg = 'Fetching recommendations and posters...' if fetch_posters else 'Fetching recommendations...'
    with st.spinner(spinner_msg):
        recommended_movie_names,recommended_movie_posters = recommend(selected_movie, fetch_posters)
    columns = st.columns(5)
    for col, name, poster in zip(columns, recommended_movie_names, recommended_movie_posters):
        with col:
            st.text(name)
            if poster:
                st.image(poster)





