import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests

# Load the preprocessed data (new_df) from pickle file
with open('new_df.pkl', 'rb') as f:
    new_df = pickle.load(f)

# --- Regenerate vec and similarity matrix on the fly ---
# Vectorize the tags (assuming new_df['tags'] is available)
cv = CountVectorizer(max_features=5000, stop_words='english', lowercase=True)
vec = cv.fit_transform(new_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vec)
# --- End regeneration ---


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Movie recommendation function
def recommend(movie):
    # Ensure movie exists in new_df
    if movie not in new_df['original_title'].values:
        return ["Movie not found in database."], [] # Return empty list for posters if movie not found

    index = new_df[new_df['original_title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    recommended_movies_posters = []
    for i in distances[1:6]: # Get top 5 similar movies
        movie_id = new_df.iloc[i[0]].movie_id
        recommended_movies_posters.append(fetch_poster(movie_id))
        recommended_movies.append(new_df.iloc[i[0]].original_title)
    return recommended_movies, recommended_movies_posters

# Streamlit app interface
st.title('Movie Recommender System')

selected_movie = st.selectbox(
    'Select a movie to get recommendations:',
    new_df['original_title'].values
)

if st.button('Show Recommendation'):
    if selected_movie:
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
        col1, col2, col3, col4, col5 = st.columns(5)
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
    else:
        st.write("Please select a movie.")
