app.py

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

Page Title

st.title("🎬 Movie Recommendation System")

Movie Dataset

movies = [
"Avengers is a superhero action movie",
"Iron Man is a superhero movie",
"Batman fights crime in Gotham",
"Titanic is a romantic drama",
"The Notebook is a love story",
"Interstellar is a science fiction space movie",
"The Martian is about survival in space"
]

Load Embedding Model

@st.cache_resource
def load_model():
return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

Generate Movie Embeddings

movie_embeddings = model.encode(movies)

User Input

query = st.text_input(
"Enter your movie preference:",
placeholder="Example: I love crime movies"
)

if query:

# Convert Query to Embedding
query_embedding = model.encode([query])
# Calculate Similarity
scores = cosine_similarity(
    query_embedding,
    movie_embeddings
)
# Get Best Match
best_match = np.argmax(scores)
# Display Recommendation
st.subheader("🎯 Recommended Movie")
st.success(movies[best_match])
# Similarity Table
df = pd.DataFrame({
    "Movie": movies,
    "Similarity Score": scores[0]
})
st.subheader("📊 Similarity Scores")
st.dataframe(
    df.sort_values(
        by="Similarity Score",
        ascending=False
    ),
    use_container_width=True
)
