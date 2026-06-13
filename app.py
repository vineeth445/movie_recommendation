import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

st.title(“🎬 Movie Recommendation System”)

movies = [
“Avengers is a superhero action movie”,
“Iron Man is a superhero movie”,
“Batman fights crime in Gotham”,
“Titanic is a romantic drama”,
“The Notebook is a love story”,
“Interstellar is a science fiction space movie”,
“The Martian is about survival in space”
]

model = SentenceTransformer(“all-MiniLM-L6-v2”)

movie_embeddings = model.encode(movies)

query = st.text_input(“Enter movie preference”)

if query:

query_embedding = model.encode([query])
scores = cosine_similarity(
    query_embedding,
    movie_embeddings
)
best_match = np.argmax(scores)
st.subheader("Recommended Movie")
st.success(movies[best_match])
df = pd.DataFrame({
    "Movie": movies,
    "Similarity Score": scores[0]
})
st.subheader("Similarity Scores")
st.dataframe(
    df.sort_values(
        by="Similarity Score",
        ascending=False
    )
)
