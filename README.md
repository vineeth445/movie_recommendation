# movie_recommendation

Movie Recommendation System using Semantic Search

Overview

This project demonstrates how semantic search can be used to recommend movies based on user preferences.

Instead of matching exact keywords, the application converts movie descriptions and user queries into vector embeddings using the all-MiniLM-L6-v2 model and finds the most semantically similar movie using cosine similarity.

Technologies Used

* Python
* Sentence Transformers
* all-MiniLM-L6-v2
* Scikit-learn
* NumPy
* Pandas
* Streamlit

Architecture

User Query
→ Embedding Model
→ Query Vector

Movie Descriptions
→ Embedding Model
→ Movie Vectors

Query Vector

* Movie Vectors
    → Cosine Similarity

Highest Similarity
→ Recommended Movie

Key Concepts

* Embeddings
* Semantic Search
* Vector Similarity
* Cosine Similarity
* Recommendation Systems
* Information Retrieval

Example

Query:
“I love crime”

Recommendation:
“Batman fights crime in Gotham”

Run Locally

pip install -r requirements.txt

streamlit run app.py
