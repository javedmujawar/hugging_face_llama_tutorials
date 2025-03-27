import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import ollama

# Load browser history dataset
def load_data():
    data = pd.DataFrame([
        {"user_id": 101, "url": "https://news.bbc.com/world/tech", "title": "Latest Tech News - BBC", "category": "News", "visit_count": 5},
        {"user_id": 101, "url": "https://github.com/ollama-ai/ollama", "title": "Ollama GitHub Repo", "category": "Coding", "visit_count": 10},
        {"user_id": 101, "url": "https://stackoverflow.com/questions/xyz", "title": "How to fine-tune Llama model", "category": "Coding", "visit_count": 7},
        {"user_id": 102, "url": "https://amazon.com/deals", "title": "Amazon Best Deals", "category": "Shopping", "visit_count": 3},
        {"user_id": 102, "url": "https://netflix.com", "title": "Netflix Homepage", "category": "Entertainment", "visit_count": 4},
        {"user_id": 103, "url": "https://medium.com/ml-trends", "title": "Top ML Trends 2025", "category": "AI/ML", "visit_count": 8},
    ])
    return data

# Content-based recommendation using TF-IDF
def content_based_recommendations(data, url, top_n=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['title'] + " " + data['category'])
    
    url_idx = data[data['url'] == url].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    similar_indices = cosine_sim[url_idx].argsort()[-top_n-1:-1][::-1]
    recommended_urls = data.iloc[similar_indices]['url'].tolist()
    return recommended_urls

# Collaborative Filtering using SVD
def collaborative_filtering_recommendations(data, user_id, top_n=3):
    user_item_matrix = data.pivot(index='user_id', columns='url', values='visit_count').fillna(0)

    # ✅ Convert DataFrame to a sparse matrix
    sparse_matrix = csr_matrix(user_item_matrix.values.astype(float))

    # ✅ Ensure k < min(dimensions)
    k = min(sparse_matrix.shape) - 1
    if k <= 0:
        raise ValueError("Matrix too small for SVD, add more data.")

    U, sigma, Vt = svds(sparse_matrix, k=k)
    sigma = np.diag(sigma)

    # Get predictions
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    preds_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

    # ✅ Ensure user exists in the DataFrame
    if user_id not in preds_df.index:
        return []

    user_preds = preds_df.loc[user_id].sort_values(ascending=False).head(top_n)
    return user_preds.index.tolist()

# AI-based recommendation using Ollama
def ai_recommendations(user_history):
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": f"I visited {', '.join(user_history)}, suggest some similar websites"}])
    return response['message']['content']

# Run the recommendation system
data = load_data()

# Example: Get recommendations for a given URL
print("Content-Based Recommendations:", content_based_recommendations(data, "https://github.com/ollama-ai/ollama"))

# Example: Get recommendations for a given user
print("Collaborative Filtering Recommendations:", collaborative_filtering_recommendations(data, 101))

# Example: AI-based recommendation
test_history = ["https://github.com/ollama-ai/ollama", "https://stackoverflow.com/questions/xyz"]
#print("AI-Based Recommendations:", ai_recommendations(test_history))
