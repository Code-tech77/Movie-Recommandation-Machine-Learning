import pandas as pd
import pickle
import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("📥 Loading dataset...")

credits = pd.read_csv('data/tmdb_5000_credits.csv', low_memory=False)


def extract_names(text):
    try:
        data = ast.literal_eval(text)
        return " ".join([i['name'] for i in data])
    except:
        return ""

credits['cast'] = credits['cast'].apply(extract_names)
credits['crew'] = credits['crew'].apply(extract_names)

movies = credits[['movie_id', 'title', 'cast', 'crew']].copy()
movies.rename(columns={'movie_id': 'movieId'}, inplace=True)

# Create tags
movies['tags'] = movies['cast'] + " " + movies['crew']

print("🧠 Building content-based model...")

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

print("📊 Creating synthetic ratings (for hybrid)...")


np.random.seed(42)

ratings = pd.DataFrame({
    'userId': np.random.randint(1, 50, 10000),
    'movieId': np.random.choice(movies['movieId'], 10000),
    'rating': np.random.randint(1, 6, 10000)
})

user_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

user_similarity = cosine_similarity(user_matrix)

print("💾 Saving models...")

pickle.dump(movies, open('model/model.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))
pickle.dump(user_matrix, open('model/user_matrix.pkl', 'wb'))
pickle.dump(user_similarity, open('model/user_similarity.pkl', 'wb'))

print("✅ Training complete!")