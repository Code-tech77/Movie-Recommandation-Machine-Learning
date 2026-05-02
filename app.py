from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


movies = pickle.load(open('model/model.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))
user_matrix = pickle.load(open('model/user_matrix.pkl', 'rb'))
user_similarity = pickle.load(open('model/user_similarity.pkl', 'rb'))


def collaborative_score(user_id=1):
    sim_users = user_similarity[user_id].argsort()[::-1][1:6]
    scores = user_matrix.iloc[sim_users].mean()
    return scores

def hybrid_recommend(movie_title, user_id=1, alpha=0.6):

    idx = movies[movies['title'] == movie_title].index[0]

    content_scores = list(enumerate(similarity[idx]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)

    collab_scores = collaborative_score(user_id)

    hybrid_scores = {}

    for i, score in content_scores:
        collab_score = collab_scores.get(movies.iloc[i].movieId, 0)
        hybrid_scores[i] = (alpha * score) + ((1 - alpha) * collab_score)

    sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    recommended = [movies.iloc[i[0]].title for i in sorted_movies[1:11]]

    return recommended


@app.route('/')
def home():
    return render_template('index.html', movie_list=movies['title'].values)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form['selected_movie']
    recommendations = hybrid_recommend(movie)

    return render_template(
        'index.html',
        movie_list=movies['title'].values,
        recommended_movie_titles=recommendations
    )


if __name__ == '__main__':
    app.run(debug=True)