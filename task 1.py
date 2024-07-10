import pandas as pd

# Load the dataset
ratings = pd.read_csv('path_to_ratings.csv')
movies = pd.read_csv('path_to_movies.csv')

# Display basic informat
# print(ratings.info())
print(movies.info())

print(ratings.isnull().sum())
print(movies.isnull().sum())
print(ratings.describe())
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_item_matrix.fillna(0)

import os
from surprise import SVD, Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import accuracy

# Load data into surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split data into training and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# User-based collaborative filtering
algo = KNNBasic(sim_options={'user_based': True})
algo.fit(trainset)

# Make predictions and evaluate
predictions = algo.test(testset)
accuracy.rmse(predictions)
def get_top_n_recommendations(algo, userId, n=10):
    # Get a list of all movieIds
    movie_ids = ratings['movieId'].unique()
    
    # Get the movies the user has already rated
    user_rated_movies = ratings[ratings['userId'] == userId]['movieId']
    
    # Predict ratings for all movies
    predictions = [algo.predict(userId, movie_id) for movie_id in movie_ids if movie_id not in user_rated_movies]
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get the top n predictions
    top_n = predictions[:n]
    
    # Movie titles
    top_n_titles = [movies[movies['movieId'] == pred.iid]['title'].values[0] for pred in top_n]
    
    return top_n_titles
print(get_top_n_recommendations(algo, userId=1, n=10))
from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['userId'])
    recommendations = get_top_n_recommendations(algo, userId=user_id, n=10)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)


