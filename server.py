# Description: This file contains the code for the Flask server that will be used to serve the book recommendation system.
# The server will be hosted on the local machine and will provide a REST API endpoint to get book recommendations based on a given book title.

# %pip install flask
# %pip install flask-cors
# %pip install transformers
# %pip install torch
# %pip install pandas
# %pip install numpy
# %pip install scikit-learn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging

app = Flask(__name__)
CORS(app)
print(f'Flask app started: {__name__}')

data = pd.read_csv("books_data.csv", nrows=40000)
data['Title'] = data['Title'].fillna('Unknown')
data['categories'] = data['categories'].fillna('Unknown')
data['description'] = data['description'].fillna('')
data['description'] = data['description'].apply(lambda x: x.lower())
data['book_content'] = (
    (data['Title'] + ' ') * 2
    + data['description'] + ' '
    + data['authors'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' '
    + data['categories'].apply(lambda x: ' '.join(x) * 5 if isinstance(x, list) else '')
)
data['book_content'] = data['book_content'].str.replace(r'[^\w\s]', '', regex=True).str.lower()


import os
import pickle
import numpy as np

method = 'manhattan'

folder_path = f'dumped_matrices/{method}'
file_count = len(os.listdir(folder_path))

matrix = []

for i in range(file_count):
    file_path = os.path.join(folder_path, f'{method}_matrix_chunk_{i}.pkl')
    with open(file_path, 'rb') as f:
        chunk = pickle.load(f)
        matrix.append(chunk)
    print(f'Loaded {i} / {file_count} chunks from {file_path}')

# Convert the list of chunks into a full matrix
matrix = np.concatenate(matrix, axis=0)

def recommend_books(book_title, threshold, method='manhattan'):
    idx = data[data['Title'] == book_title].index[0]
    
    dist_scores = list(enumerate(matrix[idx]))
    dist_scores = sorted(dist_scores, key=lambda x: x[1], reverse=False)
    
    # Filter recommendations based on the threshold
    recommendations = [
        {'title': data['Title'].iloc[i],
         'score': "{:.5f}".format(score),
         'description': data['description'].fillna('').iloc[i],
         'thumbnail': data['image'].fillna('').iloc[i],
         'url': data['previewLink'].fillna('').iloc[i]}
        for i, score in dist_scores if score <= threshold
    ][:100]  # Limit results to 100
    
    return recommendations


@app.route('/recommend/<string:book_title>', methods=['GET'])
def recommend(book_title):
    recommended_books = recommend_books(book_title, 200, 'manhattan')
    return jsonify(recommended_books)

app.run()
print(f'flask app running at {app.url_map}')