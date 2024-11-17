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


# # Load the BERT tokenizer and model
# logging.set_verbosity_error()
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# bert_model = BertModel.from_pretrained('bert-large-uncased').cuda()

# def tokenize_texts(texts, max_len=128):
#     return tokenizer(
#         texts.tolist(),
#         max_length=max_len,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )

# tokenized_data = tokenize_texts(data['book_content'], max_len=128)
# print("Finished tokenizing the data.")

# def generate_bert_embeddings_in_batches(tokenized_data, bert_model, batch_size=32, device='cuda'):
#     # Move the model to the specified device (GPU or CPU)
#     bert_model = bert_model.to(device)
    
#     # Initialize an empty list to store the embeddings
#     all_embeddings = []
    
#     # Calculate total batches
#     total_samples = tokenized_data['input_ids'].shape[0]
    
#     for start_idx in range(0, total_samples, batch_size):
#         end_idx = min(start_idx + batch_size, total_samples)
        
#         # Slice batch input_ids and attention_mask
#         input_ids_batch = tokenized_data['input_ids'][start_idx:end_idx].to(device)  # Move to the same device
#         attention_mask_batch = tokenized_data['attention_mask'][start_idx:end_idx].to(device)  # Move to the same device

#         # Get BERT embeddings without computing gradients
#         with torch.no_grad():
#             batch_embeddings = bert_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)[1]
        
#         # Move embeddings back to CPU to save GPU memory
#         all_embeddings.append(batch_embeddings.cpu())
        
#         # Optionally clear cache to free memory
#         torch.cuda.empty_cache()

#     # Concatenate all batch embeddings into a single tensor
#     return torch.cat(all_embeddings, dim=0)

# # Use the function to generate embeddings in batches
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# bert_embeddings = generate_bert_embeddings_in_batches(tokenized_data, bert_model, batch_size=32, device=device)
# print("Finished generating BERT embeddings.")

# class PairDataset(Dataset):
#     def __init__(self, data, bert_embeddings):
#         self.data = data
#         self.bert_embeddings = bert_embeddings

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         book1_emb = self.bert_embeddings[idx]
#         book2_emb = self.bert_embeddings[(idx + 1) % len(self.data)]  # Pair with next item
#         label = 1 if self.data['categories'].iloc[idx] == self.data['categories'].iloc[(idx + 1) % len(self.data)] else 0
#         return book1_emb, book2_emb, label
    
# class CustomBranch(nn.Module):
#     def __init__(self):
#         super(CustomBranch, self).__init__()
#         self.fc1 = nn.Linear(1024, 512)  # First dense layer, increased number of units
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(0.3)  # Dropout to prevent overfitting
        
#         self.fc2 = nn.Linear(512, 256)  # Second dense layer
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(0.3)  # Dropout again

#         self.fc3 = nn.Linear(256, 128)  # Third dense layer (matches original)
#         self.relu3 = nn.ReLU()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.dropout1(x)

#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.dropout2(x)
        
#         x = self.fc3(x)
#         x = self.relu3(x)
        
#         return x
    

# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()
#         self.cnn_branch = CustomBranch()
#         self.fc1 = nn.Linear(1024, 128)  # Assuming BERT output size is 768
#         self.fc2 = nn.Linear(128, 64)   # Reducing to 64 dimensions
#         self.fc3 = nn.Linear(64 * 2, 2)  # Concatenating two 64-dim vectors, and output size 2 for binary classification

#     def forward_once(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return x

#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         concatenated = torch.cat((output1, output2), dim=1)
#         output = self.fc3(concatenated)
#         return output

# # Dataset and DataLoader
# pair_dataset = PairDataset(data, bert_embeddings)
# pair_loader = DataLoader(pair_dataset, batch_size=64, shuffle=True)
# print("Finished creating the PairDataset and DataLoader.")

# # Initialize the Siamese model, loss function, and optimizer
# siamese_model = SiameseNetwork().cuda()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(siamese_model.parameters(), lr=1e-5, weight_decay=1e-2)
# print("Finished initializing the Siamese model, loss function, and optimizer.")


# epochs = 100
# for epoch in range(epochs):
#     siamese_model.train()
#     running_loss = 0.0
#     for batch in pair_loader:
#         book1_emb, book2_emb, labels = batch
#         book1_emb, book2_emb, labels = book1_emb.cuda(), book2_emb.cuda(), labels.cuda()

#         optimizer.zero_grad()
#         outputs = siamese_model(book1_emb, book2_emb)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(pair_loader):.4f}')

# # Save the trained model
# torch.save(siamese_model.state_dict(), 'siamese_model.pth')


# # Extract book embeddings from the CNN branch of the Siamese model
# def extract_embeddings_from_model(bert_embeddings, siamese_model):
#     siamese_model.eval()  # Set model to evaluation mode
#     with torch.no_grad():  # No gradient computation
#         book_embeddings = siamese_model.cnn_branch(bert_embeddings.cuda())  # Pass through CNN branch
#     return book_embeddings.cpu().detach().numpy()  # Move to CPU and detach from computation graph

# # Use the function to get the processed embeddings
# book_embeddings = extract_embeddings_from_model(bert_embeddings, siamese_model)
# print("Finished extracting book embeddings from the Siamese model.")

# # Save the embeddings for future use
# torch.save(book_embeddings, 'bert_embeddings.pt')

# normalized_book_embeddings = normalize(book_embeddings)

# Manhattan distance
# manhattan_dist_matrix = cdist(normalized_book_embeddings, normalized_book_embeddings, metric='cityblock') 
# print("Finished computing the Manhattan distance matrix.")

# p_value = 3
# minkowski_dist_matrix = cdist(normalized_book_embeddings, normalized_book_embeddings, metric='minkowski', p=p_value)
# print("Finished computing the Minkowski distance matrix.")


# def recommend_books_by_manhattan(book_title, threshold, manhattan_dist_matrix):
#     idx = data[data['Title'] == book_title].index[0]
    
#     # Compute the Manhattan distance scores
#     dist_scores = list(enumerate(manhattan_dist_matrix[idx]))
    
#     # Sort the books based on Manhattan distance (lower distance means more similar)
#     dist_scores = sorted(dist_scores, key=lambda x: x[1], reverse=False)
    
#     # Filter recommendations based on the threshold (optional)
#     recommendations = [
#         {'title': data['Title'].iloc[i],
#          'score': "{:.5f}".format(score),
#          'description': data['description'].fillna('').iloc[i],
#          'thumbnail': data['image'].fillna('').iloc[i],
#          'url': data['previewLink'].fillna('').iloc[i]}
#         for i, score in dist_scores if score <= threshold
#     ][:100]  # Limit results to 100
    
#     return recommendations

import os
import pickle
import numpy as np

method = 'chebyshev'

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
    recommended_books = recommend_books(book_title, 1.0, 'manhattan')
    return jsonify(recommended_books)

app.run()
print(f'flask app running at {app.url_map}')