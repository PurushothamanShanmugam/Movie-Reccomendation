# AI-Powered Movie Recommendation System

A comprehensive **machine learning based movie recommendation system** built using the **MovieLens dataset**.  
This project implements multiple recommendation techniques including **Content-Based Filtering, Collaborative Filtering, Matrix Factorization, Hybrid Models, Deep Learning, and Reinforcement Learning**.

The system evaluates recommendation performance using **RMSE, Precision@K, and Recall@K** metrics and generates visualizations and model artifacts.

---

# Project Features

✔ Content-Based Recommendation using **TF-IDF and cosine similarity**

✔ User-Based Collaborative Filtering

✔ Item-Based Collaborative Filtering

✔ Matrix Factorization using **SVD**

✔ Hybrid Recommendation Model

✔ Deep Learning Neural Recommendation Model

✔ Reinforcement Learning based recommender

✔ Model evaluation using:

- RMSE
- Precision@K
- Recall@K

✔ Automatic generation of:

- Model metrics
- Recommendation CSV outputs
- Visualization figures
- Saved trained models

---

# Project Structure
MovieRecommendation
│
├── data
│ ├── raw
│ │ ├── movies.csv
│ │ ├── ratings.csv
│ │ ├── tags.csv
│ │ └── links.csv
│ │
│ └── processed
│
├── notebooks
│
├── outputs
│ ├── figures
│ ├── metrics
│ ├── models
│ └── recommendations
│
├── src
│ ├── config.py
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── content_based.py
│ ├── collaborative.py
│ ├── matrix_factorization.py
│ ├── hybrid_model.py
│ ├── deep_model.py
│ └── rl_recommender.py
│
├── tests
│ └── test_smoke.py
│
├── main.py
├── requirements.txt
└── README.md


---

# Installation

### Clone the repository


git clone https://github.com/YOUR_USERNAME/MovieRecommendation.git
cd MovieRecommendation

# Create a Conda environment
conda create -n movie_rec_env python=3.10
conda activate movie_rec_env

# Install dependencies
pip install -r requirements.txt
Run the Project

# Execute the main pipeline:
python main.py

# This will:
Load the MovieLens dataset
Train all recommendation models
Generate recommendations
Evaluate models

# Save outputs
Outputs Generated
Processed Data
data/processed/

# Example:
processed_movies_ratings.csv
Recommendations
outputs/recommendations/

# Example:
content_recommendations.csv
user_cf_recommendations.csv
item_cf_recommendations.csv
svd_recommendations.csv
hybrid_recommendations.csv
neural_recommendations.csv
rl_recommendations.csv

# Metrics
outputs/metrics/

# Example:
model_metrics.json
model_metrics.csv

# Metrics include:
RMSE
Precision@10
Recall@10

# Figures
outputs/figures/

# Example:
content_based.png
user_cf.png
item_cf.png
svd.png
hybrid.png
neural.png
rl.png

# Saved Models
outputs/models/

# Example:
content_tfidf_matrix.pkl
user_item_matrix.pkl
svd_artifacts.npz
neural_model.keras
q_table.pkl
Recommendation Algorithms Implemented
Content-Based Filtering

# Uses:
TF-IDF vectorization
Cosine similarity
Movie genre similarity
Collaborative Filtering

# Two approaches implemented:
User-Based CF
Finds similar users
Item-Based CF
Finds similar movies
Matrix Factorization (SVD)
Uses Singular Value Decomposition to learn latent features of users and movies.

# Hybrid Recommendation

# Combines:

Content-based filtering
Collaborative filtering
for improved recommendation quality.

# Deep Learning Model
Neural network based recommender trained using:
User features
Movie features
Genre embeddings

# Framework used:
TensorFlow / Keras
Reinforcement Learning Recommender

Implements Q-learning to optimize recommendation policy based on reward signals.
Evaluation Metrics
Metric	Purpose
RMSE	Prediction error for ratings
Precision@K	Recommendation relevance
Recall@K	Coverage of relevant items

# Dataset

Dataset used:
MovieLens Dataset

Source:
https://grouplens.org/datasets/movielens/

Files used:
movies.csv
ratings.csv
tags.csv
links.csv

# Technologies Used

Python
Pandas
NumPy
Scikit-learn
TensorFlow / Keras
Matplotlib
Joblib

# Future Improvements

Possible extensions:
Transformer-based recommender models
Graph Neural Networks (GNN)
Online learning recommender
Real-time recommendation API
Web application interface

# Author
Purushothaman Shanmugam (M25DE1033)
M.Tech / Data Engineering
Indian Institute of Technology Jodhpur

