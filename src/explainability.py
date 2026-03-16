import pandas as pd
import numpy as np
import os

from src.data_loader import load_movielens_data
from src.content_model import (
    build_tfidf_model,
    recommend_movies_by_title,
    build_user_profile,
    recommend_movies_for_user_profile
)

from src.collaborative_model import (
    build_user_cf,
    recommend_user_cf,
    build_item_cf,
    recommend_item_cf
)

from src.svd_model import (
    build_svd_model,
    recommend_svd
)

from src.hybrid_model import recommend_hybrid

from src.deep_model import (
    train_neural_recommender,
    recommend_neural_model
)

from src.rl_model import (
    train_q_learning,
    recommend_rl
)

from src.explainability import (
    explain_content_recommendation,
    explain_cf_recommendation,
    explain_hybrid_recommendation,
    explain_neural_recommendation,
    explain_rl_recommendation
)


# ---------------------------------------------------
# Utility
# ---------------------------------------------------

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

def main():

    print_header("Movie Recommendation Project")

    # ---------------------------------------------------
    # Load dataset
    # ---------------------------------------------------

    movies, ratings, tags, links = load_movielens_data()

    print("\nDataset loaded successfully.\n")

    print("Movies shape  :", movies.shape)
    print("Ratings shape :", ratings.shape)
    print("Tags shape    :", tags.shape)
    print("Links shape   :", links.shape)

    sample_title = "Toy Story (1995)"
    sample_user_id = 1


    # ---------------------------------------------------
    # Content Based Recommender
    # ---------------------------------------------------

    print_header("Building TF-IDF Content-Based Recommender")

    tfidf_matrix, tfidf_vectorizer = build_tfidf_model(movies)

    movie_recs = recommend_movies_by_title(
        sample_title,
        movies,
        tfidf_matrix
    )

    print("\nTop Recommendations for Movie:", sample_title)
    print(movie_recs)


    # ---------------------------------------------------
    # User Profile Content Recommender
    # ---------------------------------------------------

    print_header("Top Recommendations for User Profile")

    user_profile = build_user_profile(
        sample_user_id,
        ratings,
        movies,
        tfidf_matrix
    )

    profile_recs = recommend_movies_for_user_profile(
        user_profile,
        movies,
        tfidf_matrix
    )

    print(profile_recs)


    # ---------------------------------------------------
    # User-Based Collaborative Filtering
    # ---------------------------------------------------

    print_header("Building User-Based Collaborative Filtering")

    user_matrix, user_similarity = build_user_cf(ratings)

    user_cf_recs = recommend_user_cf(
        sample_user_id,
        ratings,
        movies,
        user_matrix,
        user_similarity
    )

    print(user_cf_recs)


    # ---------------------------------------------------
    # Item-Based Collaborative Filtering
    # ---------------------------------------------------

    print_header("Building Item-Based Collaborative Filtering")

    item_matrix, item_similarity = build_item_cf(ratings)

    item_cf_recs = recommend_item_cf(
        sample_user_id,
        ratings,
        movies,
        item_matrix,
        item_similarity
    )

    print(item_cf_recs)


    # ---------------------------------------------------
    # Matrix Factorization (SVD)
    # ---------------------------------------------------

    print_header("Building Matrix Factorization with NumPy SVD")

    svd_model = build_svd_model(ratings)

    svd_recs = recommend_svd(
        sample_user_id,
        ratings,
        movies,
        svd_model
    )

    print(svd_recs)


    # ---------------------------------------------------
    # Hybrid Recommender
    # ---------------------------------------------------

    print_header("Building Hybrid Recommender")

    hybrid_recs = recommend_hybrid(
        sample_user_id,
        ratings,
        movies,
        tfidf_matrix,
        user_matrix
    )

    print(hybrid_recs)


    # ---------------------------------------------------
    # Neural Network Recommender
    # ---------------------------------------------------

    print_header("Building Neural Network Recommender")

    neural_model = train_neural_recommender(ratings)

    neural_recs = recommend_neural_model(
        sample_user_id,
        neural_model,
        ratings,
        movies
    )

    print(neural_recs)


    # ---------------------------------------------------
    # Reinforcement Learning Recommender
    # ---------------------------------------------------

    print_header("Building Reinforcement Learning Recommender")

    q_table = train_q_learning(ratings)

    rl_recs = recommend_rl(
        sample_user_id,
        q_table,
        movies
    )

    print(rl_recs)


    # ---------------------------------------------------
    # Explainability Section
    # ---------------------------------------------------

    print_header("Explainability of Recommendations")

    print("\nContent Explanation\n")

    content_explanations = explain_content_recommendation(
        sample_title,
        movie_recs
    )

    print(content_explanations[0])


    print("\nCollaborative Filtering Explanation\n")

    cf_explanations = explain_cf_recommendation(
        sample_user_id,
        user_cf_recs
    )

    print(cf_explanations[0])


    print("\nHybrid Model Explanation\n")

    hybrid_explanations = explain_hybrid_recommendation(
        hybrid_recs
    )

    print(hybrid_explanations[0])


    print("\nNeural Model Explanation\n")

    neural_explanations = explain_neural_recommendation(
        neural_recs
    )

    print(neural_explanations[0])


    print("\nReinforcement Learning Explanation\n")

    rl_explanations = explain_rl_recommendation(
        rl_recs
    )

    print(rl_explanations[0])


    print("\nProject setup is working correctly.")


# ---------------------------------------------------

if __name__ == "__main__":
    main()