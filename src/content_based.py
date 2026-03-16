import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_similarity(movies: pd.DataFrame):
    """
    Build TF-IDF matrix from movie genres and compute cosine similarity.
    """
    movies = movies.copy()

    # Replace missing genres if any
    movies["genres"] = movies["genres"].fillna("")

    # Convert pipe-separated genres into space-separated text
    movies["genres_text"] = movies["genres"].str.replace("|", " ", regex=False)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres_text"])

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies, tfidf_matrix, similarity_matrix


def get_content_recommendations(
    title: str,
    movies: pd.DataFrame,
    similarity_matrix,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Recommend similar movies based on genre TF-IDF similarity.
    """
    matches = movies[movies["title"].str.lower() == title.lower()]

    if matches.empty:
        raise ValueError(f"Movie '{title}' not found in dataset.")

    movie_index = matches.index[0]

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Exclude the same movie itself
    similarity_scores = similarity_scores[1: top_n + 1]

    movie_indices = [i[0] for i in similarity_scores]
    scores = [i[1] for i in similarity_scores]

    recommendations = movies.iloc[movie_indices][["movieId", "title", "genres"]].copy()
    recommendations["similarity_score"] = scores

    return recommendations.reset_index(drop=True)


def build_user_profile(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix
):
    """
    Build a user profile as a weighted average of TF-IDF movie vectors,
    weighted by the user's ratings.
    """
    user_ratings = ratings[ratings["userId"] == user_id].copy()

    if user_ratings.empty:
        raise ValueError(f"User {user_id} not found in ratings data.")

    # Merge ratings with movie indices
    movie_index_map = pd.Series(movies.index, index=movies["movieId"]).to_dict()
    user_ratings["movie_index"] = user_ratings["movieId"].map(movie_index_map)
    user_ratings = user_ratings.dropna(subset=["movie_index"]).copy()
    user_ratings["movie_index"] = user_ratings["movie_index"].astype(int)

    if user_ratings.empty:
        raise ValueError(f"No matching rated movies found for user {user_id}.")

    rated_indices = user_ratings["movie_index"].values
    weights = user_ratings["rating"].values.reshape(-1, 1)

    rated_movie_vectors = tfidf_matrix[rated_indices]

    # Weighted sum of rated movie vectors
    weighted_profile = rated_movie_vectors.multiply(weights).sum(axis=0)

    # Normalize by total rating sum
    total_rating = user_ratings["rating"].sum()
    user_profile = weighted_profile / total_rating

    return csr_matrix(user_profile), user_ratings


def get_user_profile_recommendations(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Recommend movies to a user based on cosine similarity between
    the user profile and all movie TF-IDF vectors.
    """
    user_profile, user_ratings = build_user_profile(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
    )

    similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()

    recommendations = movies.copy()
    recommendations["similarity_score"] = similarity_scores

    # Remove already-rated movies
    rated_movie_ids = set(user_ratings["movieId"].tolist())
    recommendations = recommendations[~recommendations["movieId"].isin(rated_movie_ids)]

    recommendations = recommendations.sort_values(
        by="similarity_score", ascending=False
    ).head(top_n)

    return recommendations[["movieId", "title", "genres", "similarity_score"]].reset_index(drop=True)


def precision_recall_at_k_content(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix,
    k: int = 10,
    relevant_threshold: float = 4.0
):
    """
    Compute Precision@K and Recall@K for a user-profile recommender.

    Relevant movies = movies the user rated >= relevant_threshold.
    Ground truth is derived from the same user's highly rated movies.
    This is a simple educational evaluation aligned with the assignment.
    """
    user_all_ratings = ratings[ratings["userId"] == user_id].copy()

    if user_all_ratings.empty:
        raise ValueError(f"User {user_id} not found in ratings data.")

    relevant_movies = set(
        user_all_ratings[user_all_ratings["rating"] >= relevant_threshold]["movieId"].tolist()
    )

    if len(relevant_movies) == 0:
        return 0.0, 0.0

    recommendations = get_user_profile_recommendations(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        top_n=k
    )

    recommended_movies = set(recommendations["movieId"].tolist())
    true_positives = len(recommended_movies.intersection(relevant_movies))

    precision = true_positives / k if k > 0 else 0.0
    recall = true_positives / len(relevant_movies) if len(relevant_movies) > 0 else 0.0

    return precision, recall