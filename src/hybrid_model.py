import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def get_content_scores_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix
) -> pd.DataFrame:
    """
    Compute content-based similarity scores for all unseen movies
    for a given user using the user-profile method.
    """
    user_ratings = ratings[ratings["userId"] == user_id].copy()

    if user_ratings.empty:
        raise ValueError(f"User {user_id} not found.")

    movie_index_map = pd.Series(movies.index, index=movies["movieId"]).to_dict()
    user_ratings["movie_index"] = user_ratings["movieId"].map(movie_index_map)
    user_ratings = user_ratings.dropna(subset=["movie_index"]).copy()
    user_ratings["movie_index"] = user_ratings["movie_index"].astype(int)

    rated_indices = user_ratings["movie_index"].values
    weights = user_ratings["rating"].values.reshape(-1, 1)

    rated_movie_vectors = tfidf_matrix[rated_indices]
    weighted_profile = rated_movie_vectors.multiply(weights).sum(axis=0)

    total_rating = user_ratings["rating"].sum()
    user_profile = weighted_profile / total_rating

    user_profile = np.asarray(user_profile)
    similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()

    scores_df = movies.copy()
    scores_df["content_score"] = similarity_scores

    rated_movie_ids = set(user_ratings["movieId"].tolist())
    scores_df = scores_df[~scores_df["movieId"].isin(rated_movie_ids)].copy()

    return scores_df[["movieId", "title", "genres", "content_score"]].reset_index(drop=True)


def get_item_cf_scores_for_user(
    target_user: int,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 20
) -> pd.DataFrame:
    """
    Compute item-based CF predicted scores for all unseen movies.
    """
    if target_user not in user_item_matrix.index:
        raise ValueError(f"User {target_user} not found.")

    user_rated_movies = user_item_matrix.loc[target_user].dropna().index.tolist()
    unseen_movies = [m for m in user_item_matrix.columns if m not in user_rated_movies]

    predictions = []
    user_ratings = user_item_matrix.loc[target_user].dropna()

    for movie_id in unseen_movies:
        if movie_id not in item_similarity.index:
            continue

        sims = item_similarity.loc[movie_id, user_rated_movies]
        top_k_items = sims.sort_values(ascending=False).head(k)

        neighbor_ratings = user_ratings.loc[top_k_items.index]
        denominator = np.abs(top_k_items).sum()

        if denominator == 0:
            continue

        predicted_rating = np.dot(top_k_items.values, neighbor_ratings.values) / denominator
        predictions.append((movie_id, predicted_rating))

    pred_df = pd.DataFrame(predictions, columns=["movieId", "cf_score"])
    result = pred_df.merge(movies, on="movieId", how="left")

    return result[["movieId", "title", "genres", "cf_score"]].reset_index(drop=True)


def min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Normalize values to [0,1].
    """
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)

    return (series - min_val) / (max_val - min_val)


def recommend_hybrid(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    alpha: float = 0.5,
    k_cf: int = 20,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Hybrid recommender combining content-based and item-based CF scores.
    """
    content_df = get_content_scores_for_user(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        tfidf_matrix=tfidf_matrix
    )

    cf_df = get_item_cf_scores_for_user(
        target_user=user_id,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
        movies=movies,
        k=k_cf
    )

    merged = content_df.merge(
        cf_df[["movieId", "cf_score"]],
        on="movieId",
        how="inner"
    )

    if merged.empty:
        raise ValueError("No overlapping hybrid candidates found.")

    merged["content_score_norm"] = min_max_normalize(merged["content_score"])
    merged["cf_score_norm"] = min_max_normalize(merged["cf_score"])

    merged["hybrid_score"] = (
        alpha * merged["content_score_norm"] +
        (1 - alpha) * merged["cf_score_norm"]
    )

    merged = merged.sort_values(by="hybrid_score", ascending=False).head(top_n)

    return merged[
        ["movieId", "title", "genres", "content_score", "cf_score", "hybrid_score"]
    ].reset_index(drop=True)


def evaluate_rmse_hybrid(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    alpha: float = 0.5,
    sample_size: int = 500
) -> float:
    """
    Simple hybrid RMSE evaluation on sampled known ratings.
    """
    sample = ratings.sample(n=min(sample_size, len(ratings)), random_state=42)

    y_true = []
    y_pred = []

    movie_index_map = pd.Series(movies.index, index=movies["movieId"]).to_dict()

    for _, row in sample.iterrows():
        user_id = row["userId"]
        movie_id = row["movieId"]
        true_rating = row["rating"]

        if user_id not in user_item_matrix.index or movie_id not in movie_index_map:
            continue

        # Content score
        try:
            user_ratings = ratings[ratings["userId"] == user_id].copy()
            user_ratings["movie_index"] = user_ratings["movieId"].map(movie_index_map)
            user_ratings = user_ratings.dropna(subset=["movie_index"]).copy()
            user_ratings["movie_index"] = user_ratings["movie_index"].astype(int)

            rated_indices = user_ratings["movie_index"].values
            weights = user_ratings["rating"].values.reshape(-1, 1)
            rated_movie_vectors = tfidf_matrix[rated_indices]
            weighted_profile = rated_movie_vectors.multiply(weights).sum(axis=0)
            total_rating = user_ratings["rating"].sum()
            user_profile = weighted_profile / total_rating

            target_index = movie_index_map[movie_id]
            content_score = cosine_similarity(user_profile, tfidf_matrix[target_index]).flatten()[0]
        except Exception:
            continue

        # CF score
        try:
            user_rated_movies = user_item_matrix.loc[user_id].dropna()
            rated_movie_ids = user_rated_movies.index.tolist()

            if movie_id not in item_similarity.index:
                continue

            sims = item_similarity.loc[movie_id, rated_movie_ids]
            top_k_items = sims.sort_values(ascending=False).head(20)
            neighbor_ratings = user_rated_movies.loc[top_k_items.index]
            denominator = np.abs(top_k_items).sum()

            if denominator == 0:
                continue

            cf_score = np.dot(top_k_items.values, neighbor_ratings.values) / denominator
        except Exception:
            continue

        # Simple blend without normalization for single prediction
        pred = alpha * (content_score * 5.0) + (1 - alpha) * cf_score
        pred = float(np.clip(pred, 0.5, 5.0))

        y_true.append(true_rating)
        y_pred.append(pred)

    if len(y_true) == 0:
        return np.nan

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(rmse)


def precision_recall_at_k_hybrid(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    alpha: float = 0.5,
    top_k: int = 10,
    relevant_threshold: float = 4.0
):
    """
    Compute Precision@K and Recall@K for the hybrid recommender.
    """
    relevant_items = set(
        ratings[
            (ratings["userId"] == user_id) &
            (ratings["rating"] >= relevant_threshold)
        ]["movieId"].tolist()
    )

    if len(relevant_items) == 0:
        return 0.0, 0.0

    recs = recommend_hybrid(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
        alpha=alpha,
        k_cf=20,
        top_n=top_k
    )

    recommended_items = set(recs["movieId"].tolist())
    true_positives = len(recommended_items.intersection(relevant_items))

    precision = true_positives / top_k if top_k > 0 else 0.0
    recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0.0

    return precision, recall