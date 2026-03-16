import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Create user-item rating matrix.
    Rows = users, columns = movies, values = ratings.
    """
    user_item = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )
    return user_item


def compute_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user-user cosine similarity.
    Missing ratings are filled with 0 for similarity calculation.
    """
    filled_matrix = user_item_matrix.fillna(0)
    similarity = cosine_similarity(filled_matrix)

    user_similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    return user_similarity_df


def compute_item_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute item-item cosine similarity.
    Missing ratings are filled with 0 for similarity calculation.
    """
    filled_matrix = user_item_matrix.fillna(0).T
    similarity = cosine_similarity(filled_matrix)

    item_similarity_df = pd.DataFrame(
        similarity,
        index=filled_matrix.index,
        columns=filled_matrix.index
    )
    return item_similarity_df


def predict_rating_user_based(
    target_user: int,
    movie_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    k: int = 20
) -> float:
    """
    Predict a user's rating for a movie using top-K similar users.
    """
    if movie_id not in user_item_matrix.columns:
        return np.nan

    if target_user not in user_item_matrix.index:
        return np.nan

    movie_ratings = user_item_matrix[movie_id].dropna()

    if movie_ratings.empty:
        return np.nan

    sims = user_similarity.loc[target_user, movie_ratings.index]

    if target_user in sims.index:
        sims = sims.drop(target_user, errors="ignore")
        movie_ratings = movie_ratings.drop(target_user, errors="ignore")

    if sims.empty:
        return np.nan

    top_k_users = sims.sort_values(ascending=False).head(k)
    neighbor_ratings = movie_ratings.loc[top_k_users.index]

    denominator = np.abs(top_k_users).sum()
    if denominator == 0:
        return np.nan

    predicted_rating = np.dot(top_k_users.values, neighbor_ratings.values) / denominator
    return float(predicted_rating)


def predict_rating_item_based(
    target_user: int,
    movie_id: int,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    k: int = 20
) -> float:
    """
    Predict a user's rating for a movie using top-K similar items
    that the user has already rated.
    """
    if target_user not in user_item_matrix.index:
        return np.nan

    if movie_id not in user_item_matrix.columns:
        return np.nan

    user_ratings = user_item_matrix.loc[target_user].dropna()

    if user_ratings.empty:
        return np.nan

    rated_movie_ids = user_ratings.index.tolist()

    if movie_id not in item_similarity.index:
        return np.nan

    sims = item_similarity.loc[movie_id, rated_movie_ids]

    if sims.empty:
        return np.nan

    top_k_items = sims.sort_values(ascending=False).head(k)
    neighbor_ratings = user_ratings.loc[top_k_items.index]

    denominator = np.abs(top_k_items).sum()
    if denominator == 0:
        return np.nan

    predicted_rating = np.dot(top_k_items.values, neighbor_ratings.values) / denominator
    return float(predicted_rating)


def recommend_user_based(
    target_user: int,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 20,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Recommend top-N unseen movies for a target user.
    """
    if target_user not in user_item_matrix.index:
        raise ValueError(f"User {target_user} not found.")

    user_rated_movies = user_item_matrix.loc[target_user].dropna().index.tolist()
    unseen_movies = [m for m in user_item_matrix.columns if m not in user_rated_movies]

    predictions = []
    for movie_id in unseen_movies:
        pred = predict_rating_user_based(
            target_user=target_user,
            movie_id=movie_id,
            user_item_matrix=user_item_matrix,
            user_similarity=user_similarity,
            k=k
        )
        if not np.isnan(pred):
            predictions.append((movie_id, pred))

    pred_df = pd.DataFrame(predictions, columns=["movieId", "predicted_rating"])
    pred_df = pred_df.sort_values(by="predicted_rating", ascending=False).head(top_n)

    result = pred_df.merge(movies, on="movieId", how="left")
    return result[["movieId", "title", "genres", "predicted_rating"]].reset_index(drop=True)


def recommend_item_based(
    target_user: int,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 20,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Recommend top-N unseen movies for a target user using item-based CF.
    """
    if target_user not in user_item_matrix.index:
        raise ValueError(f"User {target_user} not found.")

    user_rated_movies = user_item_matrix.loc[target_user].dropna().index.tolist()
    unseen_movies = [m for m in user_item_matrix.columns if m not in user_rated_movies]

    predictions = []
    for movie_id in unseen_movies:
        pred = predict_rating_item_based(
            target_user=target_user,
            movie_id=movie_id,
            user_item_matrix=user_item_matrix,
            item_similarity=item_similarity,
            k=k
        )
        if not np.isnan(pred):
            predictions.append((movie_id, pred))

    pred_df = pd.DataFrame(predictions, columns=["movieId", "predicted_rating"])
    pred_df = pred_df.sort_values(by="predicted_rating", ascending=False).head(top_n)

    result = pred_df.merge(movies, on="movieId", how="left")
    return result[["movieId", "title", "genres", "predicted_rating"]].reset_index(drop=True)


def evaluate_rmse_user_based(
    ratings: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    k: int = 20,
    sample_size: int = 1000
) -> float:
    """
    Simple RMSE evaluation on a sample of known ratings.
    """
    sample = ratings.sample(n=min(sample_size, len(ratings)), random_state=42)

    y_true = []
    y_pred = []

    for _, row in sample.iterrows():
        user_id = row["userId"]
        movie_id = row["movieId"]
        true_rating = row["rating"]

        pred = predict_rating_user_based(
            target_user=user_id,
            movie_id=movie_id,
            user_item_matrix=user_item_matrix,
            user_similarity=user_similarity,
            k=k
        )

        if not np.isnan(pred):
            y_true.append(true_rating)
            y_pred.append(pred)

    if len(y_true) == 0:
        return np.nan

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(rmse)


def evaluate_rmse_item_based(
    ratings: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    k: int = 20,
    sample_size: int = 1000
) -> float:
    """
    Simple RMSE evaluation on a sample of known ratings for item-based CF.
    """
    sample = ratings.sample(n=min(sample_size, len(ratings)), random_state=42)

    y_true = []
    y_pred = []

    for _, row in sample.iterrows():
        user_id = row["userId"]
        movie_id = row["movieId"]
        true_rating = row["rating"]

        pred = predict_rating_item_based(
            target_user=user_id,
            movie_id=movie_id,
            user_item_matrix=user_item_matrix,
            item_similarity=item_similarity,
            k=k
        )

        if not np.isnan(pred):
            y_true.append(true_rating)
            y_pred.append(pred)

    if len(y_true) == 0:
        return np.nan

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(rmse)


def precision_recall_at_k_user_based(
    target_user: int,
    ratings: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    movies: pd.DataFrame,
    k_neighbors: int = 20,
    top_k: int = 10,
    relevant_threshold: float = 4.0
):
    """
    Compute Precision@K and Recall@K for user-based CF.
    """
    user_true = ratings[ratings["userId"] == target_user]
    relevant_items = set(user_true[user_true["rating"] >= relevant_threshold]["movieId"].tolist())

    if len(relevant_items) == 0:
        return 0.0, 0.0

    recs = recommend_user_based(
        target_user=target_user,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
        movies=movies,
        k=k_neighbors,
        top_n=top_k
    )

    recommended_items = set(recs["movieId"].tolist())
    true_positives = len(recommended_items.intersection(relevant_items))

    precision = true_positives / top_k if top_k > 0 else 0.0
    recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0.0

    return precision, recall


def precision_recall_at_k_item_based(
    target_user: int,
    ratings: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    movies: pd.DataFrame,
    k_neighbors: int = 20,
    top_k: int = 10,
    relevant_threshold: float = 4.0
):
    """
    Compute Precision@K and Recall@K for item-based CF.
    """
    user_true = ratings[ratings["userId"] == target_user]
    relevant_items = set(user_true[user_true["rating"] >= relevant_threshold]["movieId"].tolist())

    if len(relevant_items) == 0:
        return 0.0, 0.0

    recs = recommend_item_based(
        target_user=target_user,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
        movies=movies,
        k=k_neighbors,
        top_n=top_k
    )

    recommended_items = set(recs["movieId"].tolist())
    true_positives = len(recommended_items.intersection(relevant_items))

    precision = true_positives / top_k if top_k > 0 else 0.0
    recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0.0

    return precision, recall