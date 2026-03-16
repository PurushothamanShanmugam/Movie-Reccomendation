import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def prepare_svd_matrix(user_item_matrix: pd.DataFrame):
    """
    Fill missing values with user mean, then center by user mean.
    Returns:
        filled_matrix_df
        centered_matrix
        user_means
    """
    filled_matrix = user_item_matrix.copy()

    # Mean rating for each user
    user_means = filled_matrix.mean(axis=1)

    # Fill missing ratings with each user's mean rating
    filled_matrix = filled_matrix.apply(
        lambda row: row.fillna(row.mean()),
        axis=1
    )

    # If a user has all NaN values, fill with global mean fallback
    global_mean = user_item_matrix.stack().mean()
    filled_matrix = filled_matrix.fillna(global_mean)

    # Center rows by subtracting user mean
    centered_matrix = filled_matrix.sub(user_means.fillna(global_mean), axis=0)

    return filled_matrix, centered_matrix.values, user_means.fillna(global_mean)


def compute_svd(centered_matrix: np.ndarray, k: int = 20):
    """
    Perform truncated SVD using numpy.
    """
    U, s, Vt = np.linalg.svd(centered_matrix, full_matrices=False)

    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    sigma_k = np.diag(s_k)

    return U_k, sigma_k, Vt_k


def reconstruct_ratings(
    U_k: np.ndarray,
    sigma_k: np.ndarray,
    Vt_k: np.ndarray,
    user_means: pd.Series
) -> np.ndarray:
    """
    Reconstruct approximate rating matrix and add back user means.
    """
    reconstructed_centered = U_k @ sigma_k @ Vt_k
    reconstructed = reconstructed_centered + user_means.values.reshape(-1, 1)

    # Clip to MovieLens rating scale
    reconstructed = np.clip(reconstructed, 0.5, 5.0)

    return reconstructed


def recommend_svd(
    target_user: int,
    user_item_matrix: pd.DataFrame,
    reconstructed_matrix: np.ndarray,
    movies: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Recommend top-N unseen movies for a user from reconstructed SVD ratings.
    """
    if target_user not in user_item_matrix.index:
        raise ValueError(f"User {target_user} not found.")

    user_idx = user_item_matrix.index.get_loc(target_user)
    predicted_ratings = reconstructed_matrix[user_idx]

    pred_df = pd.DataFrame({
        "movieId": user_item_matrix.columns,
        "predicted_rating": predicted_ratings
    })

    # Remove already rated movies
    rated_movies = user_item_matrix.loc[target_user].dropna().index.tolist()
    pred_df = pred_df[~pred_df["movieId"].isin(rated_movies)]

    pred_df = pred_df.sort_values(by="predicted_rating", ascending=False).head(top_n)
    result = pred_df.merge(movies, on="movieId", how="left")

    return result[["movieId", "title", "genres", "predicted_rating"]].reset_index(drop=True)


def evaluate_rmse_svd(
    ratings: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    reconstructed_matrix: np.ndarray,
    sample_size: int = 1000
) -> float:
    """
    Evaluate RMSE on a sample of known ratings.
    """
    sample = ratings.sample(n=min(sample_size, len(ratings)), random_state=42)

    y_true = []
    y_pred = []

    user_index_map = {uid: idx for idx, uid in enumerate(user_item_matrix.index)}
    movie_index_map = {mid: idx for idx, mid in enumerate(user_item_matrix.columns)}

    for _, row in sample.iterrows():
        user_id = row["userId"]
        movie_id = row["movieId"]
        true_rating = row["rating"]

        if user_id in user_index_map and movie_id in movie_index_map:
            uidx = user_index_map[user_id]
            midx = movie_index_map[movie_id]
            pred = reconstructed_matrix[uidx, midx]

            y_true.append(true_rating)
            y_pred.append(pred)

    if len(y_true) == 0:
        return np.nan

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(rmse)


def precision_recall_at_k_svd(
    target_user: int,
    ratings: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    reconstructed_matrix: np.ndarray,
    movies: pd.DataFrame,
    top_k: int = 10,
    relevant_threshold: float = 4.0
):
    """
    Compute Precision@K and Recall@K for SVD recommender.
    """
    user_true = ratings[ratings["userId"] == target_user]
    relevant_items = set(user_true[user_true["rating"] >= relevant_threshold]["movieId"].tolist())

    if len(relevant_items) == 0:
        return 0.0, 0.0

    recs = recommend_svd(
        target_user=target_user,
        user_item_matrix=user_item_matrix,
        reconstructed_matrix=reconstructed_matrix,
        movies=movies,
        top_n=top_k
    )

    recommended_items = set(recs["movieId"].tolist())
    true_positives = len(recommended_items.intersection(relevant_items))

    precision = true_positives / top_k if top_k > 0 else 0.0
    recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0.0

    return precision, recall