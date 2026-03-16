import numpy as np
import pandas as pd


def load_surprise_data(ratings: pd.DataFrame):
    """
    Convert pandas ratings dataframe into Surprise dataset.
    """
    from surprise import Dataset, Reader

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader
    )
    return data


def train_surprise_svd(
    ratings: pd.DataFrame,
    n_factors: int = 50,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train Surprise SVD and evaluate RMSE on a held-out test set.
    """
    from surprise import SVD
    from surprise.model_selection import train_test_split
    from surprise.accuracy import rmse

    data = load_surprise_data(ratings)
    trainset, testset = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
    )

    model.fit(trainset)
    predictions = model.test(testset)
    test_rmse = rmse(predictions, verbose=False)

    return model, trainset, testset, predictions, float(test_rmse)


def get_surprise_top_n(
    model,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Recommend top-N unseen movies for a user using Surprise SVD.
    """
    rated_movie_ids = set(
        ratings.loc[ratings["userId"] == user_id, "movieId"].tolist()
    )

    all_movie_ids = movies["movieId"].tolist()

    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movie_ids:
            pred = model.predict(uid=user_id, iid=movie_id)
            predictions.append((movie_id, pred.est))

    pred_df = pd.DataFrame(predictions, columns=["movieId", "predicted_rating"])
    pred_df = pred_df.sort_values(by="predicted_rating", ascending=False).head(top_n)

    result = pred_df.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    return result[["movieId", "title", "genres", "predicted_rating"]].reset_index(drop=True)


def precision_recall_at_k_surprise(
    predictions,
    k: int = 10,
    threshold: float = 4.0
):
    """
    Compute average Precision@K and Recall@K across users
    from Surprise prediction objects.
    """
    from collections import defaultdict

    user_est_true = defaultdict(list)

    for pred in predictions:
        uid = pred.uid
        est = pred.est
        true_r = pred.r_ui
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        top_k = user_ratings[:k]
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in top_k
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0.0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0.0

    avg_precision = np.mean(list(precisions.values())) if precisions else 0.0
    avg_recall = np.mean(list(recalls.values())) if recalls else 0.0

    return float(avg_precision), float(avg_recall)