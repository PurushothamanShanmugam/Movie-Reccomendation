import json
import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    RECOMMENDATIONS_DIR,
)
from src.data_loader import load_movielens_data
from src.preprocess import preprocess_data

from src.content_based import (
    build_tfidf_similarity,
    get_content_recommendations,
    get_user_profile_recommendations,
    precision_recall_at_k_content,
)

from src.collaborative import (
    build_user_item_matrix,
    compute_user_similarity,
    compute_item_similarity,
    recommend_user_based,
    recommend_item_based,
    evaluate_rmse_user_based,
    evaluate_rmse_item_based,
    precision_recall_at_k_user_based,
    precision_recall_at_k_item_based,
)

from src.matrix_factorization import (
    prepare_svd_matrix,
    compute_svd,
    reconstruct_ratings,
    recommend_svd,
    evaluate_rmse_svd,
    precision_recall_at_k_svd,
)

from src.hybrid_model import (
    recommend_hybrid,
    evaluate_rmse_hybrid,
    precision_recall_at_k_hybrid,
)

from src.deep_model import (
    train_neural_recommender,
    recommend_neural_model,
    precision_recall_at_k_neural,
)

from src.rl_recommender import (
    train_q_learning,
    recommend_rl,
    evaluate_rl_rmse,
)

from src.surprise_model import(
    load_surprise_data,
    train_surprise_svd,
    get_surprise_top_n,
    precision_recall_at_k_surprise,
)

MODELS_DIR = OUTPUT_DIR / "models"


def print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def ensure_directories():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def safe_len(df):
    return int(len(df)) if isinstance(df, pd.DataFrame) else 0


def save_recommendation_plot(df: pd.DataFrame, title: str, score_col: str, out_path: Path):
    if df is None or df.empty or score_col not in df.columns:
        return

    plot_df = df.head(10).copy()
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["title"].astype(str), plot_df[score_col].astype(float))
    plt.title(title)
    plt.xlabel(score_col)
    plt.ylabel("Movie")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_json(data: dict, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_pickle(obj, out_path: Path):
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)


def main():
    print_header("Movie Recommendation Project")

    ensure_directories()

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    movies, ratings, tags, links = load_movielens_data()

    print("\nDataset loaded successfully.\n")
    print("Movies shape  :", movies.shape)
    print("Ratings shape :", ratings.shape)
    print("Tags shape    :", tags.shape)
    print("Links shape   :", links.shape)

    sample_title = "Toy Story (1995)"
    sample_user_id = 1
    top_n = 10

    metrics = {
        "dataset_summary": {
            "movies_rows": int(movies.shape[0]),
            "movies_cols": int(movies.shape[1]),
            "ratings_rows": int(ratings.shape[0]),
            "ratings_cols": int(ratings.shape[1]),
            "tags_rows": int(tags.shape[0]),
            "links_rows": int(links.shape[0]),
            "unique_users": int(ratings["userId"].nunique()),
            "unique_movies_rated": int(ratings["movieId"].nunique()),
            "average_rating": float(ratings["rating"].mean()),
        }
    }

    # --------------------------------------------------
    # Preprocess
    # --------------------------------------------------
    print_header("Preprocessing Data")
    processed_data = preprocess_data(movies, ratings)

    if isinstance(processed_data, pd.DataFrame):
        processed_out = PROCESSED_DATA_DIR / "processed_movies_ratings.csv"
        processed_data.to_csv(processed_out, index=False)
        print(f"Processed dataset saved to: {processed_out}")
        metrics["processed_data"] = {
            "rows": int(processed_data.shape[0]),
            "cols": int(processed_data.shape[1]),
        }
    else:
        print("Warning: preprocess_data() did not return a DataFrame.")
        metrics["processed_data"] = {
            "rows": 0,
            "cols": 0,
            "note": "preprocess_data did not return a DataFrame"
        }

    # --------------------------------------------------
    # Content-based
    # --------------------------------------------------
    print_header("Content-Based Recommendation")

    movies_cb, tfidf_matrix, similarity_matrix = build_tfidf_similarity(movies)

    movie_recs = get_content_recommendations(
        title=sample_title,
        movies=movies_cb,
        similarity_matrix=similarity_matrix,
        top_n=top_n,
    )

    print(movie_recs)

    content_precision, content_recall = precision_recall_at_k_content(
        user_id=sample_user_id,
        ratings=ratings,
        movies=movies_cb,
        tfidf_matrix=tfidf_matrix,
        k=top_n,
    )

    metrics["content_based"] = {
        "recommendation_count": safe_len(movie_recs),
        "precision_at_10": float(content_precision),
        "recall_at_10": float(content_recall),
    }

    # save content artifacts
    movie_recs.to_csv(RECOMMENDATIONS_DIR / "content_recommendations.csv", index=False)
    save_recommendation_plot(
        movie_recs,
        "Content-Based Recommendations",
        "similarity_score",
        FIGURES_DIR / "content_based.png",
    )
    save_pickle(tfidf_matrix, MODELS_DIR / "content_tfidf_matrix.pkl")
    np.save(MODELS_DIR / "content_similarity_matrix.npy", similarity_matrix)

    # --------------------------------------------------
    # User profile content
    # --------------------------------------------------
    print_header("User Profile Content Recommendation")

    profile_recs = get_user_profile_recommendations(
        user_id=sample_user_id,
        ratings=ratings,
        movies=movies_cb,
        tfidf_matrix=tfidf_matrix,
        top_n=top_n,
    )

    print(profile_recs)

    profile_recs.to_csv(RECOMMENDATIONS_DIR / "profile_recommendations.csv", index=False)
    save_recommendation_plot(
        profile_recs,
        "User Profile Content Recommendations",
        "similarity_score",
        FIGURES_DIR / "profile_content.png",
    )

    metrics["user_profile_content"] = {
        "recommendation_count": safe_len(profile_recs),
    }

    # --------------------------------------------------
    # Collaborative filtering
    # --------------------------------------------------
    print_header("Collaborative Filtering")

    user_item_matrix = build_user_item_matrix(ratings)
    user_similarity = compute_user_similarity(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)

    user_cf_recs = recommend_user_based(
        target_user=sample_user_id,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
        movies=movies,
        top_n=top_n,
    )

    item_cf_recs = recommend_item_based(
        target_user=sample_user_id,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
        movies=movies,
        top_n=top_n,
    )

    print("\nUser-based CF recommendations:")
    print(user_cf_recs)

    print("\nItem-based CF recommendations:")
    print(item_cf_recs)

    user_cf_rmse = evaluate_rmse_user_based(
        ratings=ratings,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
    )

    item_cf_rmse = evaluate_rmse_item_based(
        ratings=ratings,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
    )

    user_cf_precision, user_cf_recall = precision_recall_at_k_user_based(
        target_user=sample_user_id,
        ratings=ratings,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
        movies=movies,
        top_k=top_n,
    )

    item_cf_precision, item_cf_recall = precision_recall_at_k_item_based(
        target_user=sample_user_id,
        ratings=ratings,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
        movies=movies,
        top_k=top_n,
    )

    metrics["user_based_cf"] = {
        "rmse": float(user_cf_rmse) if not np.isnan(user_cf_rmse) else None,
        "precision_at_10": float(user_cf_precision),
        "recall_at_10": float(user_cf_recall),
        "recommendation_count": safe_len(user_cf_recs),
    }

    metrics["item_based_cf"] = {
        "rmse": float(item_cf_rmse) if not np.isnan(item_cf_rmse) else None,
        "precision_at_10": float(item_cf_precision),
        "recall_at_10": float(item_cf_recall),
        "recommendation_count": safe_len(item_cf_recs),
    }

    user_cf_recs.to_csv(RECOMMENDATIONS_DIR / "user_cf_recommendations.csv", index=False)
    item_cf_recs.to_csv(RECOMMENDATIONS_DIR / "item_cf_recommendations.csv", index=False)

    save_recommendation_plot(
        user_cf_recs,
        "User-Based Collaborative Filtering",
        "predicted_rating",
        FIGURES_DIR / "user_cf.png",
    )
    save_recommendation_plot(
        item_cf_recs,
        "Item-Based Collaborative Filtering",
        "predicted_rating",
        FIGURES_DIR / "item_cf.png",
    )

    joblib.dump(user_item_matrix, MODELS_DIR / "user_item_matrix.pkl")
    joblib.dump(user_similarity, MODELS_DIR / "user_similarity.pkl")
    joblib.dump(item_similarity, MODELS_DIR / "item_similarity.pkl")

    # --------------------------------------------------
    # SVD
    # --------------------------------------------------
    print_header("Matrix Factorization (SVD)")

    filled_matrix, centered_matrix, user_means = prepare_svd_matrix(user_item_matrix)
    U_k, sigma_k, Vt_k = compute_svd(centered_matrix, k=20)
    reconstructed_matrix = reconstruct_ratings(U_k, sigma_k, Vt_k, user_means)

    svd_recs = recommend_svd(
        target_user=sample_user_id,
        user_item_matrix=user_item_matrix,
        reconstructed_matrix=reconstructed_matrix,
        movies=movies,
        top_n=top_n,
    )

    print(svd_recs)

    svd_rmse = evaluate_rmse_svd(
        ratings=ratings,
        user_item_matrix=user_item_matrix,
        reconstructed_matrix=reconstructed_matrix,
    )

    svd_precision, svd_recall = precision_recall_at_k_svd(
        target_user=sample_user_id,
        ratings=ratings,
        user_item_matrix=user_item_matrix,
        reconstructed_matrix=reconstructed_matrix,
        movies=movies,
        top_k=top_n,
    )

    metrics["svd"] = {
        "rmse": float(svd_rmse) if not np.isnan(svd_rmse) else None,
        "precision_at_10": float(svd_precision),
        "recall_at_10": float(svd_recall),
        "recommendation_count": safe_len(svd_recs),
        "latent_factors": 20,
    }

    svd_recs.to_csv(RECOMMENDATIONS_DIR / "svd_recommendations.csv", index=False)
    save_recommendation_plot(
        svd_recs,
        "SVD Recommendations",
        "predicted_rating",
        FIGURES_DIR / "svd.png",
    )

    np.savez(
        MODELS_DIR / "svd_artifacts.npz",
        U_k=U_k,
        sigma_k=np.diag(sigma_k) if sigma_k.ndim == 2 else sigma_k,
        Vt_k=Vt_k,
        reconstructed_matrix=reconstructed_matrix,
    )
    joblib.dump(user_means, MODELS_DIR / "svd_user_means.pkl")

    # --------------------------------------------------
    # Hybrid
    # --------------------------------------------------
    print_header("Hybrid Recommendation")

    hybrid_recs = recommend_hybrid(
        user_id=sample_user_id,
        ratings=ratings,
        movies=movies_cb,
        tfidf_matrix=tfidf_matrix,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
        top_n=top_n,
    )

    print(hybrid_recs)

    hybrid_rmse = evaluate_rmse_hybrid(
        ratings=ratings,
        movies=movies_cb,
        tfidf_matrix=tfidf_matrix,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
    )

    hybrid_precision, hybrid_recall = precision_recall_at_k_hybrid(
        user_id=sample_user_id,
        ratings=ratings,
        movies=movies_cb,
        tfidf_matrix=tfidf_matrix,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity,
        top_k=top_n,
    )

    metrics["hybrid"] = {
        "rmse": float(hybrid_rmse) if not np.isnan(hybrid_rmse) else None,
        "precision_at_10": float(hybrid_precision),
        "recall_at_10": float(hybrid_recall),
        "recommendation_count": safe_len(hybrid_recs),
    }

    hybrid_recs.to_csv(RECOMMENDATIONS_DIR / "hybrid_recommendations.csv", index=False)
    save_recommendation_plot(
        hybrid_recs,
        "Hybrid Recommendations",
        "hybrid_score",
        FIGURES_DIR / "hybrid.png",
    )

    # --------------------------------------------------
    # Neural model
    # --------------------------------------------------
    print_header("Neural Network Recommendation")

    neural_artifacts = train_neural_recommender(
        ratings=ratings,
        movies=movies,
        epochs=3,
        batch_size=256,
    )

    neural_recs = recommend_neural_model(
        user_id=sample_user_id,
        ratings=ratings,
        movies=movies,
        artifacts=neural_artifacts,
        top_n=top_n,
    )

    print(neural_recs)

    neural_precision, neural_recall = precision_recall_at_k_neural(
        user_id=sample_user_id,
        ratings=ratings,
        movies=movies,
        artifacts=neural_artifacts,
        top_k=top_n,
    )

    metrics["neural"] = {
        "rmse": float(neural_artifacts["rmse"]),
        "precision_at_10": float(neural_precision),
        "recall_at_10": float(neural_recall),
        "recommendation_count": safe_len(neural_recs),
    }

    neural_recs.to_csv(RECOMMENDATIONS_DIR / "neural_recommendations.csv", index=False)
    save_recommendation_plot(
        neural_recs,
        "Neural Recommendations",
        "predicted_rating",
        FIGURES_DIR / "neural.png",
    )

    neural_artifacts["model"].save(MODELS_DIR / "neural_model.keras")
    joblib.dump(neural_artifacts["movie_features"], MODELS_DIR / "neural_movie_features.pkl")
    joblib.dump(neural_artifacts["user_features"], MODELS_DIR / "neural_user_features.pkl")
    save_json(
        {
            "genre_columns": neural_artifacts["genre_columns"],
            "user_feature_cols": neural_artifacts["user_feature_cols"],
            "movie_feature_cols": neural_artifacts["movie_feature_cols"],
            "rmse": neural_artifacts["rmse"],
        },
        MODELS_DIR / "neural_metadata.json",
    )

    # --------------------------------------------------
    # Reinforcement learning
    # --------------------------------------------------
    print_header("Reinforcement Learning Recommendation")

    q_table = train_q_learning(ratings)

    rl_recs = recommend_rl(
        user_id=sample_user_id,
        q_table=q_table,
        movies=movies,
        ratings=ratings,
        top_n=top_n,
    )

    print(rl_recs)

    rl_rmse = evaluate_rl_rmse(
        ratings=ratings,
        q_table=q_table,
    )

    metrics["reinforcement_learning"] = {
        "rmse": float(rl_rmse) if not np.isnan(rl_rmse) else None,
        "recommendation_count": safe_len(rl_recs),
    }

    rl_recs.to_csv(RECOMMENDATIONS_DIR / "rl_recommendations.csv", index=False)
    save_recommendation_plot(
        rl_recs,
        "RL Recommendations",
        "q_score",
        FIGURES_DIR / "rl.png",
    )

    joblib.dump(q_table, MODELS_DIR / "q_table.pkl")

    # -------------------------------------------------
    # Surprise Library SVD (Task 6)
    # -------------------------------------------------
    print_header("Building Surprise Library SVD")

    surprise_model, trainset, testset, predictions, surprise_rmse = train_surprise_svd(
        ratings=ratings,
        n_factors=50,
        n_epochs=20,
    )

    print("Surprise SVD model trained successfully.")

    print_header(f"Top Surprise Recommendations for User: {sample_user_id}")

    surprise_recs = get_surprise_top_n(
        model=surprise_model,
        ratings=ratings,
        movies=movies,
        user_id=sample_user_id,
        top_n=10,
    )

    print(surprise_recs)

    precision_surprise, recall_surprise = precision_recall_at_k_surprise(
        predictions=predictions,
        k=10,
        threshold=4.0,
    )

    print_header("Evaluation for Surprise SVD")
    print(f"RMSE        : {surprise_rmse:.4f}")
    print(f"Precision@10: {precision_surprise:.4f}")
    print(f"Recall@10   : {recall_surprise:.4f}")

    # --------------------------------------------------
    # Save metrics summary
    # --------------------------------------------------
    print_header("Saving Metrics")

    save_json(metrics, METRICS_DIR / "model_metrics.json")

    metrics_rows = []
    for model_name, values in metrics.items():
        if isinstance(values, dict):
            row = {"model": model_name}
            row.update(values)
            metrics_rows.append(row)

    pd.DataFrame(metrics_rows).to_csv(METRICS_DIR / "model_metrics.csv", index=False)

    print("All outputs saved successfully.")
    print(f"Processed data folder      : {PROCESSED_DATA_DIR}")
    print(f"Recommendations folder    : {RECOMMENDATIONS_DIR}")
    print(f"Figures folder            : {FIGURES_DIR}")
    print(f"Metrics folder            : {METRICS_DIR}")
    print(f"Models folder             : {MODELS_DIR}")


if __name__ == "__main__":
    main()