from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_core_imports():
    from src.data_loader import load_movielens_data
    from src.preprocess import preprocess_data
    from src.content_based import build_tfidf_similarity
    from src.collaborative import build_user_item_matrix
    from src.surprise_model import load_surprise_data

    assert callable(load_movielens_data)
    assert callable(preprocess_data)
    assert callable(build_tfidf_similarity)
    assert callable(build_user_item_matrix)
    assert callable(load_surprise_data)


def test_project_structure():
    assert Path("main.py").is_file()
    assert Path("src").is_dir()
    assert Path("tests").is_dir()
    assert Path("requirements.txt").is_file()