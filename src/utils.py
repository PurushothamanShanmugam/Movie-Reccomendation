from pathlib import Path


def ensure_directory(path: Path) -> None:
    """
    Create a directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def ensure_project_directories(paths: list[Path]) -> None:
    """
    Create multiple project directories.
    """
    for path in paths:
        ensure_directory(path)


def print_header(title: str) -> None:
    """
    Print a clean header in terminal output.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)