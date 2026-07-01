"""Dataset loading utilities."""

import json
from pathlib import Path

import yaml

from oracle_flexcube_copilot.evaluation.models import EvalQuery


def load_dataset(path: str | Path) -> list[EvalQuery]:
    """Load an evaluation dataset from a JSON or YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    text = path.read_text("utf-8")

    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of queries")

    return [EvalQuery(**item) for item in data]
