from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LibraryNormalizerConfig:
    prefix_mapping: Mapping[str, str]
    excluded_packages: set[str]


@dataclass(frozen=True)
class NormalizeImportsRequest:
    raw_imports: set[str]
    config: LibraryNormalizerConfig


@dataclass(frozen=True)
class NormalizeImportsResult:
    normalized_imports: set[str]


@dataclass(frozen=True)
class NormalizeImportRequest:
    raw_import: str
    ordered_prefixes: Sequence[str]
    prefix_mapping: Mapping[str, str]
    excluded_packages: set[str]


def default_normalizer_config() -> LibraryNormalizerConfig:
    mapping: dict[str, str] = {
    # =========================
    # Core Scientific Stack
    # =========================
    "numpy": "numpy",
    "numpy.random": "numpy",
    "scipy": "scipy",
    "scipy.stats": "scipy",
    "scipy.sparse": "scipy",
    "pandas": "pandas",
    "pandas.io": "pandas",
    "pandas.tseries": "pandas",
    "matplotlib": "matplotlib",
    "matplotlib.pyplot": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "plotly.express": "plotly",
    "plotly.graph_objects": "plotly",

    # =========================
    # Machine Learning
    # =========================
    "sklearn": "scikit-learn",
    "sklearn.metrics": "scikit-learn",
    "sklearn.model_selection": "scikit-learn",
    "sklearn.preprocessing": "scikit-learn",
    "sklearn.pipeline": "scikit-learn",
    "sklearn.compose": "scikit-learn",

    "xgboost": "xgboost",
    "xgboost.sklearn": "xgboost",
    "lightgbm": "lightgbm",
    "lightgbm.sklearn": "lightgbm",
    "catboost": "catboost",
    "catboost.core": "catboost",

    "imblearn": "imbalanced-learn",

    # =========================
    # Deep Learning
    # =========================
    "torch": "torch",
    "torch.nn": "torch",
    "torch.optim": "torch",
    "torch.utils.data": "torch",

    "torchvision": "torchvision",
    "torchvision.transforms": "torchvision",
    "torchvision.datasets": "torchvision",

    "torchaudio": "torchaudio",
    "torchtext": "torchtext",

    "pytorch_lightning": "pytorch-lightning",
    "torchmetrics": "torchmetrics",

    "tensorflow": "tensorflow",
    "tensorflow.keras": "tensorflow",
    "tensorflow.compat": "tensorflow",

    "keras": "keras",

    # =========================
    # NLP
    # =========================
    "transformers": "transformers",
    "datasets": "datasets",
    "tokenizers": "tokenizers",
    "spacy": "spacy",
    "nltk": "nltk",
    "sentence_transformers": "sentence-transformers",

    # =========================
    # Computer Vision
    # =========================
    "cv2": "opencv-python",
    "PIL": "pillow",
    "pil": "pillow",
    "skimage": "scikit-image",
    "albumentations": "albumentations",

    # =========================
    # Statistics / Econometrics
    # =========================
    "statsmodels": "statsmodels",
    "statsmodels.api": "statsmodels",
    "statsmodels.tsa": "statsmodels",
    "pymc": "pymc",
    "arviz": "arviz",

    # =========================
    # Model Interpretation
    # =========================
    "shap": "shap",
    "lime": "lime",

    # =========================
    # Hyperparameter Optimization
    # =========================
    "optuna": "optuna",
    "hyperopt": "hyperopt",
    "ray.tune": "ray",

    # =========================
    # Distributed / Big Data
    # =========================
    "dask": "dask",
    "dask.dataframe": "dask",
    "pyspark": "pyspark",

    # =========================
    # Experiment Tracking
    # =========================
    "mlflow": "mlflow",
    "wandb": "wandb",
    "tensorboard": "tensorboard",

    # =========================
    # API / Deployment
    # =========================
    "fastapi": "fastapi",
    "flask": "flask",
    "uvicorn": "uvicorn",
    "gradio": "gradio",
    "streamlit": "streamlit",

    # =========================
    # Utilities
    # =========================
    "tqdm": "tqdm",
    "joblib": "joblib",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "requests": "requests",
    "beautifulsoup4": "beautifulsoup4",
    "bs4": "beautifulsoup4",

    # =========================
    # Google Ecosystem
    # =========================
    "google.colab": "google-colab",
    "google.generativeai": "google-generativeai",
    }
    excluded_packages: set[str] = {
        "argparse",
        "colab",
        "collections",
        "learntools",
        "copy",
        "json",
        "uuid",
        "base64",
        "dataclasses",
        "datetime",
        "functools",
        "gc",
        "glob",
        "io",
        "ipython",
        "json",
        "kaggle",
        "kagglehub",
        "kaggle_datasets",   
        "kaggle_secrets",
        "kaggle_benchmarks",
        "kaggle_evaluation",
        "logging",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "random",
        "re",
        "shutil",
        "statistics",
        "subprocess",
        "sys",
        "tempfile",
        "textwrap",
        "time",
        "typing",
        "utils",
        "warnings",
    }
    return LibraryNormalizerConfig(prefix_mapping=mapping, excluded_packages=excluded_packages)


def normalize_import(request: NormalizeImportRequest) -> str:
    raw_import: str = request.raw_import.strip().lower()
    if raw_import == "":
        return ""
    top_level: str = raw_import.split(".")[0]
    if top_level in request.excluded_packages:
        return ""
    for prefix in request.ordered_prefixes:
        if raw_import == prefix or raw_import.startswith(f"{prefix}."):
            canonical_name: str = request.prefix_mapping[prefix]
            if canonical_name in request.excluded_packages:
                return ""
            return canonical_name
    return top_level


def normalize_imports(request: NormalizeImportsRequest) -> NormalizeImportsResult:
    ordered_prefixes: list[str] = sorted(request.config.prefix_mapping.keys(), key=len, reverse=True)
    normalized_imports: set[str] = set()
    for raw_import in request.raw_imports:
        canonical_name: str = normalize_import(NormalizeImportRequest(raw_import=raw_import, ordered_prefixes=ordered_prefixes, prefix_mapping=request.config.prefix_mapping, excluded_packages=request.config.excluded_packages))
        if canonical_name != "":
            normalized_imports.add(canonical_name)
    return NormalizeImportsResult(normalized_imports=normalized_imports)
