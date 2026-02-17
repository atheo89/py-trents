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
    # Agentic / LLM Ecosystem
    # =========================
    "huggingface_hub": "huggingface-hub",
    "openai": "openai",
    "anthropic": "anthropic",
    "diffusers": "diffusers",
    "peft": "peft",
    "trl": "trl",
    "langchain": "langchain",
    "langchain_core": "langchain-core",
    "langchain_community": "langchain-community",
    "langchain_openai": "langchain-openai",
    "langchain_google_genai": "langchain-google-genai",
    "langchain_anthropic": "langchain-anthropic",
    "langchain_ollama": "langchain-ollama",
    "langchain_chroma": "langchain-chroma",
    "langgraph": "langgraph",
    "llama_index": "llama-index",
    "crewai": "crewai",
    "autogen": "autogen",
    "smolagents": "smolagents",
    "haystack": "haystack",
    "dspy": "dspy",
    "langfuse": "langfuse",
    "chromadb": "chromadb",
    "faiss": "faiss",
    "google.generativeai": "google-generativeai",
    "unsloth": "unsloth",
    "evaluate": "evaluate",

    # =========================
    # Computer Vision
    # =========================
    "cv2": "opencv-python",
    "PIL": "pillow",
    "pil": "pillow",
    "skimage": "scikit-image",
    "albumentations": "albumentations",
    "timm": "timm",
    "torch_geometric": "torch-geometric",
    "onnxruntime": "onnxruntime",
    "imageio": "imageio",
    "moviepy": "moviepy",
    "segment_anything": "segment-anything",
    "ultralytics": "ultralytics",

    # =========================
    # Statistics / Econometrics
    # =========================
    "statsmodels": "statsmodels",
    "statsmodels.api": "statsmodels",
    "statsmodels.tsa": "statsmodels",
    "pymc": "pymc",
    "arviz": "arviz",
    "prophet": "prophet",
    "arch": "arch",
    "pennylane": "pennylane",

    # =========================
    # Model Interpretation
    # =========================
    "shap": "shap",
    "lime": "lime",
    "yellowbrick": "yellowbrick",

    # =========================
    # Hyperparameter Optimization
    # =========================
    "optuna": "optuna",
    "hyperopt": "hyperopt",
    "ray.tune": "ray",
    "kerastuner": "kerastuner",

    # =========================
    # Distributed / Big Data
    # =========================
    "dask": "dask",
    "dask.dataframe": "dask",
    "pyspark": "pyspark",
    "polars": "polars",
    "geopandas": "geopandas",
    "shapely": "shapely",
    "networkx": "networkx",
    "sympy": "sympy",
    "h5py": "h5py",
    "yfinance": "yfinance",
    "rdkit": "rdkit",
    "biotite": "biotite",
    "bio": "biopython",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "mlxtend": "mlxtend",
    "category_encoders": "category-encoders",
    "tensorflow_decision_forests": "tensorflow-decision-forests",
    "einops": "einops",
    "bokeh": "bokeh",
    "folium": "folium",

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
    }
    excluded_packages: set[str] = {
        "__future__",
        "addit_flux_pipeline",
        "addit_flux_transformer",
        "addit_methods",
        "addit_scheduler",
        "app",
        "argparse",
        "ast",
        "asyncio",
        "builtins",
        "calendar",
        "charset_normalizer",
        "colorama",
        "colab",
        "concurrent",
        "condacolab",
        "config",
        "configs",
        "collections",
        "contextlib",
        "csv",
        "learntools",
        "copy",
        "json",
        "uuid",
        "base64",
        "dataclasses",
        "datetime",
        "email",
        "enum",
        "functools",
        "gc",
        "general_utils",
        "getpass",
        "glob",
        "google",
        "gzip",
        "hashlib",
        "heapq",
        "html",
        "importlib",
        "inspect",
        "io",
        "ipywidgets",
        "ipython",
        "itertools",
        "jupyter_server",
        "jupytergis",
        "kaggle",
        "kaggle_environments",
        "kagglehub",
        "kaggle_datasets",
        "kaggle_secrets",
        "kaggle_benchmarks",
        "kaggle_evaluation",
        "locale",
        "logging",
        "math",
        "markdownify",
        "models",
        "multiprocessing",
        "omegaconf",
        "operator",
        "opentelemetry",
        "openinference",
        "orjson",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "pprint",
        "pydantic",
        "pyngrok",
        "psutil",
        "queue",
        "random",
        "re",
        "runpy",
        "runner",
        "shutil",
        "socket",
        "src",
        "sqlite3",
        "stat",
        "statistics",
        "string",
        "subprocess",
        "sys",
        "tarfile",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "traceback",
        "tracemalloc",
        "types",
        "typing",
        "typing_extensions",
        "unicodedata",
        "urllib",
        "utils",
        "evaluation_utils",
        "xml",
        "wave",
        "warnings",
        "zipfile",
        "zlib",
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
