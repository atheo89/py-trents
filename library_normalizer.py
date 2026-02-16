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
        "pil": "pillow",
        "catboost.core": "catboost",
        "cv2": "opencv",
        "keras": "tensorflow",
        "lightgbm.sklearn": "lightgbm",
        "matplotlib.pyplot": "matplotlib",
        "matplotlib": "matplotlib",
        "numpy.random": "numpy",
        "pandas.io": "pandas",
        "pandas.tseries": "pandas",
        "plotly.express": "plotly",
        "plotly.graph_objects": "plotly",
        "pytorch_lightning": "pytorch",
        "scipy.stats": "scipy",
        "skimage": "scikit-image",
        "sklearn": "scikit-learn",
        "sklearn.metrics": "scikit-learn",
        "sklearn.model_selection": "scikit-learn",
        "sklearn.preprocessing": "scikit-learn",
        "statsmodels.api": "statsmodels",
        "statsmodels.tsa": "statsmodels",
        "tensorflow.compat": "tensorflow",
        "tensorflow.keras": "tensorflow",
        "torch": "pytorch",
        "torchvision": "pytorch",
        "torch.nn": "pytorch",
        "xgboost.sklearn": "xgboost",
    }
    excluded_packages: set[str] = {
        "argparse",
        "collections",
        "learntools",
        "copy",
        "uuid",
        "base64",
        "dataclasses",
        "datetime",
        "functools",
        "glob",
        "json",
        "kaggle",
        "kagglehub",
        "kaggle_datasets",   
        "kaggle_secrets",
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
