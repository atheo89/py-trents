from __future__ import annotations

import argparse
import csv
import functools
import re
import sys
from types import ModuleType
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from library_normalizer import NormalizeImportsRequest, default_normalizer_config, normalize_imports
from metrics import FeatureUsage, NotebookFeatureUsage, UsageMetricsRequest, UsageReport, compute_usage_metrics
from notebook_parser import NotebookParseRequest, parse_notebook
from visualization import build_chart_data, ensure_plotly

if TYPE_CHECKING:
    from dash import Dash
    from dash.development.base_component import Component
    from plotly.graph_objects import Figure


DEFAULT_DATA_DIR: str = "data"
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8050
DEFAULT_MAX_ITEMS: int = 20
DEFAULT_MAX_NOTEBOOKS: int = 0
DEFAULT_KAGGLE_SELECTION: str = "top"
DEFAULT_GITHUB_SELECTION: str = "most-starred"
DEFAULT_HF_SELECTION: str = "most-liked"
DEFAULT_HF_SELECTION_ORDER: list[str] = ["most-liked", "most-downloaded", "trending", "most-created_at", "most-last_modified"]
DEFAULT_LIBRARY_TABLE_CSV_FILENAME: str = "complete_libraries_dashboard_table_all_resources.csv"
DEFAULT_CURRENT_PACKAGES_CSV_FILENAME: str = "odh_notebooks_pyproject_dependency_index.csv"
PACKAGE_COMPARISON_TAB_KEY: str = "package-comparison"
PACKAGE_COMPARISON_ADD_NOW_THRESHOLD: float = 1.0
PACKAGE_COMPARISON_WATCHLIST_THRESHOLD: float = 0.28
PACKAGE_COMPARISON_HOVER_PACKAGE_LIMIT: int = 25
PACKAGE_COMPARISON_ACTION_STATUSES: set[str] = {"Add now", "Watchlist", "Review candidate", "Hot (covered)"}
PACKAGE_COMPARISON_STATUS_PRIORITY: dict[str, int] = {
    "Add now": 1,
    "Watchlist": 2,
    "Review candidate": 3,
    "Hot (covered)": 4,
    "Covered (long tail)": 5,
    "Specialized covered": 6,
    "Core/Internal": 7,
    "Long-tail gap": 8,
}
PACKAGE_COMPARISON_STATUS_COLORS: dict[str, str] = {
    "Add now": "#ef4444",
    "Watchlist": "#f97316",
    "Review candidate": "#8b5cf6",
    "Hot (covered)": "#22c55e",
    "Covered (long tail)": "#16a34a",
    "Specialized covered": "#0ea5e9",
    "Core/Internal": "#64748b",
    "Long-tail gap": "#94a3b8",
}
PLATFORM_RUNTIME_PREFIXES: tuple[str, ...] = ("odh-", "jupyter", "ipython", "nb", "kfp", "kubeflow")
PLATFORM_RUNTIME_PACKAGES: set[str] = {
    "debugpy",
    "ipykernel",
    "jinja2",
    "jupyter-client",
    "markupsafe",
    "micropipenv",
    "nvidia-ml-py",
    "opencensus",
    "prometheus-client",
    "prompt-toolkit",
    "py-spy",
    "pyzmq",
    "requests-toolbelt",
    "setuptools",
    "tornado",
    "traitlets",
    "urllib3",
    "uv",
    "virtualenv",
    "wheel",
}
SELECTION_LABEL_OVERRIDES: dict[str, str] = {
    "most-created_at": "Latest",
}
DEFAULT_ALL_OPTION: str = "all"
DEFAULT_ALL_SELECTION: str = "all-sources"
KAGGLE_SELECTION_ORDER: list[str] = ["top", "hotness", "latest"]
FAMILY_DISPLAY_ORDER: list[str] = [
    "Data Science",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Agentic",
    "Computer Vision",
    "Statistics",
    "Model Interpretation",
    "Hyperparameter Optimization",
    "Distributed Computing",
    "Experiment Tracking",
    "Deployment and Apps",
    "Utilities",
    "Other",
]
FAMILY_ORDER_LOOKUP: dict[str, int] = {family: index for index, family in enumerate(FAMILY_DISPLAY_ORDER)}
DEFAULT_FAMILY: str = "Other"
DEFAULT_FAMILY_COLOR: str = "#6b7280"
FAMILY_COLOR_MAPPING: dict[str, str] = {
    "Data Science": "#2563eb",
    "Machine Learning": "#7c3aed",
    "Deep Learning": "#db2777",
    "Natural Language Processing": "#0ea5e9",
    "Agentic": "#0891b2",
    "Computer Vision": "#f97316",
    "Statistics": "#0d9488",
    "Model Interpretation": "#b45309",
    "Hyperparameter Optimization": "#22c55e",
    "Distributed Computing": "#9333ea",
    "Experiment Tracking": "#1d4ed8",
    "Deployment and Apps": "#dc2626",
    "Utilities": "#4b5563",
    "Other": DEFAULT_FAMILY_COLOR,
}
PACKAGE_FAMILY_MAPPING: dict[str, str] = {
    "numpy": "Data Science",
    "scipy": "Data Science",
    "pandas": "Data Science",
    "matplotlib": "Data Science",
    "seaborn": "Data Science",
    "plotly": "Data Science",
    "boto3": "Data Science",
    "scikit-learn": "Machine Learning",
    "xgboost": "Machine Learning",
    "lightgbm": "Machine Learning",
    "catboost": "Machine Learning",
    "vertexai": "Machine Learning",
    "imbalanced-learn": "Machine Learning",
    "torch": "Deep Learning",
    "torchvision": "Deep Learning",
    "torchaudio": "Deep Learning",
    "torchtext": "Deep Learning",
    "pytorch-lightning": "Deep Learning",
    "torchmetrics": "Deep Learning",
    "tensorflow": "Deep Learning",
    "keras": "Deep Learning",
    "transformers": "Natural Language Processing",
    "datasets": "Natural Language Processing",
    "tokenizers": "Natural Language Processing",
    "spacy": "Natural Language Processing",
    "nltk": "Natural Language Processing",
    "sentence-transformers": "Natural Language Processing",
    "evaluate": "Natural Language Processing",
    "langchain": "Agentic",
    "langchain-core": "Agentic",
    "langchain-community": "Agentic",
    "langchain-openai": "Agentic",
    "langchain-anthropic": "Agentic",
    "langchain-ollama": "Agentic",
    "langchain-chroma": "Agentic",
    "langchain-google-genai": "Agentic",
    "langgraph": "Agentic",
    "llama-index": "Agentic",
    "openai": "Agentic",
    "anthropic": "Agentic",
    "huggingface-hub": "Agentic",
    "diffusers": "Agentic",
    "peft": "Agentic",
    "trl": "Agentic",
    "chromadb": "Agentic",
    "faiss": "Agentic",
    "langfuse": "Agentic",
    "google-generativeai": "Agentic",
    "crewai": "Agentic",
    "autogen": "Agentic",
    "semantic-kernel": "Agentic",
    "haystack": "Agentic",
    "dspy": "Agentic",
    "pydantic-ai": "Agentic",
    "smolagents": "Agentic",
    "agno": "Agentic",
    "fastai": "Deep Learning",
    "timm": "Deep Learning",
    "onnxruntime": "Deep Learning",
    "torch-geometric": "Deep Learning",
    "einops": "Deep Learning",
    "unsloth": "Deep Learning",
    "whisper": "Deep Learning",
    "audiocraft": "Deep Learning",
    "segment-anything": "Deep Learning",
    "opencv-python": "Computer Vision",
    "pillow": "Computer Vision",
    "supervision": "Computer Vision",
    "scikit-image": "Computer Vision",
    "albumentations": "Computer Vision",
    "ultralytics": "Computer Vision",
    "statsmodels": "Statistics",
    "pymc": "Statistics",
    "arviz": "Statistics",
    "prophet": "Statistics",
    "arch": "Statistics",
    "shap": "Model Interpretation",
    "lime": "Model Interpretation",
    "yellowbrick": "Model Interpretation",
    "optuna": "Hyperparameter Optimization",
    "hyperopt": "Hyperparameter Optimization",
    "kerastuner": "Hyperparameter Optimization",
    "ray": "Hyperparameter Optimization",
    "dask": "Distributed Computing",
    "pyspark": "Distributed Computing",
    "polars": "Data Science",
    "geopandas": "Data Science",
    "bokeh": "Data Science",
    "folium": "Data Science",
    "shapely": "Data Science",
    "yfinance": "Data Science",
    "rdkit": "Data Science",
    "biotite": "Data Science",
    "biopython": "Data Science",
    "librosa": "Data Science",
    "soundfile": "Data Science",
    "networkx": "Data Science",
    "h5py": "Data Science",
    "mlxtend": "Machine Learning",
    "category-encoders": "Machine Learning",
    "tensorflow-decision-forests": "Machine Learning",
    "mlflow": "Experiment Tracking",
    "wandb": "Experiment Tracking",
    "tensorboard": "Experiment Tracking",
    "fastapi": "Deployment and Apps",
    "flask": "Deployment and Apps",
    "uvicorn": "Deployment and Apps",
    "gradio": "Deployment and Apps",
    "streamlit": "Deployment and Apps",
    "tqdm": "Utilities",
    "joblib": "Utilities",
    "pyyaml": "Utilities",
    "python-dotenv": "Utilities",
    "requests": "Utilities",
    "beautifulsoup4": "Utilities",
    "google-colab": "Utilities",
}
FAMILY_PREFIX_MAPPING: list[tuple[str, str]] = [
    ("openai", "Agentic"),
    ("anthropic", "Agentic"),
    ("huggingface_hub", "Agentic"),
    ("huggingface-hub", "Agentic"),
    ("diffusers", "Agentic"),
    ("peft", "Agentic"),
    ("trl", "Agentic"),
    ("langchain", "Agentic"),
    ("langgraph", "Agentic"),
    ("llama_index", "Agentic"),
    ("llama-index", "Agentic"),
    ("langfuse", "Agentic"),
    ("chromadb", "Agentic"),
    ("faiss", "Agentic"),
    ("crewai", "Agentic"),
    ("autogen", "Agentic"),
    ("semantic_kernel", "Agentic"),
    ("semantic-kernel", "Agentic"),
    ("haystack", "Agentic"),
    ("dspy", "Agentic"),
    ("pydantic_ai", "Agentic"),
    ("pydantic-ai", "Agentic"),
    ("smolagents", "Agentic"),
    ("agno", "Agentic"),
]

APP_STYLE: dict[str, str] = {"fontFamily": "Arial, sans-serif", "padding": "16px"}
CARD_CONTAINER_STYLE: dict[str, str] = {"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginTop": "12px"}
CARD_STYLE: dict[str, str] = {"border": "1px solid #e5e7eb", "borderRadius": "8px", "padding": "12px", "minWidth": "160px", "backgroundColor": "#f9fafb"}
SECTION_STYLE: dict[str, str] = {"marginTop": "16px"}
TABLE_STYLE: dict[str, str] = {"marginTop": "8px"}

LIBRARY_TABLE_COLUMNS: list[dict[str, str]] = [
    {"name": "Name", "id": "name"},
    {"name": "Family", "id": "family"},
    {"name": "Notebook Count", "id": "count"},
    {"name": "Usage %", "id": "percent"},
]
EXTENSION_TABLE_COLUMNS: list[dict[str, str]] = [
    {"name": "Name", "id": "name"},
    {"name": "Notebook Count", "id": "count"},
    {"name": "Usage %", "id": "percent"},
]
PACKAGE_COMPARISON_ACTION_COLUMNS: list[dict[str, str]] = [
    {"name": "Package", "id": "package_name"},
    {"name": "Status", "id": "status"},
    {"name": "Recommendation", "id": "recommendation"},
    {"name": "Family", "id": "family"},
    {"name": "External Usage %", "id": "external_percent"},
    {"name": "External Count", "id": "external_count"},
    {"name": "Flavor Coverage", "id": "flavor_count"},
]
PACKAGE_COMPARISON_CURRENT_COLUMNS: list[dict[str, str]] = [
    {"name": "Row", "id": "row_number"},
    {"name": "Package", "id": "package_name"},
    {"name": "Flavor Count", "id": "flavor_count"},
    {"name": "Flavors", "id": "flavors"},
]
PACKAGE_COMPARISON_EXTERNAL_COLUMNS: list[dict[str, str]] = [
    {"name": "Row", "id": "row_number"},
    {"name": "Package", "id": "name"},
    {"name": "Family", "id": "family"},
    {"name": "Notebook Count", "id": "count"},
    {"name": "Usage %", "id": "percent"},
]


@dataclass(frozen=True)
class DashboardConfig:
    data_dir: Path
    host: str
    port: int
    debug: bool
    max_items: int
    max_notebooks: int


@dataclass(frozen=True)
class ParseDashboardArgsRequest:
    argv: Sequence[str]


@dataclass(frozen=True)
class SourceDefinition:
    key: str
    label: str
    base_dir: Path
    default_selection: str
    selection_label: str
    is_aggregate: bool


@dataclass(frozen=True)
class ResolveSourcesRequest:
    data_dir: Path


@dataclass(frozen=True)
class SelectionOptions:
    options: list[str]
    default_value: str


@dataclass(frozen=True)
class ResolveSelectionOptionsRequest:
    base_dir: Path
    default_selection: str
    is_aggregate: bool
    fallback_options: Sequence[str] | None = None
    include_all: bool = False
    preferred_order: Sequence[str] | None = None


@dataclass(frozen=True)
class ListSelectionDirsRequest:
    base_dir: Path


@dataclass(frozen=True)
class SourceComponentIds:
    selection_dropdown: str
    summary_container: str
    library_chart: str
    family_chart: str
    library_table: str
    extension_table: str
    status_message: str


@dataclass(frozen=True)
class ComponentIdsRequest:
    source_key: str


@dataclass(frozen=True)
class DashModules:
    dcc: ModuleType
    html: ModuleType
    dash_table: ModuleType


@dataclass(frozen=True)
class BuildTabLayoutRequest:
    modules: DashModules
    source: SourceDefinition
    selection_options: SelectionOptions
    ids: SourceComponentIds


@dataclass(frozen=True)
class BuildTabsLayoutRequest:
    modules: DashModules
    sources: list[SourceDefinition]
    selection_options: dict[str, SelectionOptions]
    component_ids: dict[str, SourceComponentIds]
    comparison_tab: Component | None = None


@dataclass(frozen=True)
class BuildPackageComparisonTabRequest:
    modules: DashModules
    config: DashboardConfig


@dataclass(frozen=True)
class BuildAppRequest:
    config: DashboardConfig


@dataclass(frozen=True)
class NotebookPathsRequest:
    selection_dir: Path
    max_notebooks: int


@dataclass(frozen=True)
class NotebookPathsResult:
    notebook_paths: tuple[Path, ...]


@dataclass(frozen=True)
class NotebookUsagesRequest:
    notebook_paths: Sequence[Path]


@dataclass(frozen=True)
class NotebookUsagesResult:
    library_usages: list[NotebookFeatureUsage]
    extension_usages: list[NotebookFeatureUsage]
    skipped_count: int


@dataclass(frozen=True)
class SelectionReportRequest:
    selection_dir: Path
    max_notebooks: int


@dataclass(frozen=True)
class SelectionReport:
    library_report: UsageReport
    extension_report: UsageReport
    analyzed_notebooks: int
    skipped_notebooks: int


@dataclass(frozen=True)
class SummaryMetricsRequest:
    library_report: UsageReport
    extension_report: UsageReport
    skipped_notebooks: int


@dataclass(frozen=True)
class KpiMetric:
    label: str
    value: str


@dataclass(frozen=True)
class CurrentPackageRecord:
    package_name: str
    canonical_name: str
    flavor_count: int
    flavors: str


@dataclass(frozen=True)
class ExternalPackageRecord:
    package_name: str
    canonical_name: str
    family: str
    count: int
    percent: float
    row_number: int


@dataclass(frozen=True)
class PackageComparisonRow:
    package_name: str
    canonical_name: str
    family: str
    status: str
    recommendation: str
    in_current: bool
    in_external: bool
    flavor_count: int
    external_count: int
    external_percent: float
    marker_size: float


@dataclass(frozen=True)
class PackageComparisonDataset:
    current_records: dict[str, CurrentPackageRecord]
    external_records: dict[str, ExternalPackageRecord]
    comparison_rows: list[PackageComparisonRow]
    action_rows: list[dict[str, int | float | str]]
    current_table_rows: list[dict[str, int | float | str]]
    external_table_rows: list[dict[str, int | float | str]]
    metrics: list[KpiMetric]
    status_message: str


@dataclass(frozen=True)
class MetricCardsRequest:
    modules: DashModules
    metrics: list[KpiMetric]


@dataclass(frozen=True)
class UsageTableRequest:
    report: UsageReport
    include_family: bool = False


@dataclass(frozen=True)
class ChartBuildRequest:
    report: UsageReport
    title: str
    x_label: str
    max_items: int


@dataclass(frozen=True)
class FamilyChartBuildRequest:
    report: UsageReport
    title: str
    max_items: int


@dataclass(frozen=True)
class FamilyUsage:
    usage: FeatureUsage
    family: str


@dataclass(frozen=True)
class EmptyFigureRequest:
    message: str


@dataclass(frozen=True)
class RegisterCallbacksRequest:
    app: Dash
    modules: DashModules
    sources: list[SourceDefinition]
    component_ids: dict[str, SourceComponentIds]
    config: DashboardConfig


def ensure_dash() -> None:
    try:
        import dash
    except ImportError as exc:
        raise RuntimeError("Dash is not installed. Run `pip install dash`.") from exc
    _ = dash


def parse_dashboard_args(request: ParseDashboardArgsRequest) -> DashboardConfig:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Dash dashboard for notebook usage.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--max-notebooks", type=int, default=DEFAULT_MAX_NOTEBOOKS)
    args: argparse.Namespace = parser.parse_args(request.argv)
    data_dir: Path = Path(str(args.data_dir)).expanduser().resolve()
    host: str = str(args.host).strip() or DEFAULT_HOST
    port: int = int(args.port)
    debug: bool = bool(args.debug)
    max_items: int = max(1, int(args.max_items))
    max_notebooks: int = max(0, int(args.max_notebooks))
    return DashboardConfig(data_dir=data_dir, host=host, port=port, debug=debug, max_items=max_items, max_notebooks=max_notebooks)


def resolve_sources(request: ResolveSourcesRequest) -> list[SourceDefinition]:
    kaggle_dir: Path = request.data_dir / "kaggle_notebooks"
    github_dir: Path = request.data_dir / "github_notebooks"
    huggingface_dir: Path = request.data_dir / "huggingface_notebooks"
    return [
        SourceDefinition(key="kaggle", label="Kaggle", base_dir=kaggle_dir, default_selection=DEFAULT_KAGGLE_SELECTION, selection_label="Selection", is_aggregate=False),
        SourceDefinition(key="github", label="GitHub", base_dir=github_dir, default_selection=DEFAULT_GITHUB_SELECTION, selection_label="Selection", is_aggregate=False),
        SourceDefinition(key="huggingface", label="Hugging Face", base_dir=huggingface_dir, default_selection=DEFAULT_HF_SELECTION, selection_label="Sort category", is_aggregate=False),
        SourceDefinition(key="all", label="All sources", base_dir=request.data_dir, default_selection=DEFAULT_ALL_SELECTION, selection_label="Selection", is_aggregate=True),
    ]


def list_selection_dirs(request: ListSelectionDirsRequest) -> list[str]:
    if not request.base_dir.exists():
        return []
    selection_dirs: list[str] = [path.name for path in request.base_dir.iterdir() if path.is_dir()]
    selection_dirs.sort()
    return selection_dirs


def resolve_selection_options(request: ResolveSelectionOptionsRequest) -> SelectionOptions:
    if request.is_aggregate:
        return SelectionOptions(options=[request.default_selection], default_value=request.default_selection)
    options: list[str] = list_selection_dirs(ListSelectionDirsRequest(base_dir=request.base_dir))
    if request.fallback_options is not None:
        options = list(request.fallback_options)
    if request.preferred_order is not None:
        order_lookup: set[str] = set(options)
        ordered: list[str] = [value for value in request.preferred_order if value in order_lookup]
        remaining: list[str] = [value for value in options if value not in set(ordered)]
        options = ordered + sorted(remaining)
    if request.include_all and len(options) > 1:
        if DEFAULT_ALL_OPTION not in options:
            options = [DEFAULT_ALL_OPTION] + options
    if len(options) == 0:
        return SelectionOptions(options=[request.default_selection], default_value=request.default_selection)
    default_value: str = request.default_selection if request.default_selection in options else options[0]
    return SelectionOptions(options=options, default_value=default_value)


def format_selection_label(selection_label: str) -> str:
    normalized_label: str = selection_label.strip().lower()
    override: str | None = SELECTION_LABEL_OVERRIDES.get(normalized_label)
    if override is not None:
        return override
    return selection_label.replace("-", " ").replace("_", " ").title()


def build_component_ids(request: ComponentIdsRequest) -> SourceComponentIds:
    source_key: str = request.source_key
    return SourceComponentIds(
        selection_dropdown=f"{source_key}-selection",
        summary_container=f"{source_key}-summary",
        library_chart=f"{source_key}-library-chart",
        family_chart=f"{source_key}-family-chart",
        library_table=f"{source_key}-library-table",
        extension_table=f"{source_key}-extension-table",
        status_message=f"{source_key}-status",
    )


def collect_notebook_paths(request: NotebookPathsRequest) -> NotebookPathsResult:
    if not request.selection_dir.exists():
        return NotebookPathsResult(notebook_paths=tuple())
    all_paths: list[Path] = sorted(request.selection_dir.rglob("*.ipynb"))
    if request.max_notebooks > 0:
        all_paths = all_paths[:request.max_notebooks]
    return NotebookPathsResult(notebook_paths=tuple(all_paths))


def collect_notebook_usages(request: NotebookUsagesRequest) -> NotebookUsagesResult:
    normalizer_config = default_normalizer_config()
    library_usages: list[NotebookFeatureUsage] = []
    extension_usages: list[NotebookFeatureUsage] = []
    skipped_count: int = 0
    for notebook_path in request.notebook_paths:
        try:
            parse_result = parse_notebook(NotebookParseRequest(notebook_path=notebook_path))
        except RuntimeError:
            skipped_count += 1
            continue
        normalize_result = normalize_imports(NormalizeImportsRequest(raw_imports=parse_result.imports, config=normalizer_config))
        library_usages.append(NotebookFeatureUsage(notebook_path=notebook_path, features=normalize_result.normalized_imports))
        extension_usages.append(NotebookFeatureUsage(notebook_path=notebook_path, features=parse_result.extensions))
    return NotebookUsagesResult(library_usages=library_usages, extension_usages=extension_usages, skipped_count=skipped_count)


@lru_cache(maxsize=32)
def build_selection_report(request: SelectionReportRequest) -> SelectionReport:
    notebook_paths_result: NotebookPathsResult = collect_notebook_paths(NotebookPathsRequest(selection_dir=request.selection_dir, max_notebooks=request.max_notebooks))
    usages_result: NotebookUsagesResult = collect_notebook_usages(NotebookUsagesRequest(notebook_paths=notebook_paths_result.notebook_paths))
    library_report: UsageReport = compute_usage_metrics(UsageMetricsRequest(notebook_usages=usages_result.library_usages))
    extension_report: UsageReport = compute_usage_metrics(UsageMetricsRequest(notebook_usages=usages_result.extension_usages))
    analyzed_notebooks: int = len(usages_result.library_usages)
    return SelectionReport(library_report=library_report, extension_report=extension_report, analyzed_notebooks=analyzed_notebooks, skipped_notebooks=usages_result.skipped_count)


def build_summary_metrics(request: SummaryMetricsRequest) -> list[KpiMetric]:
    total_notebooks: int = request.library_report.total_notebooks
    unique_libraries: int = len(request.library_report.usage)
    unique_extensions: int = len(request.extension_report.usage)
    top_library: str = request.library_report.usage[0].name if unique_libraries > 0 else "None"
    top_extension: str = request.extension_report.usage[0].name if unique_extensions > 0 else "None"
    skipped_value: str = str(request.skipped_notebooks)
    return [
        KpiMetric(label="Analyzed notebooks", value=str(total_notebooks)),
        KpiMetric(label="Skipped notebooks", value=skipped_value),
        KpiMetric(label="Unique libraries", value=str(unique_libraries)),
        KpiMetric(label="Unique extensions", value=str(unique_extensions)),
        KpiMetric(label="Top library", value=top_library),
        KpiMetric(label="Top extension", value=top_extension),
    ]


def build_metric_cards(request: MetricCardsRequest) -> list[Component]:
    cards: list[Component] = []
    for metric in request.metrics:
        cards.append(request.modules.html.Div([request.modules.html.Div(metric.label, style={"fontSize": "12px", "color": "#6b7280"}), request.modules.html.Div(metric.value, style={"fontSize": "18px", "fontWeight": "600"})], style=CARD_STYLE))
    return cards


def build_usage_table_rows(request: UsageTableRequest) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    usages: list[FeatureUsage] = list(request.report.usage)
    if request.include_family:
        usages.sort(key=lambda usage: (resolve_family_order(resolve_family_name(usage.name)), -usage.notebook_count, usage.name))
    for usage in usages:
        row: dict[str, int | float | str] = {"name": usage.name, "count": usage.notebook_count, "percent": usage.usage_percent}
        if request.include_family:
            row["family"] = resolve_family_name(usage.name)
        rows.append(row)
    return rows


def build_empty_figure(request: EmptyFigureRequest) -> Figure:
    import plotly.graph_objects as go
    fig: Figure = go.Figure()
    fig.add_annotation(text=request.message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin={"l": 40, "r": 20, "t": 40, "b": 40})
    return fig


def format_hex_color(red: int, green: int, blue: int) -> str:
    return f"#{red:02x}{green:02x}{blue:02x}"


def interpolate_color_channel(start_value: int, end_value: int, ratio: float) -> int:
    return int(round(start_value + ((end_value - start_value) * ratio)))


def build_gradient_colors(counts: Sequence[int], start_color: tuple[int, int, int], end_color: tuple[int, int, int]) -> list[str]:
    if len(counts) == 0:
        return []
    max_count: int = max(counts)
    min_count: int = min(counts)
    if max_count == min_count:
        return [format_hex_color(*start_color) for _ in counts]
    range_count: int = max_count - min_count
    colors: list[str] = []
    for count in counts:
        fade_ratio: float = (max_count - count) / range_count
        red: int = interpolate_color_channel(start_color[0], end_color[0], fade_ratio)
        green: int = interpolate_color_channel(start_color[1], end_color[1], fade_ratio)
        blue: int = interpolate_color_channel(start_color[2], end_color[2], fade_ratio)
        colors.append(format_hex_color(red, green, blue))
    return colors


def parse_hex_color(color: str) -> tuple[int, int, int]:
    normalized: str = color.strip().removeprefix("#")
    if len(normalized) == 3:
        normalized = f"{normalized[0]}{normalized[0]}{normalized[1]}{normalized[1]}{normalized[2]}{normalized[2]}"
    if len(normalized) != 6:
        return (107, 114, 128)
    try:
        red: int = int(normalized[0:2], 16)
        green: int = int(normalized[2:4], 16)
        blue: int = int(normalized[4:6], 16)
    except ValueError:
        return (107, 114, 128)
    return red, green, blue


def mix_color(base_color: tuple[int, int, int], target_color: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    red: int = interpolate_color_channel(base_color[0], target_color[0], ratio)
    green: int = interpolate_color_channel(base_color[1], target_color[1], ratio)
    blue: int = interpolate_color_channel(base_color[2], target_color[2], ratio)
    return red, green, blue


def build_blue_gradient_colors(counts: Sequence[int]) -> list[str]:
    dark_blue: tuple[int, int, int] = (30, 58, 138)
    light_blue: tuple[int, int, int] = (191, 219, 254)
    return build_gradient_colors(counts=counts, start_color=dark_blue, end_color=light_blue)


def build_family_gradient_colors(counts: Sequence[int], family_color: str) -> list[str]:
    base_color: tuple[int, int, int] = parse_hex_color(family_color)
    dark_color: tuple[int, int, int] = mix_color(base_color=base_color, target_color=(0, 0, 0), ratio=0.2)
    light_color: tuple[int, int, int] = mix_color(base_color=base_color, target_color=(255, 255, 255), ratio=0.65)
    return build_gradient_colors(counts=counts, start_color=dark_color, end_color=light_color)


def build_bar_chart(request: ChartBuildRequest) -> Figure:
    labels, counts, percents = build_chart_data(request.report.usage, request.max_items)
    if len(labels) == 0:
        return build_empty_figure(EmptyFigureRequest(message=f"No {request.x_label.lower()} data available."))
    import plotly.graph_objects as go
    percent_labels: list[str] = [f"{value:.2f}%" for value in percents]
    gradient_colors: list[str] = build_blue_gradient_colors(counts)
    fig: Figure = go.Figure(go.Bar(x=counts, y=labels, orientation="h", text=percent_labels, textposition="outside", marker_color=gradient_colors, cliponaxis=False))
    fig.update_layout(title=request.title, xaxis_title="Notebook count", yaxis_title=request.x_label, yaxis={"autorange": "reversed"}, margin={"l": 80, "r": 20, "t": 60, "b": 40})
    return fig


def resolve_family_name(library_name: str) -> str:
    normalized_name: str = library_name.strip().lower()
    mapped_family: str | None = PACKAGE_FAMILY_MAPPING.get(normalized_name)
    if mapped_family is not None:
        return mapped_family
    for prefix, family_name in FAMILY_PREFIX_MAPPING:
        if normalized_name == prefix:
            return family_name
        if normalized_name.startswith(f"{prefix}.") or normalized_name.startswith(f"{prefix}_") or normalized_name.startswith(f"{prefix}-"):
            return family_name
    return DEFAULT_FAMILY


def resolve_family_order(family_name: str) -> int:
    return FAMILY_ORDER_LOOKUP.get(family_name, len(FAMILY_DISPLAY_ORDER))


def resolve_family_group_order(grouped_by_family: dict[str, list[FamilyUsage]]) -> list[str]:
    return sorted(grouped_by_family.keys(), key=lambda family_name: (-sum(item.usage.notebook_count for item in grouped_by_family[family_name]), resolve_family_order(family_name), family_name.lower()))


def build_family_grouped_bar_chart(request: FamilyChartBuildRequest) -> Figure:
    selected_usages: list[FeatureUsage] = list(request.report.usage)[:request.max_items]
    if len(selected_usages) == 0:
        return build_empty_figure(EmptyFigureRequest(message="No grouped family data available."))
    grouped_by_family: dict[str, list[FamilyUsage]] = {}
    for usage in selected_usages:
        family_name: str = resolve_family_name(usage.name)
        grouped_by_family.setdefault(family_name, []).append(FamilyUsage(usage=usage, family=family_name))
    for family_items in grouped_by_family.values():
        family_items.sort(key=lambda item: (-item.usage.notebook_count, item.usage.name))
    ordered_families: list[str] = resolve_family_group_order(grouped_by_family)
    ordered_labels: list[str] = []
    for family_name in ordered_families:
        ordered_labels.extend(item.usage.name for item in grouped_by_family[family_name])
    import plotly.graph_objects as go
    fig: Figure = go.Figure()
    for family_name in ordered_families:
        family_items: list[FamilyUsage] = grouped_by_family[family_name]
        labels: list[str] = [item.usage.name for item in family_items]
        counts: list[int] = [item.usage.notebook_count for item in family_items]
        percent_labels: list[str] = [f"{item.usage.usage_percent:.2f}%" for item in family_items]
        family_color: str = FAMILY_COLOR_MAPPING.get(family_name, DEFAULT_FAMILY_COLOR)
        family_gradient_colors: list[str] = build_family_gradient_colors(counts=counts, family_color=family_color)
        fig.add_trace(go.Bar(name=family_name, x=counts, y=labels, orientation="h", text=percent_labels, textposition="outside", marker_color=family_gradient_colors, cliponaxis=False, hovertemplate="<b>%{y}</b><br>Notebook count: %{x}<br>Usage: %{text}<extra>%{fullData.name}</extra>"))
    fig.update_layout(
        title=request.title,
        xaxis_title="Notebook count",
        yaxis_title="Library",
        yaxis={"autorange": "reversed", "categoryorder": "array", "categoryarray": ordered_labels},
        margin={"l": 120, "r": 20, "t": 60, "b": 40},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0, "traceorder": "normal"},
    )
    return fig


def build_tab_layout(request: BuildTabLayoutRequest) -> Component:
    selection_options: list[dict[str, str]] = [{"label": format_selection_label(option), "value": option} for option in request.selection_options.options]
    return request.modules.html.Div(
        [
            request.modules.html.H2(f"{request.source.label} notebooks"),
            request.modules.html.Div(
                [
                    request.modules.html.Div(request.source.selection_label, style={"fontSize": "12px", "color": "#6b7280"}),
                    request.modules.dcc.Dropdown(id=request.ids.selection_dropdown, options=selection_options, value=request.selection_options.default_value, clearable=False),
                ],
                style={"maxWidth": "320px"},
            ),
            request.modules.html.Div(id=request.ids.status_message, style={"marginTop": "8px", "color": "#6b7280"}),
            request.modules.html.Div(id=request.ids.summary_container, style=CARD_CONTAINER_STYLE),
            request.modules.html.Div(
                [
                    request.modules.html.H3("Top 20 most used libraries (unique per notebook)"),
                    request.modules.dcc.Loading(request.modules.dcc.Graph(id=request.ids.library_chart), type="default"),
                ],
                style=SECTION_STYLE,
            ),
            request.modules.html.Div(
                [
                    request.modules.html.H3("Top libraries grouped by family (All data)"),
                    request.modules.dcc.Loading(request.modules.dcc.Graph(id=request.ids.family_chart), type="default"),
                ],
                style=SECTION_STYLE,
            ),
            request.modules.html.Div(
                [
                    request.modules.html.H3("Complete list of libraries"),
                    request.modules.dash_table.DataTable(id=request.ids.library_table, columns=LIBRARY_TABLE_COLUMNS, data=[], page_size=15, sort_action="native", filter_action="native", style_table=TABLE_STYLE),
                ],
                style=SECTION_STYLE,
            ),
            request.modules.html.Div(
                [
                    request.modules.html.H3("Complete list of extensions"),
                    request.modules.dash_table.DataTable(id=request.ids.extension_table, columns=EXTENSION_TABLE_COLUMNS, data=[], page_size=15, sort_action="native", filter_action="native", style_table=TABLE_STYLE),
                ],
                style=SECTION_STYLE,
            ),
        ],
        style=APP_STYLE,
    )


def build_tabs_layout(request: BuildTabsLayoutRequest) -> Component:
    tabs: list[Component] = []
    for source in request.sources:
        options: SelectionOptions = request.selection_options[source.key]
        ids: SourceComponentIds = request.component_ids[source.key]
        tabs.append(request.modules.dcc.Tab(label=source.label, value=source.key, children=build_tab_layout(BuildTabLayoutRequest(modules=request.modules, source=source, selection_options=options, ids=ids))))
    if request.comparison_tab is not None:
        tabs.append(request.comparison_tab)
    default_tab_value: str = request.sources[0].key if len(request.sources) > 0 else PACKAGE_COMPARISON_TAB_KEY
    return request.modules.html.Div([request.modules.dcc.Tabs(id="dashboard-tabs", value=default_tab_value, children=tabs)])


def build_status_message(selection_dir: Path, analyzed: int, skipped: int) -> str:
    if not selection_dir.exists():
        return f"No notebooks found at {selection_dir}."
    return f"Analyzed {analyzed} notebooks. Skipped {skipped} invalid notebooks."


def resolve_library_table_field_names() -> list[str]:
    table_field_names: list[str] = [column["id"] for column in LIBRARY_TABLE_COLUMNS]
    return ["row_number", *table_field_names]


def build_library_table_csv_path(data_dir: Path) -> Path:
    return data_dir / DEFAULT_LIBRARY_TABLE_CSV_FILENAME


def save_library_table_csv(data_dir: Path, rows: Sequence[dict[str, int | float | str]]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = build_library_table_csv_path(data_dir=data_dir)
    field_names: list[str] = resolve_library_table_field_names()
    ordered_rows: list[dict[str, int | float | str]] = []
    for row_number, row in enumerate(rows, start=1):
        ordered_row: dict[str, int | float | str] = {"row_number": row_number}
        for field_name in field_names[1:]:
            ordered_row[field_name] = row.get(field_name, "")
        ordered_rows.append(ordered_row)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer: csv.DictWriter[str] = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(ordered_rows)
    return output_path


def build_current_packages_csv_path(data_dir: Path) -> Path:
    return data_dir / DEFAULT_CURRENT_PACKAGES_CSV_FILENAME


def normalize_package_identifier(package_name: str) -> str:
    lowered_name: str = package_name.strip().lower()
    return re.sub(r"[-_.]+", "-", lowered_name)


def parse_csv_int(raw_value: str) -> int:
    normalized_value: str = raw_value.strip()
    if normalized_value == "":
        return 0
    try:
        return int(float(normalized_value))
    except ValueError:
        return 0


def parse_csv_float(raw_value: str) -> float:
    normalized_value: str = raw_value.strip()
    if normalized_value == "":
        return 0.0
    try:
        return float(normalized_value)
    except ValueError:
        return 0.0


def is_platform_runtime_package(package_name: str) -> bool:
    normalized_name: str = normalize_package_identifier(package_name)
    if normalized_name in PLATFORM_RUNTIME_PACKAGES:
        return True
    for prefix in PLATFORM_RUNTIME_PREFIXES:
        if normalized_name.startswith(prefix):
            return True
    return False


def resolve_package_comparison_status(package_name: str, in_current: bool, in_external: bool, flavor_count: int, external_percent: float) -> tuple[str, str]:
    if in_current and in_external:
        if external_percent >= PACKAGE_COMPARISON_ADD_NOW_THRESHOLD:
            return "Hot (covered)", "Keep package and monitor version freshness."
        return "Covered (long tail)", "Keep package if maintenance cost stays low."
    if not in_current and in_external:
        if external_percent >= PACKAGE_COMPARISON_ADD_NOW_THRESHOLD:
            return "Add now", "Prioritize this package for the next image update."
        if external_percent >= PACKAGE_COMPARISON_WATCHLIST_THRESHOLD:
            return "Watchlist", "Evaluate security/size impact before inclusion."
        return "Long-tail gap", "Add only on demand."
    if is_platform_runtime_package(package_name):
        return "Core/Internal", "Keep as platform/runtime dependency."
    if flavor_count <= 2:
        return "Review candidate", "Validate telemetry usage and consider removal."
    return "Specialized covered", "Keep for targeted workloads."


def resolve_comparison_marker_size(external_count: int, flavor_count: int) -> float:
    if external_count <= 0:
        return float(10 + min(flavor_count, 12))
    scaled_size: float = (float(external_count) ** 0.5) * 2.0 + 8.0
    return max(10.0, min(46.0, scaled_size))


def format_family_hover_package_list(packages: Sequence[tuple[float, str]], max_items: int = PACKAGE_COMPARISON_HOVER_PACKAGE_LIMIT) -> str:
    if len(packages) == 0:
        return "None"
    ordered_packages: list[tuple[float, str]] = sorted(packages, key=lambda item: (-item[0], item[1].lower()))
    visible_packages: list[tuple[float, str]] = ordered_packages[:max_items]
    lines: list[str] = [f"{name} ({percent:.2f}%)" for percent, name in visible_packages]
    hidden_count: int = len(ordered_packages) - len(visible_packages)
    if hidden_count > 0:
        lines.append(f"... and {hidden_count} more")
    return "<br>".join(lines)


def load_current_package_records(data_dir: Path) -> tuple[dict[str, CurrentPackageRecord], list[dict[str, int | float | str]], str | None]:
    csv_path: Path = build_current_packages_csv_path(data_dir=data_dir)
    if not csv_path.exists():
        return {}, [], f"Current package CSV not found: {csv_path}"
    records: dict[str, CurrentPackageRecord] = {}
    table_rows: list[dict[str, int | float | str]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as input_file:
            reader: csv.DictReader[str] = csv.DictReader(input_file)
            for row_number, row in enumerate(reader, start=1):
                package_name: str = str(row.get("package_name", "")).strip()
                if package_name == "":
                    continue
                flavors: str = str(row.get("flavors", "")).strip()
                flavor_items: list[str] = [value.strip() for value in flavors.split("|") if value.strip() != ""]
                flavor_count: int = len(flavor_items)
                canonical_name: str = normalize_package_identifier(package_name)
                if canonical_name == "":
                    continue
                existing_record: CurrentPackageRecord | None = records.get(canonical_name)
                candidate_record: CurrentPackageRecord = CurrentPackageRecord(package_name=package_name, canonical_name=canonical_name, flavor_count=flavor_count, flavors=flavors)
                if existing_record is None or candidate_record.flavor_count > existing_record.flavor_count:
                    records[canonical_name] = candidate_record
                table_rows.append({"row_number": row_number, "package_name": package_name, "flavor_count": flavor_count, "flavors": flavors})
    except OSError as exc:
        return {}, [], f"Failed to read current package CSV at {csv_path}: {exc}"
    return records, table_rows, None


def load_external_package_records(data_dir: Path) -> tuple[dict[str, ExternalPackageRecord], list[dict[str, int | float | str]], str | None]:
    csv_path: Path = build_library_table_csv_path(data_dir=data_dir)
    if not csv_path.exists():
        return {}, [], f"External package CSV not found: {csv_path}"
    records: dict[str, ExternalPackageRecord] = {}
    table_rows: list[dict[str, int | float | str]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as input_file:
            reader: csv.DictReader[str] = csv.DictReader(input_file)
            for fallback_row_number, row in enumerate(reader, start=1):
                package_name: str = str(row.get("name", "")).strip()
                if package_name == "":
                    continue
                canonical_name: str = normalize_package_identifier(package_name)
                if canonical_name == "":
                    continue
                row_number: int = parse_csv_int(str(row.get("row_number", "")))
                if row_number <= 0:
                    row_number = fallback_row_number
                family_name: str = str(row.get("family", "")).strip()
                if family_name == "":
                    family_name = resolve_family_name(package_name)
                count_value: int = parse_csv_int(str(row.get("count", "")))
                percent_value: float = parse_csv_float(str(row.get("percent", "")))
                candidate_record: ExternalPackageRecord = ExternalPackageRecord(package_name=package_name, canonical_name=canonical_name, family=family_name, count=count_value, percent=percent_value, row_number=row_number)
                existing_record: ExternalPackageRecord | None = records.get(canonical_name)
                if existing_record is None or candidate_record.count > existing_record.count:
                    records[canonical_name] = candidate_record
                table_rows.append({"row_number": row_number, "name": package_name, "family": family_name, "count": count_value, "percent": round(percent_value, 2)})
    except OSError as exc:
        return {}, [], f"Failed to read external package CSV at {csv_path}: {exc}"
    return records, table_rows, None


def build_package_comparison_rows(current_records: dict[str, CurrentPackageRecord], external_records: dict[str, ExternalPackageRecord]) -> list[PackageComparisonRow]:
    all_package_keys: list[str] = sorted(set(current_records.keys()) | set(external_records.keys()))
    comparison_rows: list[PackageComparisonRow] = []
    for package_key in all_package_keys:
        current_record: CurrentPackageRecord | None = current_records.get(package_key)
        external_record: ExternalPackageRecord | None = external_records.get(package_key)
        in_current: bool = current_record is not None
        in_external: bool = external_record is not None
        if external_record is not None:
            package_name: str = external_record.package_name
        else:
            if current_record is None:
                continue
            package_name = current_record.package_name
        family_name: str = external_record.family if external_record is not None else resolve_family_name(package_name)
        external_count: int = external_record.count if external_record is not None else 0
        external_percent: float = external_record.percent if external_record is not None else 0.0
        flavor_count: int = current_record.flavor_count if current_record is not None else 0
        status, recommendation = resolve_package_comparison_status(package_name=package_name, in_current=in_current, in_external=in_external, flavor_count=flavor_count, external_percent=external_percent)
        marker_size: float = resolve_comparison_marker_size(external_count=external_count, flavor_count=flavor_count)
        comparison_rows.append(PackageComparisonRow(package_name=package_name, canonical_name=package_key, family=family_name, status=status, recommendation=recommendation, in_current=in_current, in_external=in_external, flavor_count=flavor_count, external_count=external_count, external_percent=external_percent, marker_size=marker_size))
    comparison_rows.sort(key=lambda row: (PACKAGE_COMPARISON_STATUS_PRIORITY.get(row.status, 99), -row.external_percent, -row.external_count, row.flavor_count, row.package_name.lower()))
    return comparison_rows


def build_package_comparison_action_rows(comparison_rows: Sequence[PackageComparisonRow]) -> list[dict[str, int | float | str]]:
    selected_rows: list[PackageComparisonRow] = [row for row in comparison_rows if row.status in PACKAGE_COMPARISON_ACTION_STATUSES]
    selected_rows.sort(key=lambda row: (PACKAGE_COMPARISON_STATUS_PRIORITY.get(row.status, 99), -row.external_percent, -row.external_count, row.flavor_count, row.package_name.lower()))
    action_rows: list[dict[str, int | float | str]] = []
    for row in selected_rows:
        action_rows.append({"package_name": row.package_name, "status": row.status, "recommendation": row.recommendation, "family": row.family, "external_percent": round(row.external_percent, 2), "external_count": row.external_count, "flavor_count": row.flavor_count})
    return action_rows


def build_package_comparison_metrics(current_records: dict[str, CurrentPackageRecord], external_records: dict[str, ExternalPackageRecord], comparison_rows: Sequence[PackageComparisonRow]) -> list[KpiMetric]:
    current_count: int = len(current_records)
    external_count: int = len(external_records)
    overlap_count: int = len(set(current_records.keys()) & set(external_records.keys()))
    total_external_usage: int = sum(record.count for record in external_records.values())
    covered_external_usage: int = sum(record.count for package_key, record in external_records.items() if package_key in current_records)
    weighted_coverage: float = (100.0 * covered_external_usage / total_external_usage) if total_external_usage > 0 else 0.0
    add_now_count: int = sum(1 for row in comparison_rows if row.status == "Add now")
    review_count: int = sum(1 for row in comparison_rows if row.status == "Review candidate")
    return [
        KpiMetric(label="Current packages", value=str(current_count)),
        KpiMetric(label="External packages", value=str(external_count)),
        KpiMetric(label="Overlap", value=f"{overlap_count}"),
        KpiMetric(label="Weighted external coverage", value=f"{weighted_coverage:.2f}%"),
        KpiMetric(label="Add now candidates", value=str(add_now_count)),
        KpiMetric(label="Review candidates", value=str(review_count)),
    ]


def build_package_priority_matrix_figure(comparison_rows: Sequence[PackageComparisonRow]) -> Figure:
    if len(comparison_rows) == 0:
        return build_empty_figure(EmptyFigureRequest(message="No package comparison data available."))
    import plotly.graph_objects as go
    ordered_statuses: list[str] = sorted({row.status for row in comparison_rows}, key=lambda status: (PACKAGE_COMPARISON_STATUS_PRIORITY.get(status, 99), status))
    fig: Figure = go.Figure()
    for status_name in ordered_statuses:
        status_rows: list[PackageComparisonRow] = [row for row in comparison_rows if row.status == status_name]
        custom_data: list[list[str | int]] = [[row.family, row.recommendation, row.external_count, "yes" if row.in_current else "no", "yes" if row.in_external else "no"] for row in status_rows]
        fig.add_trace(
            go.Scatter(
                x=[row.external_percent for row in status_rows],
                y=[row.flavor_count for row in status_rows],
                mode="markers",
                name=status_name,
                text=[row.package_name for row in status_rows],
                customdata=custom_data,
                marker={
                    "size": [row.marker_size for row in status_rows],
                    "opacity": 0.78,
                    "color": PACKAGE_COMPARISON_STATUS_COLORS.get(status_name, DEFAULT_FAMILY_COLOR),
                    "line": {"width": 1, "color": "#ffffff"},
                },
                hovertemplate="<b>%{text}</b><br>Status: "
                + status_name
                + "<br>External usage: %{x:.2f}%<br>Flavor coverage: %{y}<br>External count: %{customdata[2]}<br>Family: %{customdata[0]}<br>In IDE: %{customdata[3]}<br>In external set: %{customdata[4]}<br>Recommendation: %{customdata[1]}<extra></extra>",
            ),
        )
    fig.add_vline(x=PACKAGE_COMPARISON_WATCHLIST_THRESHOLD, line={"color": "#94a3b8", "dash": "dot", "width": 1})
    fig.add_vline(x=PACKAGE_COMPARISON_ADD_NOW_THRESHOLD, line={"color": "#64748b", "dash": "dash", "width": 1})
    fig.update_layout(
        title="Package priority matrix: external demand vs IDE coverage",
        xaxis_title="External usage share (%)",
        yaxis_title="Flavor coverage in current IDE",
        margin={"l": 80, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    fig.update_xaxes(range=[0, max(2.0, max(row.external_percent for row in comparison_rows) * 1.05)])
    fig.update_yaxes(dtick=1)
    return fig


def build_package_family_gap_figure(current_records: dict[str, CurrentPackageRecord], external_records: dict[str, ExternalPackageRecord]) -> Figure:
    if len(external_records) == 0:
        return build_empty_figure(EmptyFigureRequest(message="No external package data available."))
    import plotly.graph_objects as go
    family_totals: dict[str, dict[str, float]] = {}
    family_packages: dict[str, dict[str, list[tuple[float, str]]]] = {}
    for package_key, external_record in external_records.items():
        family_name: str = external_record.family if external_record.family != "" else resolve_family_name(external_record.package_name)
        if family_name not in family_totals:
            family_totals[family_name] = {"covered": 0.0, "missing": 0.0}
            family_packages[family_name] = {"covered": [], "missing": []}
        if package_key in current_records:
            family_totals[family_name]["covered"] += external_record.percent
            family_packages[family_name]["covered"].append((external_record.percent, external_record.package_name))
        else:
            family_totals[family_name]["missing"] += external_record.percent
            family_packages[family_name]["missing"].append((external_record.percent, external_record.package_name))
    ordered_families: list[str] = sorted(family_totals.keys(), key=lambda family_name: (-(family_totals[family_name]["covered"] + family_totals[family_name]["missing"]), resolve_family_order(family_name), family_name.lower()))
    covered_values: list[float] = [round(family_totals[family_name]["covered"], 2) for family_name in ordered_families]
    missing_values: list[float] = [round(family_totals[family_name]["missing"], 2) for family_name in ordered_families]
    covered_package_lists: list[str] = [format_family_hover_package_list(packages=family_packages[family_name]["covered"]) for family_name in ordered_families]
    missing_package_lists: list[str] = [format_family_hover_package_list(packages=family_packages[family_name]["missing"]) for family_name in ordered_families]
    fig: Figure = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Covered in IDE",
            x=covered_values,
            y=ordered_families,
            orientation="h",
            marker_color="#22c55e",
            customdata=covered_package_lists,
            hovertemplate="<b>%{y}</b><br>Covered usage: %{x:.2f}%<br><br><b>Packages</b><br>%{customdata}<extra></extra>",
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Missing in IDE",
            x=missing_values,
            y=ordered_families,
            orientation="h",
            marker_color="#ef4444",
            customdata=missing_package_lists,
            hovertemplate="<b>%{y}</b><br>Missing usage: %{x:.2f}%<br><br><b>Packages</b><br>%{customdata}<extra></extra>",
        ),
    )
    fig.update_layout(
        title="External usage share by family: covered vs missing",
        barmode="stack",
        xaxis_title="External usage share (%)",
        yaxis_title="Family",
        yaxis={"autorange": "reversed"},
        margin={"l": 140, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return fig


@lru_cache(maxsize=8)
def build_package_comparison_dataset(data_dir: Path) -> PackageComparisonDataset:
    resolved_data_dir: Path = data_dir.resolve()
    current_records, current_table_rows, current_error = load_current_package_records(data_dir=resolved_data_dir)
    external_records, external_table_rows, external_error = load_external_package_records(data_dir=resolved_data_dir)
    comparison_rows: list[PackageComparisonRow] = build_package_comparison_rows(current_records=current_records, external_records=external_records)
    action_rows: list[dict[str, int | float | str]] = build_package_comparison_action_rows(comparison_rows=comparison_rows)
    metrics: list[KpiMetric] = build_package_comparison_metrics(current_records=current_records, external_records=external_records, comparison_rows=comparison_rows)
    error_messages: list[str] = [message for message in [current_error, external_error] if message is not None]
    if len(error_messages) > 0:
        status_message: str = " ".join(error_messages)
    else:
        overlap_count: int = len(set(current_records.keys()) & set(external_records.keys()))
        status_message = f"Compared {len(current_records)} current IDE packages against {len(external_records)} external packages. Overlap: {overlap_count}."
    return PackageComparisonDataset(current_records=current_records, external_records=external_records, comparison_rows=comparison_rows, action_rows=action_rows, current_table_rows=current_table_rows, external_table_rows=external_table_rows, metrics=metrics, status_message=status_message)


def build_package_comparison_tab(request: BuildPackageComparisonTabRequest) -> Component:
    dataset: PackageComparisonDataset = build_package_comparison_dataset(data_dir=request.config.data_dir)
    priority_figure: Figure = build_package_priority_matrix_figure(comparison_rows=dataset.comparison_rows)
    family_gap_figure: Figure = build_package_family_gap_figure(current_records=dataset.current_records, external_records=dataset.external_records)
    summary_cards: list[Component] = build_metric_cards(MetricCardsRequest(modules=request.modules, metrics=dataset.metrics))
    return request.modules.dcc.Tab(
        label="Package comparison",
        value=PACKAGE_COMPARISON_TAB_KEY,
        children=request.modules.html.Div(
            [
                request.modules.html.H2("Package comparison"),
                request.modules.html.Div(
                    "Comparison of current IDE packages vs broad external package usage to highlight hot packages, obsolete candidates, and potential additions.",
                    style={"fontSize": "14px", "color": "#374151"},
                ),
                request.modules.html.Div(dataset.status_message, style={"marginTop": "8px", "color": "#6b7280"}),
                request.modules.html.Div(id="package-comparison-summary", children=summary_cards, style=CARD_CONTAINER_STYLE),
                request.modules.html.Div(
                    [
                        request.modules.html.H3("Priority matrix"),
                        request.modules.dcc.Graph(id="package-comparison-priority-matrix", figure=priority_figure),
                    ],
                    style=SECTION_STYLE,
                ),
                request.modules.html.Div(
                    [
                        request.modules.html.H3("Family coverage gap"),
                        request.modules.dcc.Graph(id="package-comparison-family-gap", figure=family_gap_figure),
                    ],
                    style=SECTION_STYLE,
                ),
                request.modules.html.Div(
                    [
                        request.modules.html.H3("Action table (Hot, Add now, Watchlist, Review)"),
                        request.modules.dash_table.DataTable(
                            id="package-comparison-action-table",
                            columns=PACKAGE_COMPARISON_ACTION_COLUMNS,
                            data=dataset.action_rows,
                            page_size=15,
                            sort_action="native",
                            filter_action="native",
                            style_table=TABLE_STYLE,
                            style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto"},
                        ),
                    ],
                    style=SECTION_STYLE,
                ),
                request.modules.html.Div(
                    [
                        request.modules.html.H3("Current packages CSV (raw dataframe)"),
                        request.modules.dash_table.DataTable(
                            id="package-comparison-current-csv-table",
                            columns=PACKAGE_COMPARISON_CURRENT_COLUMNS,
                            data=dataset.current_table_rows,
                            page_size=12,
                            sort_action="native",
                            filter_action="native",
                            style_table=TABLE_STYLE,
                            style_cell={"textAlign": "left"},
                        ),
                    ],
                    style=SECTION_STYLE,
                ),
                request.modules.html.Div(
                    [
                        request.modules.html.H3("External packages CSV (raw dataframe)"),
                        request.modules.dash_table.DataTable(
                            id="package-comparison-external-csv-table",
                            columns=PACKAGE_COMPARISON_EXTERNAL_COLUMNS,
                            data=dataset.external_table_rows,
                            page_size=12,
                            sort_action="native",
                            filter_action="native",
                            style_table=TABLE_STYLE,
                            style_cell={"textAlign": "left"},
                        ),
                    ],
                    style=SECTION_STYLE,
                ),
            ],
            style=APP_STYLE,
        ),
    )


def update_source_tab(selection_label: str, source: SourceDefinition, modules: DashModules, config: DashboardConfig) -> tuple[list[Component], Figure, Figure, list[dict[str, int | float | str]], list[dict[str, int | float | str]], str]:
    if selection_label is None or selection_label.strip() == "":
        empty_summary: list[Component] = build_metric_cards(MetricCardsRequest(modules=modules, metrics=[]))
        empty_figure: Figure = build_empty_figure(EmptyFigureRequest(message="No data available."))
        return empty_summary, empty_figure, empty_figure, [], [], "No selection provided."
    if source.is_aggregate or selection_label == DEFAULT_ALL_OPTION:
        selection_dir: Path = source.base_dir.resolve()
    else:
        selection_dir = (source.base_dir / selection_label).resolve()
    report: SelectionReport = build_selection_report(SelectionReportRequest(selection_dir=selection_dir, max_notebooks=config.max_notebooks))
    metrics: list[KpiMetric] = build_summary_metrics(SummaryMetricsRequest(library_report=report.library_report, extension_report=report.extension_report, skipped_notebooks=report.skipped_notebooks))
    summary_cards: list[Component] = build_metric_cards(MetricCardsRequest(modules=modules, metrics=metrics))
    selection_title: str = "All sources" if source.is_aggregate else format_selection_label(selection_label)
    if selection_label == DEFAULT_ALL_OPTION and not source.is_aggregate:
        selection_title = "All sort categories" if source.key == "huggingface" else "All selections"
    library_title: str = f"Top 20 most used libraries (unique per notebook) — {selection_title}"
    library_figure: Figure = build_bar_chart(ChartBuildRequest(report=report.library_report, title=library_title, x_label="Library", max_items=config.max_items))
    all_selection_dir: Path = source.base_dir.resolve()
    all_report: SelectionReport = report if all_selection_dir == selection_dir else build_selection_report(SelectionReportRequest(selection_dir=all_selection_dir, max_notebooks=config.max_notebooks))
    family_chart_title: str = f"Top {config.max_items} libraries grouped by family (All data) — {source.label}"
    family_figure: Figure = build_family_grouped_bar_chart(FamilyChartBuildRequest(report=all_report.library_report, title=family_chart_title, max_items=config.max_items))
    library_rows: list[dict[str, int | float | str]] = build_usage_table_rows(UsageTableRequest(report=report.library_report, include_family=True))
    extension_rows: list[dict[str, int | float | str]] = build_usage_table_rows(UsageTableRequest(report=report.extension_report))
    status_message: str = build_status_message(selection_dir=selection_dir, analyzed=report.analyzed_notebooks, skipped=report.skipped_notebooks)
    if source.is_aggregate:
        library_csv_path: Path = save_library_table_csv(data_dir=config.data_dir, rows=library_rows)
        status_message = f"{status_message} Library table CSV: {library_csv_path}"
    return summary_cards, library_figure, family_figure, library_rows, extension_rows, status_message


def execute_register_callbacks(request: RegisterCallbacksRequest) -> None:
    from dash import Input, Output
    for source in request.sources:
        ids: SourceComponentIds = request.component_ids[source.key]
        callback_func = functools.partial(update_source_tab, source=source, modules=request.modules, config=request.config)
        request.app.callback(
            Output(ids.summary_container, "children"),
            Output(ids.library_chart, "figure"),
            Output(ids.family_chart, "figure"),
            Output(ids.library_table, "data"),
            Output(ids.extension_table, "data"),
            Output(ids.status_message, "children"),
            Input(ids.selection_dropdown, "value"),
        )(callback_func)


def build_app(request: BuildAppRequest) -> Dash:
    ensure_dash()
    from dash import Dash, dcc, html, dash_table
    sources: list[SourceDefinition] = resolve_sources(ResolveSourcesRequest(data_dir=request.config.data_dir))
    selection_options: dict[str, SelectionOptions] = {}
    for source in sources:
        fallback_options: Sequence[str] | None = None
        include_all: bool = source.key in ("kaggle", "huggingface")
        preferred_order: Sequence[str] | None = None
        if source.key == "kaggle":
            preferred_order = KAGGLE_SELECTION_ORDER
        if source.key == "huggingface":
            preferred_order = DEFAULT_HF_SELECTION_ORDER
        selection_options[source.key] = resolve_selection_options(ResolveSelectionOptionsRequest(base_dir=source.base_dir, default_selection=source.default_selection, is_aggregate=source.is_aggregate, fallback_options=fallback_options, include_all=include_all, preferred_order=preferred_order))
    component_ids: dict[str, SourceComponentIds] = {source.key: build_component_ids(ComponentIdsRequest(source_key=source.key)) for source in sources}
    modules: DashModules = DashModules(dcc=dcc, html=html, dash_table=dash_table)
    comparison_tab: Component = build_package_comparison_tab(BuildPackageComparisonTabRequest(modules=modules, config=request.config))
    app: Dash = Dash(__name__)
    app.layout = build_tabs_layout(BuildTabsLayoutRequest(modules=modules, sources=sources, selection_options=selection_options, component_ids=component_ids, comparison_tab=comparison_tab))
    execute_register_callbacks(RegisterCallbacksRequest(app=app, modules=modules, sources=sources, component_ids=component_ids, config=request.config))
    return app


def main() -> None:
    config: DashboardConfig = parse_dashboard_args(ParseDashboardArgsRequest(argv=sys.argv[1:]))
    ensure_plotly()
    app: Dash = build_app(BuildAppRequest(config=config))
    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == "__main__":
    main()
