from __future__ import annotations

import argparse
import functools
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
DEFAULT_HF_REPO_TYPE: str = "model"
DEFAULT_HF_REPO_TYPES: list[str] = ["model", "dataset", "space"]
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
    huggingface_selection_dir: Path = resolve_huggingface_base_dir(huggingface_dir)
    return [
        SourceDefinition(key="kaggle", label="Kaggle", base_dir=kaggle_dir, default_selection=DEFAULT_KAGGLE_SELECTION, selection_label="Selection", is_aggregate=False),
        SourceDefinition(key="github", label="GitHub", base_dir=github_dir, default_selection=DEFAULT_GITHUB_SELECTION, selection_label="Selection", is_aggregate=False),
        SourceDefinition(key="huggingface", label="Hugging Face", base_dir=huggingface_selection_dir, default_selection=DEFAULT_HF_REPO_TYPE, selection_label="Repo type", is_aggregate=False),
        SourceDefinition(key="all", label="All sources", base_dir=request.data_dir, default_selection=DEFAULT_ALL_SELECTION, selection_label="Selection", is_aggregate=True),
    ]


def list_selection_dirs(request: ListSelectionDirsRequest) -> list[str]:
    if not request.base_dir.exists():
        return []
    selection_dirs: list[str] = [path.name for path in request.base_dir.iterdir() if path.is_dir()]
    selection_dirs.sort()
    return selection_dirs


def resolve_huggingface_base_dir(base_dir: Path) -> Path:
    default_dir: Path = base_dir / DEFAULT_HF_SELECTION
    if default_dir.exists():
        return default_dir
    selection_dirs: list[str] = list_selection_dirs(ListSelectionDirsRequest(base_dir=base_dir))
    if len(selection_dirs) > 0:
        return base_dir / selection_dirs[0]
    return default_dir


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
    return selection_label.replace("-", " ").title()


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


def build_bar_chart(request: ChartBuildRequest) -> Figure:
    labels, counts, percents = build_chart_data(request.report.usage, request.max_items)
    if len(labels) == 0:
        return build_empty_figure(EmptyFigureRequest(message=f"No {request.x_label.lower()} data available."))
    import plotly.graph_objects as go
    percent_labels: list[str] = [f"{value:.2f}%" for value in percents]
    fig: Figure = go.Figure(go.Bar(x=counts, y=labels, orientation="h", text=percent_labels, textposition="outside", marker_color="#2563eb", cliponaxis=False))
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


def build_family_grouped_bar_chart(request: FamilyChartBuildRequest) -> Figure:
    selected_usages: list[FeatureUsage] = list(request.report.usage)[:request.max_items]
    if len(selected_usages) == 0:
        return build_empty_figure(EmptyFigureRequest(message="No grouped family data available."))
    family_usages: list[FamilyUsage] = []
    for usage in selected_usages:
        family_name: str = resolve_family_name(usage.name)
        family_usages.append(FamilyUsage(usage=usage, family=family_name))
    family_usages.sort(key=lambda item: (resolve_family_order(item.family), -item.usage.notebook_count, item.usage.name))
    ordered_labels: list[str] = [item.usage.name for item in family_usages]
    grouped_by_family: dict[str, list[FamilyUsage]] = {}
    for family_usage in family_usages:
        grouped_by_family.setdefault(family_usage.family, []).append(family_usage)
    import plotly.graph_objects as go
    fig: Figure = go.Figure()
    ordered_families: list[str] = sorted(grouped_by_family.keys(), key=resolve_family_order)
    for family_name in ordered_families:
        family_items: list[FamilyUsage] = grouped_by_family[family_name]
        labels: list[str] = [item.usage.name for item in family_items]
        counts: list[int] = [item.usage.notebook_count for item in family_items]
        percent_labels: list[str] = [f"{item.usage.usage_percent:.2f}%" for item in family_items]
        family_color: str = FAMILY_COLOR_MAPPING.get(family_name, DEFAULT_FAMILY_COLOR)
        fig.add_trace(go.Bar(name=family_name, x=counts, y=labels, orientation="h", text=percent_labels, textposition="outside", marker_color=family_color, cliponaxis=False, hovertemplate="<b>%{y}</b><br>Notebook count: %{x}<br>Usage: %{text}<extra>%{fullData.name}</extra>"))
    fig.update_layout(
        title=request.title,
        xaxis_title="Notebook count",
        yaxis_title="Library",
        yaxis={"autorange": "reversed", "categoryorder": "array", "categoryarray": ordered_labels},
        margin={"l": 120, "r": 20, "t": 60, "b": 40},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
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
    return request.modules.html.Div([request.modules.dcc.Tabs(value=request.sources[0].key, children=tabs)])


def build_status_message(selection_dir: Path, analyzed: int, skipped: int) -> str:
    if not selection_dir.exists():
        return f"No notebooks found at {selection_dir}."
    return f"Analyzed {analyzed} notebooks. Skipped {skipped} invalid notebooks."


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
        selection_title = "All repo types" if source.key == "huggingface" else "All selections"
    if source.key == "huggingface":
        selection_root_label: str = format_selection_label(source.base_dir.name)
        selection_title = f"{selection_title} ({selection_root_label})"
    library_title: str = f"Top 20 most used libraries (unique per notebook) — {selection_title}"
    library_figure: Figure = build_bar_chart(ChartBuildRequest(report=report.library_report, title=library_title, x_label="Library", max_items=config.max_items))
    all_selection_dir: Path = source.base_dir.resolve()
    all_report: SelectionReport = report if all_selection_dir == selection_dir else build_selection_report(SelectionReportRequest(selection_dir=all_selection_dir, max_notebooks=config.max_notebooks))
    family_chart_title: str = f"Top {config.max_items} libraries grouped by family (All data) — {source.label}"
    family_figure: Figure = build_family_grouped_bar_chart(FamilyChartBuildRequest(report=all_report.library_report, title=family_chart_title, max_items=config.max_items))
    library_rows: list[dict[str, int | float | str]] = build_usage_table_rows(UsageTableRequest(report=report.library_report, include_family=True))
    extension_rows: list[dict[str, int | float | str]] = build_usage_table_rows(UsageTableRequest(report=report.extension_report))
    status_message: str = build_status_message(selection_dir=selection_dir, analyzed=report.analyzed_notebooks, skipped=report.skipped_notebooks)
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
        fallback_options: Sequence[str] | None = DEFAULT_HF_REPO_TYPES if source.key == "huggingface" else None
        include_all: bool = source.key in ("kaggle", "huggingface")
        preferred_order: Sequence[str] | None = None
        if source.key == "kaggle":
            preferred_order = KAGGLE_SELECTION_ORDER
        if source.key == "huggingface":
            preferred_order = DEFAULT_HF_REPO_TYPES
        selection_options[source.key] = resolve_selection_options(ResolveSelectionOptionsRequest(base_dir=source.base_dir, default_selection=source.default_selection, is_aggregate=source.is_aggregate, fallback_options=fallback_options, include_all=include_all, preferred_order=preferred_order))
    component_ids: dict[str, SourceComponentIds] = {source.key: build_component_ids(ComponentIdsRequest(source_key=source.key)) for source in sources}
    modules: DashModules = DashModules(dcc=dcc, html=html, dash_table=dash_table)
    app: Dash = Dash(__name__)
    app.layout = build_tabs_layout(BuildTabsLayoutRequest(modules=modules, sources=sources, selection_options=selection_options, component_ids=component_ids))
    execute_register_callbacks(RegisterCallbacksRequest(app=app, modules=modules, sources=sources, component_ids=component_ids, config=request.config))
    return app


def main() -> None:
    config: DashboardConfig = parse_dashboard_args(ParseDashboardArgsRequest(argv=sys.argv[1:]))
    ensure_plotly()
    app: Dash = build_app(BuildAppRequest(config=config))
    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == "__main__":
    main()
