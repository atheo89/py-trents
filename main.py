from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from github_client import GitHubSelection, download_notebooks as download_github_notebooks
from huggingface_client import HuggingFaceSelection, download_notebooks as download_huggingface_notebooks
from kaggle_client import KaggleDownloadRequest, KaggleSelection, SelectionType, download_notebooks as download_kaggle_notebooks
from library_normalizer import NormalizeImportsRequest, default_normalizer_config, normalize_imports
from metrics import NotebookFeatureUsage, UsageMetricsRequest, UsageReport, compute_usage_metrics
from notebook_parser import NotebookParseRequest, parse_notebook
from visualization import PlotReportRequest, render_usage_report


@dataclass(frozen=True)
class AppConfig:
    source: "SourceType"
    selection_type: SelectionType
    max_notebooks: int
    competition: str | None
    output_dir: Path
    sort_by: str | None
    github_query: str
    hf_repo_types: list[str]
    hf_sort: str


@dataclass(frozen=True)
class ParseArgsRequest:
    argv: Sequence[str]


@dataclass(frozen=True)
class AppReport:
    library_report: UsageReport
    extension_report: UsageReport
    source_name: str
    selection_label: str
    requested_max: int


class SourceType(Enum):
    KAGGLE = "kaggle"
    GITHUB = "github"
    HUGGINGFACE = "huggingface"


DEFAULT_GITHUB_QUERY: str = "extension:ipynb"
DEFAULT_HF_SORT: str = "likes"
DEFAULT_HF_REPO_TYPES: list[str] = ["model", "dataset", "space"]


def parse_selection_type(raw_value: str) -> SelectionType:
    normalized_value: str = raw_value.strip().lower()
    for selection_type in SelectionType:
        if selection_type.value == normalized_value:
            return selection_type
    raise ValueError(f"Unsupported selection type: {raw_value}")


def parse_source_type(raw_value: str) -> SourceType:
    normalized_value: str = raw_value.strip().lower()
    for source_type in SourceType:
        if source_type.value == normalized_value:
            return source_type
    raise ValueError(f"Unsupported source: {raw_value}")


def format_source_label(source_name: str) -> str:
    if source_name.lower() == "github":
        return "GitHub"
    if source_name.lower() == "huggingface":
        return "Hugging Face"
    return source_name.title()


def parse_hf_repo_types(raw_value: str) -> list[str]:
    normalized_value: str = raw_value.strip()
    if normalized_value == "":
        return DEFAULT_HF_REPO_TYPES
    raw_types: list[str] = [value.strip().lower() for value in normalized_value.split(",") if value.strip() != ""]
    resolved_types: list[str] = []
    for repo_type in raw_types:
        if repo_type in ("models", "model"):
            resolved_types.append("model")
        elif repo_type in ("datasets", "dataset"):
            resolved_types.append("dataset")
        elif repo_type in ("spaces", "space"):
            resolved_types.append("space")
        else:
            raise ValueError(f"Unsupported Hugging Face repo type: {repo_type}")
    if len(resolved_types) == 0:
        return DEFAULT_HF_REPO_TYPES
    return list(dict.fromkeys(resolved_types))


def build_hf_selection_label(sort_by: str) -> str:
    normalized_sort: str = sort_by.strip().lower()
    if normalized_sort == "":
        normalized_sort = DEFAULT_HF_SORT
    if normalized_sort == "likes":
        return "most-liked"
    if normalized_sort == "downloads":
        return "most-downloaded"
    if normalized_sort in ("trending", "trending_score"):
        return "trending"
    return f"most-{normalized_sort}"


def parse_args(request: ParseArgsRequest) -> AppConfig:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Analyze notebooks for library usage.")
    parser.add_argument("--source", choices=[source.value for source in SourceType], default=SourceType.KAGGLE.value)
    parser.add_argument("--selection", choices=[selection.value for selection in SelectionType], default=SelectionType.TOP.value)
    parser.add_argument("--max-notebooks", type=int, default=50)
    parser.add_argument("--competition", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--sort-by", type=str, default="")
    parser.add_argument("--github-query", type=str, default=DEFAULT_GITHUB_QUERY)
    parser.add_argument("--hf-repo-types", type=str, default=",".join(DEFAULT_HF_REPO_TYPES))
    parser.add_argument("--hf-sort", type=str, default=DEFAULT_HF_SORT)
    args: argparse.Namespace = parser.parse_args(request.argv)
    source: SourceType = parse_source_type(str(args.source))
    selection_type: SelectionType = parse_selection_type(str(args.selection))
    competition_value: str = str(args.competition)
    competition: str | None = competition_value.strip() if competition_value.strip() != "" else None
    sort_by_value: str = str(args.sort_by)
    sort_by: str | None = sort_by_value.strip() if sort_by_value.strip() != "" else None
    output_dir: Path = Path(str(args.output_dir)).expanduser().resolve()
    max_notebooks: int = int(args.max_notebooks)
    github_query: str = str(args.github_query).strip()
    if github_query == "":
        github_query = DEFAULT_GITHUB_QUERY
    hf_repo_types: list[str] = parse_hf_repo_types(str(args.hf_repo_types))
    hf_sort: str = str(args.hf_sort).strip().lower()
    if hf_sort == "":
        hf_sort = DEFAULT_HF_SORT
    return AppConfig(source=source, selection_type=selection_type, max_notebooks=max_notebooks, competition=competition, output_dir=output_dir, sort_by=sort_by, github_query=github_query, hf_repo_types=hf_repo_types, hf_sort=hf_sort)


def resolve_kaggle_output_dir(base_dir: Path, selection_type: SelectionType) -> Path:
    effective_base: Path = base_dir
    if base_dir.name == "github_notebooks":
        effective_base = base_dir.parent
    if effective_base.name != "kaggle_notebooks":
        effective_base = effective_base / "kaggle_notebooks"
    return effective_base / selection_type.value


def resolve_github_output_dir(base_dir: Path, selection_label: str) -> Path:
    effective_base: Path = base_dir
    if base_dir.name == "kaggle_notebooks":
        effective_base = base_dir.parent
    if effective_base.name != "github_notebooks":
        effective_base = effective_base / "github_notebooks"
    return effective_base / selection_label


def resolve_huggingface_output_dir(base_dir: Path, selection_label: str) -> Path:
    effective_base: Path = base_dir
    if base_dir.name in ("kaggle_notebooks", "github_notebooks"):
        effective_base = base_dir.parent
    if effective_base.name != "huggingface_notebooks":
        effective_base = effective_base / "huggingface_notebooks"
    return effective_base / selection_label


def run_kaggle_pipeline(config: AppConfig) -> AppReport:
    if config.max_notebooks < 1:
        raise ValueError("max_notebooks must be >= 1.")
    selection_output_dir: Path = resolve_kaggle_output_dir(config.output_dir, config.selection_type)
    selection: KaggleSelection = KaggleSelection(selection_type=config.selection_type, max_notebooks=config.max_notebooks, competition=config.competition, output_dir=selection_output_dir, sort_by=config.sort_by)
    download_result = download_kaggle_notebooks(KaggleDownloadRequest(selection=selection))
    normalizer_config = default_normalizer_config()
    library_usages: list[NotebookFeatureUsage] = []
    extension_usages: list[NotebookFeatureUsage] = []
    for notebook_path in download_result.notebook_paths:
        try:
            parse_result = parse_notebook(NotebookParseRequest(notebook_path=notebook_path))
        except RuntimeError as exc:
            print(f"Skipped invalid notebook {notebook_path}: {exc}")
            continue
        normalize_result = normalize_imports(NormalizeImportsRequest(raw_imports=parse_result.imports, config=normalizer_config))
        library_usages.append(NotebookFeatureUsage(notebook_path=notebook_path, features=normalize_result.normalized_imports))
        extension_usages.append(NotebookFeatureUsage(notebook_path=notebook_path, features=parse_result.extensions))
    library_report: UsageReport = compute_usage_metrics(UsageMetricsRequest(notebook_usages=library_usages))
    extension_report: UsageReport = compute_usage_metrics(UsageMetricsRequest(notebook_usages=extension_usages))
    return AppReport(library_report=library_report, extension_report=extension_report, source_name=config.source.value, selection_label=config.selection_type.value, requested_max=config.max_notebooks)


def collect_valid_usages(notebook_paths: Sequence[Path], normalizer_config: object, library_usages: list[NotebookFeatureUsage], extension_usages: list[NotebookFeatureUsage], processed_paths: set[Path], target_count: int) -> bool:
    processed_any: bool = False
    for notebook_path in notebook_paths:
        if notebook_path in processed_paths:
            continue
        processed_paths.add(notebook_path)
        processed_any = True
        try:
            parse_result = parse_notebook(NotebookParseRequest(notebook_path=notebook_path))
        except RuntimeError as exc:
            print(f"Skipped invalid notebook {notebook_path}: {exc}")
            continue
        normalize_result = normalize_imports(NormalizeImportsRequest(raw_imports=parse_result.imports, config=normalizer_config))
        library_usages.append(NotebookFeatureUsage(notebook_path=notebook_path, features=normalize_result.normalized_imports))
        extension_usages.append(NotebookFeatureUsage(notebook_path=notebook_path, features=parse_result.extensions))
        if len(library_usages) >= target_count:
            break
    return processed_any


def resolve_github_target_max(requested_max: int, attempt: int) -> int:
    buffer_size: int = max(5, requested_max // 5)
    return min(1000, requested_max + buffer_size * attempt)


def resolve_hf_target_max(requested_max: int, attempt: int) -> int:
    buffer_size: int = max(5, requested_max // 5)
    return requested_max + buffer_size * attempt

def run_github_pipeline(config: AppConfig) -> AppReport:
    if config.max_notebooks < 1:
        raise ValueError("max_notebooks must be >= 1.")
    selection_label: str = "most-starred"
    selection_output_dir: Path = resolve_github_output_dir(config.output_dir, selection_label)
    normalizer_config = default_normalizer_config()
    library_usages: list[NotebookFeatureUsage] = []
    extension_usages: list[NotebookFeatureUsage] = []
    processed_paths: set[Path] = set()
    attempt: int = 1
    max_attempts: int = 5
    while len(library_usages) < config.max_notebooks and attempt <= max_attempts:
        target_max: int = resolve_github_target_max(config.max_notebooks, attempt)
        selection: GitHubSelection = GitHubSelection(max_notebooks=target_max, query=config.github_query, output_dir=selection_output_dir, selection_label=selection_label)
        download_result = download_github_notebooks(selection)
        processed_any: bool = collect_valid_usages(download_result.notebook_paths, normalizer_config, library_usages, extension_usages, processed_paths, config.max_notebooks)
        if not processed_any:
            break
        attempt += 1
    library_report: UsageReport = compute_usage_metrics(UsageMetricsRequest(notebook_usages=library_usages))
    extension_report: UsageReport = compute_usage_metrics(UsageMetricsRequest(notebook_usages=extension_usages))
    return AppReport(library_report=library_report, extension_report=extension_report, source_name=config.source.value, selection_label=selection_label, requested_max=config.max_notebooks)


def run_huggingface_pipeline(config: AppConfig) -> AppReport:
    if config.max_notebooks < 1:
        raise ValueError("max_notebooks must be >= 1.")
    selection_label: str = build_hf_selection_label(config.hf_sort)
    selection_output_dir: Path = resolve_huggingface_output_dir(config.output_dir, selection_label)
    normalizer_config = default_normalizer_config()
    library_usages: list[NotebookFeatureUsage] = []
    extension_usages: list[NotebookFeatureUsage] = []
    processed_paths: set[Path] = set()
    attempt: int = 1
    max_attempts: int = 5
    while len(library_usages) < config.max_notebooks and attempt <= max_attempts:
        target_max: int = resolve_hf_target_max(config.max_notebooks, attempt)
        selection: HuggingFaceSelection = HuggingFaceSelection(max_notebooks=target_max, output_dir=selection_output_dir, selection_label=selection_label, repo_types=config.hf_repo_types, sort_by=config.hf_sort)
        download_result = download_huggingface_notebooks(selection)
        processed_any: bool = collect_valid_usages(download_result.notebook_paths, normalizer_config, library_usages, extension_usages, processed_paths, config.max_notebooks)
        if not processed_any:
            break
        attempt += 1
    library_report: UsageReport = compute_usage_metrics(UsageMetricsRequest(notebook_usages=library_usages))
    extension_report: UsageReport = compute_usage_metrics(UsageMetricsRequest(notebook_usages=extension_usages))
    return AppReport(library_report=library_report, extension_report=extension_report, source_name=config.source.value, selection_label=selection_label, requested_max=config.max_notebooks)


def run_pipeline(config: AppConfig) -> AppReport:
    if config.source == SourceType.GITHUB:
        return run_github_pipeline(config)
    if config.source == SourceType.HUGGINGFACE:
        return run_huggingface_pipeline(config)
    return run_kaggle_pipeline(config)


def print_report(report: AppReport) -> None:
    total_notebooks: int = report.library_report.total_notebooks
    selection_label: str = report.selection_label.replace("-", " ").title()
    source_label: str = format_source_label(report.source_name)
    print(f"{source_label} notebook usage — Selection: {selection_label}, Requested: {report.requested_max}, Analyzed: {total_notebooks}")
    if total_notebooks == 0:
        print("No notebooks were analyzed.")
        return
    print("Top libraries:")
    for usage in report.library_report.usage:
        print(f"{usage.name}: {usage.notebook_count} notebooks ({usage.usage_percent:.2f}%)")
    if len(report.extension_report.usage) > 0:
        print("Top Jupyter extensions:")
        for usage in report.extension_report.usage:
            print(f"{usage.name}: {usage.notebook_count} notebooks ({usage.usage_percent:.2f}%)")


def render_plots(report: AppReport) -> None:
    selection_label: str = report.selection_label
    source_label: str = format_source_label(report.source_name).lower()
    total_notebooks: int = report.library_report.total_notebooks
    library_title: str = f"Most used python libraries (unique per notebook) Analyzed {total_notebooks} {selection_label} {source_label} notebooks"
    library_max_items: int = len(report.library_report.usage)
    try:
        render_usage_report(PlotReportRequest(report=report.library_report, title=library_title, max_items=library_max_items, x_label="Library"))
    except RuntimeError as exc:
        print(str(exc))
        return
    if len(report.extension_report.usage) > 0:
        extension_title: str = f"Most used Jupyter extensions (unique per notebook) Analyzed {total_notebooks} {selection_label} {source_label} notebooks"
        extension_max_items: int = len(report.extension_report.usage)
        try:
            render_usage_report(PlotReportRequest(report=report.extension_report, title=extension_title, max_items=extension_max_items, x_label="Extension"))
        except RuntimeError as exc:
            print(str(exc))


def main() -> None:
    config: AppConfig = parse_args(ParseArgsRequest(argv=sys.argv[1:]))
    report: AppReport = run_pipeline(config)
    print_report(report)
    render_plots(report)


if __name__ == "__main__":
    main()

