from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaggle.api.kaggle_api_extended import KaggleApi


class SelectionType(Enum):
    TOP = "top"
    LATEST = "latest"
    HOTNESS = "hotness"


@dataclass(frozen=True)
class KaggleSelection:
    selection_type: SelectionType
    max_notebooks: int
    competition: str | None
    output_dir: Path
    sort_by: str | None


@dataclass(frozen=True)
class KaggleNotebookRef:
    ref: str
    title: str
    author: str


@dataclass(frozen=True)
class KaggleListRequest:
    api: KaggleApi
    selection: KaggleSelection
    page: int
    page_size: int


@dataclass(frozen=True)
class KaggleListResult:
    notebooks: list[KaggleNotebookRef]
    raw_count: int


@dataclass(frozen=True)
class RetryRequest:
    operation_name: str
    action: Callable[[], object]
    max_retries: int


@dataclass(frozen=True)
class KaggleDownloadRequest:
    selection: KaggleSelection


@dataclass(frozen=True)
class KaggleDownloadResult:
    notebook_paths: list[Path]
    notebooks: list[KaggleNotebookRef]


@dataclass(frozen=True)
class NotebookPathRequest:
    output_dir: Path
    notebook_ref: KaggleNotebookRef


_DEFAULT_SORT_BY: dict[SelectionType, str] = {
    SelectionType.TOP: "voteCount",
    SelectionType.LATEST: "dateCreated",
    SelectionType.HOTNESS: "hotness",
}
_DEFAULT_PAGE_SIZE: int = 50
_MAX_RETRIES: int = 5
_INITIAL_DELAY_SECONDS: float = 2.0
_MAX_DELAY_SECONDS: float = 60.0


def create_api() -> KaggleApi:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Kaggle API not installed. Run `pip install kaggle`.") from exc
    api: KaggleApi = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError("Kaggle authentication failed. Check kaggle.json.") from exc
    return api


def resolve_sort_by(selection: KaggleSelection) -> str:
    sort_by: str | None = selection.sort_by
    if sort_by is not None and sort_by.strip() != "":
        return sort_by.strip()
    return _DEFAULT_SORT_BY[selection.selection_type]


def is_rate_limit_error(exc: Exception) -> bool:
    response: object | None = getattr(exc, "response", None)
    status_code: int | None = getattr(response, "status_code", None) if response is not None else None
    if status_code == 429:
        return True
    message: str = str(exc)
    return "429" in message and "Too Many Requests" in message


def calculate_backoff_seconds(attempt: int) -> float:
    base_delay: float = _INITIAL_DELAY_SECONDS * (2 ** (attempt - 1))
    jitter: float = random()
    delay: float = min(_MAX_DELAY_SECONDS, base_delay + jitter)
    return delay


def execute_with_retry(request: RetryRequest) -> object:
    for attempt in range(1, request.max_retries + 1):
        try:
            return request.action()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt >= request.max_retries:
                raise
            delay_seconds: float = calculate_backoff_seconds(attempt)
            print(f"Rate limit hit during {request.operation_name}. Retrying in {delay_seconds:.1f}s.")
            time.sleep(delay_seconds)
    raise RuntimeError(f"Failed to execute {request.operation_name} after retries.")


def list_notebooks(request: KaggleListRequest) -> KaggleListResult:
    sort_by: str = resolve_sort_by(request.selection)
    list_params: dict[str, object] = {"sort_by": sort_by, "page": request.page, "page_size": request.page_size}
    if request.selection.competition is not None:
        list_params["competition"] = request.selection.competition
    try:
        kernels_value: object = execute_with_retry(RetryRequest(operation_name="kernels_list", action=lambda: request.api.kernels_list(**list_params), max_retries=_MAX_RETRIES))
    except Exception as exc:
        raise RuntimeError("Failed to list Kaggle notebooks after retries.") from exc
    kernels: list[object] = kernels_value if isinstance(kernels_value, list) else []
    notebook_refs: list[KaggleNotebookRef] = build_notebook_refs(kernels)
    return KaggleListResult(notebooks=notebook_refs, raw_count=len(kernels))


def build_notebook_refs(kernels: Sequence[object]) -> list[KaggleNotebookRef]:
    notebook_refs: list[KaggleNotebookRef] = []
    for kernel in kernels:
        notebook_ref: KaggleNotebookRef | None = extract_kernel_metadata(kernel)
        if notebook_ref is None:
            continue
        notebook_refs.append(notebook_ref)
    return notebook_refs


def extract_kernel_metadata(kernel: object) -> KaggleNotebookRef | None:
    if isinstance(kernel, Mapping):
        ref_value_object: object = kernel.get("ref", "")
        title_value_object: object = kernel.get("title", "")
        author_value_object: object = kernel.get("author", "")
        language_value_object: object | None = None
        for key in ("language", "languageName", "language_name"):
            if key in kernel:
                language_value_object = kernel.get(key)
                break
    else:
        ref_value_object = getattr(kernel, "ref", "")
        title_value_object = getattr(kernel, "title", "")
        author_value_object = getattr(kernel, "author", "")
        language_value_object = None
        for attr in ("language", "languageName", "language_name"):
            if hasattr(kernel, attr):
                language_value_object = getattr(kernel, attr)
                break
    ref_value: str = str(ref_value_object) if ref_value_object is not None else ""
    title_value: str = str(title_value_object) if title_value_object is not None else ""
    author_value: str = str(author_value_object) if author_value_object is not None else ""
    language_value: str = str(language_value_object).strip().lower() if language_value_object is not None else ""
    if language_value != "" and language_value != "python":
        return None
    if ref_value == "":
        return None
    return KaggleNotebookRef(ref=ref_value, title=title_value, author=author_value)


def resolve_notebook_path(request: NotebookPathRequest) -> Path | None:
    slug: str = request.notebook_ref.ref.split("/")[-1]
    expected_path: Path = request.output_dir / f"{slug}.ipynb"
    if expected_path.exists():
        return expected_path
    candidates: list[Path] = sorted(request.output_dir.glob(f"{slug}*.ipynb"))
    if len(candidates) > 0:
        return candidates[0]
    return None


def cleanup_non_ipynb_files(request: NotebookPathRequest) -> bool:
    slug: str = request.notebook_ref.ref.split("/")[-1]
    candidates: list[Path] = sorted(request.output_dir.glob(f"{slug}.*"))
    removed_any: bool = False
    for candidate in candidates:
        if candidate.suffix != ".ipynb":
            try:
                candidate.unlink()
                removed_any = True
            except OSError:
                continue
    return removed_any


def resolve_page_size(selection: KaggleSelection) -> int:
    if selection.max_notebooks > _DEFAULT_PAGE_SIZE:
        return min(selection.max_notebooks, 100)
    return _DEFAULT_PAGE_SIZE


def download_notebooks(request: KaggleDownloadRequest) -> KaggleDownloadResult:
    selection: KaggleSelection = request.selection
    if selection.max_notebooks < 1:
        raise ValueError("max_notebooks must be >= 1.")
    selection.output_dir.mkdir(parents=True, exist_ok=True)
    api: KaggleApi = create_api()
    page: int = 1
    page_size: int = resolve_page_size(selection)
    notebook_refs: list[KaggleNotebookRef] = []
    seen_refs: set[str] = set()
    notebook_paths: list[Path] = []
    while len(notebook_paths) < selection.max_notebooks:
        try:
            list_result: KaggleListResult = list_notebooks(KaggleListRequest(api=api, selection=selection, page=page, page_size=page_size))
        except RuntimeError as exc:
            print(f"{exc}")
            break
        if list_result.raw_count == 0:
            break
        for notebook in list_result.notebooks:
            if notebook.ref in seen_refs:
                continue
            seen_refs.add(notebook.ref)
            notebook_refs.append(notebook)
            cached_path: Path | None = resolve_notebook_path(NotebookPathRequest(output_dir=selection.output_dir, notebook_ref=notebook))
            if cached_path is not None:
                if cached_path not in notebook_paths:
                    notebook_paths.append(cached_path)
                if len(notebook_paths) >= selection.max_notebooks:
                    break
                continue
            removed_non_ipynb: bool = cleanup_non_ipynb_files(NotebookPathRequest(output_dir=selection.output_dir, notebook_ref=notebook))
            if removed_non_ipynb:
                print(f"Skipped non-ipynb notebook for {notebook.ref}")
                continue
            try:
                execute_with_retry(RetryRequest(operation_name=f"kernels_pull {notebook.ref}", action=lambda: api.kernels_pull(kernel=notebook.ref, path=str(selection.output_dir), metadata=False, quiet=True), max_retries=_MAX_RETRIES))
            except Exception as exc:
                print(f"Failed to download {notebook.ref}: {exc}")
                continue
            resolved_path: Path | None = resolve_notebook_path(NotebookPathRequest(output_dir=selection.output_dir, notebook_ref=notebook))
            if resolved_path is None:
                removed_non_ipynb: bool = cleanup_non_ipynb_files(NotebookPathRequest(output_dir=selection.output_dir, notebook_ref=notebook))
                if removed_non_ipynb:
                    print(f"Skipped non-ipynb notebook for {notebook.ref}")
                else:
                    print(f"Missing notebook file for {notebook.ref}")
                continue
            if resolved_path not in notebook_paths:
                notebook_paths.append(resolved_path)
            if len(notebook_paths) >= selection.max_notebooks:
                break
        page += 1
        if list_result.raw_count < page_size:
            break
    if len(notebook_refs) == 0:
        return KaggleDownloadResult(notebook_paths=[], notebooks=[])
    return KaggleDownloadResult(notebook_paths=notebook_paths, notebooks=notebook_refs)
