from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from random import random
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GitHubSelection:
    max_notebooks: int
    query: str
    output_dir: Path
    selection_label: str


@dataclass(frozen=True)
class GitHubNotebookRef:
    repository_full_name: str
    path: str
    default_branch: str
    stars: int


@dataclass(frozen=True)
class GitHubSearchRequest:
    query: str
    page: int
    per_page: int


@dataclass(frozen=True)
class GitHubSearchResult:
    notebooks: list[GitHubNotebookRef]
    total_count: int


@dataclass(frozen=True)
class GitHubDownloadResult:
    notebook_paths: list[Path]
    notebooks: list[GitHubNotebookRef]


@dataclass(frozen=True)
class RetryRequest:
    operation_name: str
    action: Callable[[], object]
    max_retries: int


_BASE_SEARCH_URL: str = "https://api.github.com/search/code"
_USER_AGENT: str = "py-trents"
_DEFAULT_PER_PAGE: int = 100
_MAX_PAGES: int = 10
_MAX_RETRIES: int = 5
_INITIAL_DELAY_SECONDS: float = 2.0
_MAX_DELAY_SECONDS: float = 60.0


def build_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _USER_AGENT,
    }
    token: str = os.environ.get("GITHUB_TOKEN", "").strip()
    if token != "":
        headers["Authorization"] = f"Bearer {token}"
    return headers


def build_search_url(request: GitHubSearchRequest) -> str:
    params: dict[str, str | int] = {"q": request.query, "page": request.page, "per_page": request.per_page}
    return f"{_BASE_SEARCH_URL}?{urlencode(params)}"


def read_error_body(error: HTTPError) -> str:
    try:
        body_bytes: bytes = error.read()
        return body_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        if error.code in (403, 429):
            remaining: str | None = error.headers.get("X-RateLimit-Remaining")
            if remaining == "0":
                return True
            message: str = read_error_body(error).lower()
            if "rate limit" in message:
                return True
    return False


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


def request_json(url: str) -> dict[str, object]:
    request: Request = Request(url, headers=build_headers())
    with urlopen(request, timeout=30) as response:
        body: str = response.read().decode("utf-8")
    data: object = json.loads(body)
    if isinstance(data, dict):
        return data
    return {}


def fetch_json(url: str) -> dict[str, object]:
    response_value: object = execute_with_retry(RetryRequest(operation_name="github_api", action=lambda: request_json(url), max_retries=_MAX_RETRIES))
    if isinstance(response_value, dict):
        return response_value
    return {}


def extract_total_count(response: Mapping[str, object]) -> int:
    total_value: object = response.get("total_count", 0)
    if isinstance(total_value, int):
        return total_value
    try:
        return int(total_value)
    except Exception:
        return 0


def extract_items(response: Mapping[str, object]) -> list[object]:
    items_value: object = response.get("items", [])
    if isinstance(items_value, list):
        return items_value
    return []


def get_str(data: Mapping[str, object], key: str) -> str:
    value: object = data.get(key)
    if isinstance(value, str):
        return value
    return ""


def get_int(data: Mapping[str, object], key: str) -> int:
    value: object = data.get(key)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return 0


def get_dict(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    return None


def parse_item(item: object) -> GitHubNotebookRef | None:
    item_dict: dict[str, object] | None = get_dict(item)
    if item_dict is None:
        return None
    path_value: str = get_str(item_dict, "path").lstrip("/")
    if should_skip_path(path_value):
        return None
    repo_value: object = item_dict.get("repository")
    repo_dict: dict[str, object] | None = get_dict(repo_value)
    if repo_dict is None:
        return None
    full_name: str = get_str(repo_dict, "full_name")
    default_branch: str = get_str(repo_dict, "default_branch")
    stars: int = get_int(repo_dict, "stargazers_count")
    if full_name == "" or path_value == "":
        return None
    return GitHubNotebookRef(repository_full_name=full_name, path=path_value, default_branch=default_branch, stars=stars)


def should_skip_path(path_value: str) -> bool:
    normalized_path: str = path_value.strip()
    if normalized_path == "":
        return True
    lower_path: str = normalized_path.lower()
    if not lower_path.endswith(".ipynb"):
        return True
    if "/.ipynb_checkpoints/" in lower_path or lower_path.endswith(".ipynb_checkpoints"):
        return True
    if "/." in lower_path:
        return True
    file_name: str = normalized_path.split("/")[-1]
    if file_name.startswith("."):
        return True
    return False


def parse_search_items(items: Sequence[object]) -> list[GitHubNotebookRef]:
    notebook_refs: list[GitHubNotebookRef] = []
    for item in items:
        parsed: GitHubNotebookRef | None = parse_item(item)
        if parsed is None:
            continue
        notebook_refs.append(parsed)
    return notebook_refs


def search_notebooks(selection: GitHubSelection) -> GitHubSearchResult:
    page: int = 1
    seen_keys: set[str] = set()
    notebook_refs: list[GitHubNotebookRef] = []
    target_count: int = min(selection.max_notebooks * 3, _DEFAULT_PER_PAGE * _MAX_PAGES)
    total_count: int = 0
    while page <= _MAX_PAGES and len(notebook_refs) < target_count:
        search_request: GitHubSearchRequest = GitHubSearchRequest(query=selection.query, page=page, per_page=_DEFAULT_PER_PAGE)
        response: dict[str, object] = fetch_json(build_search_url(search_request))
        total_count = extract_total_count(response)
        items: list[object] = extract_items(response)
        page_refs: list[GitHubNotebookRef] = parse_search_items(items)
        for notebook in page_refs:
            key: str = f"{notebook.repository_full_name}:{notebook.path}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            notebook_refs.append(notebook)
        if len(items) < _DEFAULT_PER_PAGE:
            break
        if total_count > 0 and page * _DEFAULT_PER_PAGE >= min(total_count, _DEFAULT_PER_PAGE * _MAX_PAGES):
            break
        page += 1
    return GitHubSearchResult(notebooks=notebook_refs, total_count=total_count)


def split_repository(full_name: str) -> tuple[str, str]:
    owner: str
    repo: str
    owner, separator, repo = full_name.partition("/")
    if separator == "":
        return full_name, "unknown"
    return owner, repo


def resolve_default_branch(default_branch: str) -> str:
    if default_branch.strip() != "":
        return default_branch
    return "main"


def build_raw_url(ref: GitHubNotebookRef) -> str:
    owner: str
    repo: str
    owner, repo = split_repository(ref.repository_full_name)
    branch: str = resolve_default_branch(ref.default_branch)
    owner_encoded: str = quote(owner, safe="")
    repo_encoded: str = quote(repo, safe="")
    branch_encoded: str = quote(branch, safe="")
    path_encoded: str = quote(ref.path, safe="/")
    return f"https://raw.githubusercontent.com/{owner_encoded}/{repo_encoded}/{branch_encoded}/{path_encoded}"


def build_output_path(base_dir: Path, ref: GitHubNotebookRef) -> Path:
    owner: str
    repo: str
    owner, repo = split_repository(ref.repository_full_name)
    file_name: str = ref.path.split("/")[-1]
    prefix: str = sanitize_filename_component(f"{owner}__{repo}")
    tail: str = sanitize_filename_component(file_name)
    if tail == "":
        tail = "notebook.ipynb"
    return base_dir / f"{prefix}__{tail}"


def sanitize_filename_component(value: str) -> str:
    normalized_value: str = value.strip()
    if normalized_value == "":
        return ""
    normalized_value = normalized_value.replace(" ", "_")
    normalized_value = re.sub(r"[^A-Za-z0-9._-]", "_", normalized_value)
    normalized_value = re.sub(r"_+", "_", normalized_value)
    return normalized_value.strip("_")


def request_download(url: str) -> bytes:
    request: Request = Request(url, headers=build_headers())
    with urlopen(request, timeout=30) as response:
        return response.read()


def download_raw_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content: bytes = request_download(url)
    output_path.write_bytes(content)


def download_notebooks(selection: GitHubSelection) -> GitHubDownloadResult:
    if selection.max_notebooks < 1:
        raise ValueError("max_notebooks must be >= 1.")
    selection.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        search_result: GitHubSearchResult = search_notebooks(selection)
    except HTTPError as exc:
        if exc.code == 401:
            print("GitHub search failed: 401 Unauthorized. Set GITHUB_TOKEN to use the API.")
        else:
            print(f"GitHub search failed: {exc}")
        return GitHubDownloadResult(notebook_paths=[], notebooks=[])
    except (URLError, RuntimeError) as exc:
        print(f"GitHub search failed: {exc}")
        return GitHubDownloadResult(notebook_paths=[], notebooks=[])
    sorted_refs: list[GitHubNotebookRef] = sorted(search_result.notebooks, key=lambda ref: (-ref.stars, ref.repository_full_name, ref.path))
    selected_refs: list[GitHubNotebookRef] = sorted_refs[: selection.max_notebooks]
    notebook_paths: list[Path] = []
    downloaded_refs: list[GitHubNotebookRef] = []
    for ref in selected_refs:
        output_path: Path = build_output_path(selection.output_dir, ref)
        if output_path.exists():
            notebook_paths.append(output_path)
            downloaded_refs.append(ref)
            continue
        raw_url: str = build_raw_url(ref)
        try:
            execute_with_retry(RetryRequest(operation_name=f"github_download {ref.repository_full_name}", action=lambda: download_raw_file(raw_url, output_path), max_retries=_MAX_RETRIES))
        except Exception as exc:
            print(f"Failed to download {ref.repository_full_name}/{ref.path}: {exc}")
            continue
        if output_path.exists():
            notebook_paths.append(output_path)
            downloaded_refs.append(ref)
    return GitHubDownloadResult(notebook_paths=notebook_paths, notebooks=downloaded_refs)
