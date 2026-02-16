from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import random
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class HuggingFaceSelection:
    max_notebooks: int
    output_dir: Path
    selection_label: str
    repo_types: Sequence[str]
    sort_by: str


@dataclass(frozen=True)
class HuggingFaceRepoRef:
    repo_id: str
    repo_type: str
    likes: int
    sha: str
    siblings: list[str]


@dataclass(frozen=True)
class HuggingFaceNotebookRef:
    repo_id: str
    repo_type: str
    path: str
    likes: int
    sha: str


@dataclass(frozen=True)
class HuggingFaceListRequest:
    repo_type: str
    sort_by: str
    limit: int
    full: bool


@dataclass(frozen=True)
class HuggingFaceListResult:
    repos: list[HuggingFaceRepoRef]


@dataclass(frozen=True)
class HuggingFaceDownloadResult:
    notebook_paths: list[Path]
    notebooks: list[HuggingFaceNotebookRef]


@dataclass(frozen=True)
class RetryRequest:
    operation_name: str
    action: Callable[[], object]
    max_retries: int


_BASE_URL: str = "https://huggingface.co"
_USER_AGENT: str = "py-trents"
_DEFAULT_LIMIT: int = 100
_MAX_LIST_LIMIT: int = 500
_MAX_REPO_PAGES: int = 20
_MAX_RETRIES: int = 5
_INITIAL_DELAY_SECONDS: float = 2.0
_MAX_DELAY_SECONDS: float = 60.0


def build_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    }
    token: str = os.environ.get("HF_TOKEN", "").strip()
    if token != "":
        headers["Authorization"] = f"Bearer {token}"
    return headers


def is_rate_limit_error(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        if error.code in (403, 429):
            message: str = read_error_body(error).lower()
            if "rate limit" in message or "too many requests" in message:
                return True
    return False


def read_error_body(error: HTTPError) -> str:
    try:
        body_bytes: bytes = error.read()
        return body_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


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


def request_json(url: str) -> object:
    request: Request = Request(url, headers=build_headers())
    with urlopen(request, timeout=30) as response:
        body: str = response.read().decode("utf-8")
    return json.loads(body)


def request_json_with_headers(url: str) -> tuple[object, Mapping[str, str]]:
    request: Request = Request(url, headers=build_headers())
    with urlopen(request, timeout=30) as response:
        body: str = response.read().decode("utf-8")
        headers: Mapping[str, str] = response.headers
    return json.loads(body), headers


def fetch_json(url: str) -> object:
    response_value: object = execute_with_retry(RetryRequest(operation_name="huggingface_api", action=lambda: request_json(url), max_retries=_MAX_RETRIES))
    return response_value


def fetch_json_with_headers(url: str) -> tuple[object, Mapping[str, str]]:
    response_value: object = execute_with_retry(RetryRequest(operation_name="huggingface_api", action=lambda: request_json_with_headers(url), max_retries=_MAX_RETRIES))
    if isinstance(response_value, tuple) and len(response_value) == 2:
        data_value: object = response_value[0]
        headers_value: Mapping[str, str] = response_value[1]
        return data_value, headers_value
    return {}, {}


def build_list_url(request: HuggingFaceListRequest) -> str:
    endpoint: str = resolve_list_endpoint(request.repo_type)
    params: dict[str, str | int] = {
        "limit": request.limit,
        "sort": request.sort_by,
        "direction": -1,
    }
    if request.full:
        params["full"] = "true"
    return f"{_BASE_URL}{endpoint}?{urlencode(params)}"


def extract_next_link(headers: Mapping[str, str]) -> str | None:
    link_header: str = headers.get("Link") or headers.get("link") or ""
    if link_header.strip() == "":
        return None
    parts: list[str] = link_header.split(",")
    for part in parts:
        section: str = part.strip()
        if 'rel="next"' not in section:
            continue
        url_part: str = section.split(";")[0].strip()
        if url_part.startswith("<") and url_part.endswith(">"):
            return url_part[1:-1]
    return None


def resolve_list_endpoint(repo_type: str) -> str:
    if repo_type == "model":
        return "/api/models"
    if repo_type == "dataset":
        return "/api/datasets"
    if repo_type == "space":
        return "/api/spaces"
    raise ValueError(f"Unsupported repo type: {repo_type}")


def resolve_repo_endpoint(repo_type: str, repo_id: str) -> str:
    repo_id_encoded: str = quote(repo_id, safe="/")
    if repo_type == "model":
        return f"{_BASE_URL}/api/models/{repo_id_encoded}"
    if repo_type == "dataset":
        return f"{_BASE_URL}/api/datasets/{repo_id_encoded}"
    if repo_type == "space":
        return f"{_BASE_URL}/api/spaces/{repo_id_encoded}"
    raise ValueError(f"Unsupported repo type: {repo_type}")


def extract_siblings(repo_data: Mapping[str, object]) -> list[str]:
    siblings_value: object = repo_data.get("siblings", [])
    siblings: list[str] = []
    if not isinstance(siblings_value, list):
        return siblings
    for sibling in siblings_value:
        sibling_dict: dict[str, object] | None = get_dict(sibling)
        if sibling_dict is None:
            continue
        rfilename: str = get_str(sibling_dict, "rfilename")
        path_value: str = get_str(sibling_dict, "path")
        file_name: str = rfilename if rfilename != "" else path_value
        if file_name != "":
            siblings.append(file_name)
    return siblings


def list_repos(request: HuggingFaceListRequest) -> HuggingFaceListResult:
    url: str = build_list_url(request)
    response: object = fetch_json(url)
    if not isinstance(response, list):
        return HuggingFaceListResult(repos=[])
    repos: list[HuggingFaceRepoRef] = []
    for item in response:
        repo_ref: HuggingFaceRepoRef | None = parse_repo(item, request.repo_type)
        if repo_ref is None:
            continue
        repos.append(repo_ref)
    return HuggingFaceListResult(repos=repos)


def list_repos_paginated(request: HuggingFaceListRequest, repo_limit: int) -> HuggingFaceListResult:
    repos: list[HuggingFaceRepoRef] = []
    url: str | None = build_list_url(request)
    page: int = 1
    while url is not None and len(repos) < repo_limit and page <= _MAX_REPO_PAGES:
        response_value, headers = fetch_json_with_headers(url)
        if not isinstance(response_value, list):
            break
        if len(response_value) == 0:
            break
        for item in response_value:
            repo_ref: HuggingFaceRepoRef | None = parse_repo(item, request.repo_type)
            if repo_ref is None:
                continue
            repos.append(repo_ref)
            if len(repos) >= repo_limit:
                break
        url = extract_next_link(headers)
        page += 1
    return HuggingFaceListResult(repos=repos)


def parse_repo(item: object, repo_type: str) -> HuggingFaceRepoRef | None:
    item_dict: dict[str, object] | None = get_dict(item)
    if item_dict is None:
        return None
    repo_id: str = get_str(item_dict, "id")
    if repo_id == "":
        repo_id = get_str(item_dict, "modelId")
    if repo_id == "":
        return None
    likes: int = get_int(item_dict, "likes")
    sha: str = get_str(item_dict, "sha")
    siblings: list[str] = extract_siblings(item_dict)
    return HuggingFaceRepoRef(repo_id=repo_id, repo_type=repo_type, likes=likes, sha=sha, siblings=siblings)


def get_repo_info(repo_type: str, repo_id: str) -> dict[str, object]:
    url: str = resolve_repo_endpoint(repo_type, repo_id)
    response: object = fetch_json(url)
    if isinstance(response, dict):
        return response
    return {}


def collect_repo_refs(selection: HuggingFaceSelection) -> list[HuggingFaceRepoRef]:
    repo_refs: list[HuggingFaceRepoRef] = []
    for repo_type in selection.repo_types:
        repo_limit: int = resolve_repo_limit(selection)
        list_limit: int = resolve_page_limit(repo_limit)
        print(f"Listing Hugging Face {repo_type} repos sorted by {selection.sort_by}...")
        list_request: HuggingFaceListRequest = HuggingFaceListRequest(repo_type=repo_type, sort_by=selection.sort_by, limit=list_limit, full=True)
        try:
            list_result: HuggingFaceListResult = list_repos_paginated(list_request, repo_limit)
        except HTTPError as exc:
            if exc.code == 401:
                print("Hugging Face request failed: 401 Unauthorized. Set HF_TOKEN to use the API.")
                continue
            print(f"Hugging Face list failed: {exc}")
            continue
        except (URLError, RuntimeError) as exc:
            print(f"Hugging Face list failed: {exc}")
            continue
        print(f"Fetched {len(list_result.repos)} {repo_type} repos.")
        repo_refs.extend(list_result.repos)
    return repo_refs


def resolve_repo_limit(selection: HuggingFaceSelection) -> int:
    target_limit: int = max(_DEFAULT_LIMIT, selection.max_notebooks * 30)
    return min(2000, target_limit)


def resolve_page_limit(repo_limit: int) -> int:
    return min(_DEFAULT_LIMIT, repo_limit)


def is_notebook_path(path_value: str) -> bool:
    normalized_path: str = path_value.strip()
    if normalized_path == "":
        return False
    lower_path: str = normalized_path.lower()
    if not lower_path.endswith(".ipynb"):
        return False
    if "/.ipynb_checkpoints/" in lower_path or lower_path.endswith(".ipynb_checkpoints"):
        return False
    if "/." in lower_path:
        return False
    file_name: str = normalized_path.split("/")[-1]
    if file_name.startswith("."):
        return False
    return True


def collect_notebook_refs(selection: HuggingFaceSelection) -> list[HuggingFaceNotebookRef]:
    repo_refs: list[HuggingFaceRepoRef] = collect_repo_refs(selection)
    print(f"Scanning {len(repo_refs)} repos for .ipynb files...")
    notebook_refs: list[HuggingFaceNotebookRef] = []
    seen_keys: set[str] = set()
    for index, repo in enumerate(repo_refs, start=1):
        if index == 1 or index % 25 == 0:
            print(f"Checked {index}/{len(repo_refs)} repos, found {len(notebook_refs)} notebooks so far.")
        siblings: list[str] = repo.siblings
        repo_sha: str = repo.sha
        if len(siblings) == 0:
            try:
                repo_info: dict[str, object] = get_repo_info(repo.repo_type, repo.repo_id)
            except Exception as exc:
                print(f"Hugging Face repo info failed for {repo.repo_id}: {exc}")
                continue
            siblings = extract_siblings(repo_info)
            if repo_sha == "":
                repo_sha = get_str(repo_info, "sha")
        for sibling in siblings:
            if not is_notebook_path(sibling):
                continue
            key: str = f"{repo.repo_type}:{repo.repo_id}:{sibling}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            notebook_refs.append(HuggingFaceNotebookRef(repo_id=repo.repo_id, repo_type=repo.repo_type, path=sibling, likes=repo.likes, sha=repo_sha))
    return notebook_refs


def build_raw_url(ref: HuggingFaceNotebookRef) -> str:
    repo_id_encoded: str = quote(ref.repo_id, safe="/")
    path_encoded: str = quote(ref.path, safe="/")
    revision: str = ref.sha if ref.sha != "" else "main"
    revision_encoded: str = quote(revision, safe="")
    if ref.repo_type == "dataset":
        return f"{_BASE_URL}/datasets/{repo_id_encoded}/resolve/{revision_encoded}/{path_encoded}"
    if ref.repo_type == "space":
        return f"{_BASE_URL}/spaces/{repo_id_encoded}/resolve/{revision_encoded}/{path_encoded}"
    return f"{_BASE_URL}/{repo_id_encoded}/resolve/{revision_encoded}/{path_encoded}"


def sanitize_filename_component(value: str) -> str:
    normalized_value: str = value.strip()
    if normalized_value == "":
        return ""
    normalized_value = normalized_value.replace(" ", "_")
    normalized_value = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in normalized_value)
    while "__" in normalized_value:
        normalized_value = normalized_value.replace("__", "_")
    return normalized_value.strip("_")


def build_output_path(base_dir: Path, ref: HuggingFaceNotebookRef) -> Path:
    prefix: str = sanitize_filename_component(ref.repo_id.replace("/", "__"))
    file_name: str = ref.path.split("/")[-1]
    tail: str = sanitize_filename_component(file_name)
    if tail == "":
        tail = "notebook.ipynb"
    return base_dir / f"{prefix}__{tail}"


def download_raw_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request: Request = Request(url, headers=build_headers())
    with urlopen(request, timeout=30) as response:
        content: bytes = response.read()
    output_path.write_bytes(content)


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


def download_notebooks(selection: HuggingFaceSelection) -> HuggingFaceDownloadResult:
    if selection.max_notebooks < 1:
        raise ValueError("max_notebooks must be >= 1.")
    selection.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Collecting Hugging Face notebooks ({selection.selection_label})...")
    notebook_refs: list[HuggingFaceNotebookRef] = collect_notebook_refs(selection)
    print(f"Found {len(notebook_refs)} notebook candidates. Downloading top {selection.max_notebooks}...")
    sorted_refs: list[HuggingFaceNotebookRef] = sorted(notebook_refs, key=lambda ref: (-ref.likes, ref.repo_id, ref.path))
    selected_refs: list[HuggingFaceNotebookRef] = sorted_refs[: selection.max_notebooks]
    notebook_paths: list[Path] = []
    downloaded_refs: list[HuggingFaceNotebookRef] = []
    for ref in selected_refs:
        output_path: Path = build_output_path(selection.output_dir, ref)
        if output_path.exists():
            notebook_paths.append(output_path)
            downloaded_refs.append(ref)
            continue
        raw_url: str = build_raw_url(ref)
        try:
            execute_with_retry(RetryRequest(operation_name=f"huggingface_download {ref.repo_id}", action=lambda: download_raw_file(raw_url, output_path), max_retries=_MAX_RETRIES))
        except Exception as exc:
            print(f"Failed to download {ref.repo_id}/{ref.path}: {exc}")
            continue
        if output_path.exists():
            notebook_paths.append(output_path)
            downloaded_refs.append(ref)
    print(f"Downloaded {len(notebook_paths)} notebooks to {selection.output_dir}.")
    return HuggingFaceDownloadResult(notebook_paths=notebook_paths, notebooks=downloaded_refs)
