from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import tarfile
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_OWNER: str = "opendatahub-io"
DEFAULT_REPO: str = "notebooks"
DEFAULT_BRANCH: str = "main"
DEFAULT_OUTPUT_PATH: Path = Path("data/odh_notebooks_pyproject_dependency_index.csv")


def ensure_pandas() -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Pandas is required for this script. Run `pip install pandas` or use your project virtualenv.") from exc
    _ = pd


@dataclass(frozen=True)
class AppConfig:
    owner: str
    repo: str
    branch: str
    output_csv: Path
    include_optional: bool
    include_poetry: bool
    local_repo_path: Path | None


@dataclass(frozen=True)
class PyprojectFile:
    path: str
    flavor: str
    content: str


@dataclass(frozen=True)
class PackageOccurrence:
    package_name: str
    flavor: str
    pyproject_path: str


def parse_cli_args(argv: Sequence[str]) -> AppConfig:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Build a package-to-flavor dependency index from pyproject.toml files in a GitHub repository.",
    )
    parser.add_argument("--owner", type=str, default=DEFAULT_OWNER)
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH)
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--exclude-optional", action="store_true")
    parser.add_argument("--exclude-poetry", action="store_true")
    parser.add_argument("--local-repo-path", type=str, default="")
    args: argparse.Namespace = parser.parse_args(argv)
    owner: str = str(args.owner).strip()
    repo: str = str(args.repo).strip()
    branch: str = str(args.branch).strip()
    output_csv: Path = Path(str(args.output_csv)).expanduser().resolve()
    include_optional: bool = not bool(args.exclude_optional)
    include_poetry: bool = not bool(args.exclude_poetry)
    local_repo_path_raw: str = str(args.local_repo_path).strip()
    local_repo_path: Path | None = None
    if local_repo_path_raw != "":
        local_repo_path = Path(local_repo_path_raw).expanduser().resolve()
    if owner == "" or repo == "" or branch == "":
        raise ValueError("owner, repo, and branch must be non-empty.")
    return AppConfig(
        owner=owner,
        repo=repo,
        branch=branch,
        output_csv=output_csv,
        include_optional=include_optional,
        include_poetry=include_poetry,
        local_repo_path=local_repo_path,
    )


def resolve_flavor_from_path(pyproject_path: str) -> str:
    parts: list[str] = [part for part in pyproject_path.split("/") if part != ""]
    if len(parts) <= 1:
        return "."
    return "/".join(parts[:-1])


def is_within_directory(target_path: Path, expected_parent: Path) -> bool:
    try:
        target_path.resolve().relative_to(expected_parent.resolve())
        return True
    except ValueError:
        return False


def download_repository_to_temp(config: AppConfig) -> tuple[Path, Path]:
    local_temp_base: Path = (Path.cwd() / ".tmp").resolve()
    local_temp_base.mkdir(parents=True, exist_ok=True)
    temp_root: Path = Path(tempfile.mkdtemp(prefix=f"{config.repo}-pyprojects-", dir=str(local_temp_base)))
    archive_url: str = f"https://codeload.github.com/{quote(config.owner, safe='')}/{quote(config.repo, safe='')}/tar.gz/refs/heads/{quote(config.branch, safe='')}"
    request: Request = Request(archive_url, headers={"User-Agent": "py-trents-odh-indexer"})
    try:
        with urlopen(request, timeout=60) as response:
            archive_bytes: bytes = response.read()
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise RuntimeError(f"Failed to download repository archive from {archive_url}")
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            for member in archive.getmembers():
                member_path: Path = temp_root / member.name
                if not is_within_directory(member_path, temp_root):
                    raise RuntimeError("Unsafe path encountered while extracting repository archive.")
            try:
                archive.extractall(path=temp_root, filter="data")
            except TypeError:
                archive.extractall(path=temp_root)
    except Exception as exc:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise RuntimeError("Failed to extract repository archive.") from exc
    candidate_directories: list[Path] = [path for path in temp_root.iterdir() if path.is_dir()]
    if len(candidate_directories) == 0:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise RuntimeError("Archive extraction did not produce a repository directory.")
    repo_dir: Path = sorted(candidate_directories, key=lambda path: path.name)[0]
    return temp_root, repo_dir


def collect_pyproject_files_from_directory(repo_dir: Path) -> list[PyprojectFile]:
    if not repo_dir.exists():
        raise RuntimeError(f"Repository path does not exist: {repo_dir}")
    pyproject_paths: list[Path] = sorted(repo_dir.rglob("pyproject.toml"))
    pyproject_files: list[PyprojectFile] = []
    for pyproject_path in pyproject_paths:
        relative_path: str = str(pyproject_path.relative_to(repo_dir)).replace("\\", "/")
        flavor: str = resolve_flavor_from_path(relative_path)
        content: str = pyproject_path.read_text(encoding="utf-8")
        pyproject_files.append(PyprojectFile(path=relative_path, flavor=flavor, content=content))
    return pyproject_files


def parse_toml_payload(text: str, source_path: str) -> dict[str, object]:
    try:
        parsed: object = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"Invalid TOML at {source_path}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected TOML structure at {source_path}")
    return parsed


def get_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value: object = payload.get(key)
    if isinstance(value, Mapping):
        return value
    return None


def get_string_list(payload: Mapping[str, object], key: str) -> list[str]:
    value: object = payload.get(key)
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized_item: str = item.strip()
            if normalized_item != "":
                result.append(normalized_item)
    return result


def normalize_package_name(name: str) -> str:
    lowered_name: str = name.strip().lower()
    collapsed_name: str = re.sub(r"[-_.]+", "-", lowered_name)
    return collapsed_name


def parse_package_name_from_spec(spec: str) -> str:
    spec_without_marker: str = spec.split(";")[0].strip()
    if spec_without_marker == "":
        return ""
    requirement_match: re.Match[str] | None = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", spec_without_marker)
    if requirement_match is None:
        return ""
    raw_name: str = requirement_match.group(1)
    return normalize_package_name(raw_name)


def extract_project_dependencies(pyproject_payload: Mapping[str, object], include_optional: bool) -> set[str]:
    project_section: Mapping[str, object] | None = get_mapping(pyproject_payload, "project")
    if project_section is None:
        return set()
    collected_specs: list[str] = []
    collected_specs.extend(get_string_list(project_section, "dependencies"))
    if include_optional:
        optional_section: Mapping[str, object] | None = get_mapping(project_section, "optional-dependencies")
        if optional_section is not None:
            for value in optional_section.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip() != "":
                            collected_specs.append(item.strip())
    package_names: set[str] = set()
    for spec in collected_specs:
        package_name: str = parse_package_name_from_spec(spec)
        if package_name != "":
            package_names.add(package_name)
    return package_names


def extract_poetry_dependencies(pyproject_payload: Mapping[str, object]) -> set[str]:
    tool_section: Mapping[str, object] | None = get_mapping(pyproject_payload, "tool")
    if tool_section is None:
        return set()
    poetry_section: Mapping[str, object] | None = get_mapping(tool_section, "poetry")
    if poetry_section is None:
        return set()
    dependencies_section: Mapping[str, object] | None = get_mapping(poetry_section, "dependencies")
    if dependencies_section is None:
        return set()
    package_names: set[str] = set()
    for key in dependencies_section.keys():
        if not isinstance(key, str):
            continue
        normalized_key: str = normalize_package_name(key)
        if normalized_key == "python":
            continue
        if normalized_key != "":
            package_names.add(normalized_key)
    return package_names


def extract_dependency_names(pyproject_payload: Mapping[str, object], include_optional: bool, include_poetry: bool) -> set[str]:
    package_names: set[str] = set()
    project_dependencies: set[str] = extract_project_dependencies(pyproject_payload, include_optional=include_optional)
    package_names.update(project_dependencies)
    if include_poetry:
        poetry_dependencies: set[str] = extract_poetry_dependencies(pyproject_payload)
        package_names.update(poetry_dependencies)
    return package_names


def collect_package_occurrences(config: AppConfig, pyproject_files: Sequence[PyprojectFile]) -> list[PackageOccurrence]:
    occurrences: list[PackageOccurrence] = []
    for pyproject_file in pyproject_files:
        payload: dict[str, object] = parse_toml_payload(pyproject_file.content, source_path=pyproject_file.path)
        package_names: set[str] = extract_dependency_names(
            payload,
            include_optional=config.include_optional,
            include_poetry=config.include_poetry,
        )
        for package_name in sorted(package_names):
            occurrences.append(
                PackageOccurrence(
                    package_name=package_name,
                    flavor=pyproject_file.flavor,
                    pyproject_path=pyproject_file.path,
                ),
            )
    return occurrences


def build_dependency_dataframe(occurrences: Sequence[PackageOccurrence]) -> pd.DataFrame:
    import pandas as pd
    flavors_by_package: dict[str, set[str]] = defaultdict(set)
    for occurrence in occurrences:
        flavors_by_package[occurrence.package_name].add(occurrence.flavor)
    rows: list[dict[str, str | int]] = []
    for package_name in sorted(flavors_by_package.keys()):
        sorted_flavors: list[str] = sorted(flavors_by_package[package_name])
        rows.append(
            {
                "package_name": package_name,
                "flavors": " | ".join(sorted_flavors),
                "_flavor_count": len(sorted_flavors),
            },
        )
    dataframe: pd.DataFrame = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    sorted_dataframe: pd.DataFrame = dataframe.sort_values(by=["_flavor_count", "package_name"], ascending=[False, True]).reset_index(drop=True)
    return sorted_dataframe.drop(columns=["_flavor_count"])


def save_dataframe(dataframe: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)


def execute() -> pd.DataFrame:
    ensure_pandas()
    config: AppConfig = parse_cli_args(sys.argv[1:])
    cleanup_root: Path | None = None
    repo_dir: Path
    if config.local_repo_path is not None:
        repo_dir = config.local_repo_path
    else:
        cleanup_root, repo_dir = download_repository_to_temp(config)
    pyproject_files: list[PyprojectFile] = collect_pyproject_files_from_directory(repo_dir)
    if len(pyproject_files) == 0:
        raise RuntimeError("No pyproject.toml files found in repository tree.")
    try:
        occurrences: list[PackageOccurrence] = collect_package_occurrences(config, pyproject_files)
        dataframe: pd.DataFrame = build_dependency_dataframe(occurrences)
        save_dataframe(dataframe, output_csv=config.output_csv)
        print(f"Pyproject files scanned: {len(pyproject_files)}")
        print(f"Unique packages found: {len(dataframe)}")
        print(f"CSV written to: {config.output_csv}")
        if not dataframe.empty:
            preview: pd.DataFrame = dataframe.head(20)
            print(preview.to_string(index=False))
        return dataframe
    finally:
        if cleanup_root is not None:
            shutil.rmtree(cleanup_root, ignore_errors=True)


if __name__ == "__main__":
    execute()
