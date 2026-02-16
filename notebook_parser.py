from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NotebookParseRequest:
    notebook_path: Path


@dataclass(frozen=True)
class NotebookParseResult:
    notebook_path: Path
    imports: set[str]
    extensions: set[str]


_LOAD_EXT_PATTERN = re.compile(r"^\s*%{1,2}load_ext\s+([^\s]+)")
_RELOAD_EXT_PATTERN = re.compile(r"^\s*%{1,2}reload_ext\s+([^\s]+)")
_JUPYTER_EXT_PATTERN = re.compile(r"^\s*!\s*jupyter\s+(?:nbextension|labextension)\s+\w+\s+([^\s]+)")


def load_notebook_json(request: NotebookParseRequest) -> dict[str, object]:
    try:
        notebook_text: str = request.notebook_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read notebook: {request.notebook_path}") from exc
    try:
        notebook_data: object = json.loads(notebook_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid notebook JSON: {request.notebook_path}") from exc
    if not isinstance(notebook_data, dict):
        raise RuntimeError(f"Unexpected notebook JSON structure: {request.notebook_path}")
    return notebook_data


def extract_code_cells(notebook_data: dict[str, object]) -> list[str]:
    cells_value: object = notebook_data.get("cells", [])
    if not isinstance(cells_value, list):
        return []
    cell_sources: list[str] = []
    for cell in cells_value:
        if not isinstance(cell, dict):
            continue
        cell_type: object = cell.get("cell_type")
        if cell_type != "code":
            continue
        source_value: object = cell.get("source", [])
        source_text: str = normalize_cell_source(source_value)
        if source_text != "":
            cell_sources.append(source_text)
    return cell_sources


def normalize_cell_source(source: object) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        source_items: list[str] = [str(item) for item in source]
        return "".join(source_items)
    return ""


def sanitize_code(code: str) -> str:
    sanitized_lines: list[str] = []
    for line in code.splitlines():
        stripped_line: str = line.lstrip()
        if stripped_line.startswith("%") or stripped_line.startswith("!"):
            continue
        sanitized_lines.append(line)
    return "\n".join(sanitized_lines)


def extract_imports_from_code(code: str) -> set[str]:
    sanitized_code: str = sanitize_code(code)
    if sanitized_code.strip() == "":
        return set()
    try:
        parsed_tree: ast.AST = ast.parse(sanitized_code)
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(parsed_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "":
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module_name: str | None = node.module
            if module_name is not None and module_name != "":
                imports.add(module_name)
    return imports


def extract_extensions_from_code(code: str) -> set[str]:
    extensions: set[str] = set()
    for line in code.splitlines():
        load_match: re.Match[str] | None = _LOAD_EXT_PATTERN.match(line)
        if load_match is not None:
            extensions.add(load_match.group(1).lower())
            continue
        reload_match: re.Match[str] | None = _RELOAD_EXT_PATTERN.match(line)
        if reload_match is not None:
            extensions.add(reload_match.group(1).lower())
            continue
        jupyter_match: re.Match[str] | None = _JUPYTER_EXT_PATTERN.match(line)
        if jupyter_match is not None:
            extensions.add(jupyter_match.group(1).lower())
    return extensions


def parse_notebook(request: NotebookParseRequest) -> NotebookParseResult:
    notebook_data: dict[str, object] = load_notebook_json(request)
    code_cells: list[str] = extract_code_cells(notebook_data)
    notebook_imports: set[str] = set()
    notebook_extensions: set[str] = set()
    for cell_source in code_cells:
        cell_imports: set[str] = extract_imports_from_code(cell_source)
        notebook_imports.update(cell_imports)
        cell_extensions: set[str] = extract_extensions_from_code(cell_source)
        notebook_extensions.update(cell_extensions)
    return NotebookParseResult(notebook_path=request.notebook_path, imports=notebook_imports, extensions=notebook_extensions)
