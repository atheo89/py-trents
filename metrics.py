from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence


@dataclass(frozen=True)
class NotebookFeatureUsage:
    notebook_path: Path
    features: set[str]


@dataclass(frozen=True)
class UsageMetricsRequest:
    notebook_usages: Sequence[NotebookFeatureUsage]


@dataclass(frozen=True)
class FeatureUsage:
    name: str
    notebook_count: int
    usage_percent: float


@dataclass(frozen=True)
class UsageReport:
    total_notebooks: int
    usage: list[FeatureUsage]


def compute_usage_metrics(request: UsageMetricsRequest) -> UsageReport:
    total_notebooks: int = len(request.notebook_usages)
    feature_counts: dict[str, int] = {}
    for notebook_usage in request.notebook_usages:
        for feature in notebook_usage.features:
            current_count: int = feature_counts.get(feature, 0)
            feature_counts[feature] = current_count + 1
    usage_list: list[FeatureUsage] = []
    for feature, count in feature_counts.items():
        usage_percent: float = round((count / total_notebooks) * 100, 2) if total_notebooks > 0 else 0.0
        usage_list.append(FeatureUsage(name=feature, notebook_count=count, usage_percent=usage_percent))
    usage_list.sort(key=lambda item: (-item.notebook_count, item.name))
    return UsageReport(total_notebooks=total_notebooks, usage=usage_list)
