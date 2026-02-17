from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from metrics import FeatureUsage, UsageReport


@dataclass(frozen=True)
class PlotReportRequest:
    report: UsageReport
    title: str
    max_items: int
    x_label: str


def ensure_plotly() -> None:
    try:
        import plotly.express as px
    except ImportError as exc:
        raise RuntimeError('Plotly Express dependencies are missing. Run `pip install "plotly[express]"`.') from exc
    _ = px


def build_chart_data(usages: Sequence[FeatureUsage], max_items: int) -> tuple[list[str], list[int], list[float]]:
    selected_usages: list[FeatureUsage] = list(usages)[:max_items]
    labels: list[str] = [usage.name for usage in selected_usages]
    counts: list[int] = [usage.notebook_count for usage in selected_usages]
    percents: list[float] = [usage.usage_percent for usage in selected_usages]
    return labels, counts, percents


def render_usage_report(request: PlotReportRequest) -> None:
    if len(request.report.usage) == 0:
        print(f"No data available for {request.title}.")
        return
    ensure_plotly()
    import plotly.express as px
    labels, counts, percents = build_chart_data(request.report.usage, request.max_items)
    fig = px.bar(x=labels, y=counts, text=percents, title=request.title, labels={"x": request.x_label, "y": "Notebook count", "text": "Usage %"})
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Notebook count", xaxis_title="Library")
    fig.show()
