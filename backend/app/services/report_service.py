from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.ai_explanation import AIExplanation
from app.models.analysis_job import AnalysisJob
from app.models.analysis_report import AnalysisReport
from app.models.enums import JobStatus, TaskType
from app.models.model_result import ModelResult
from app.models.user import User
from app.services.analysis_service import get_analysis_job


REPORT_TYPE_HTML = "html"
REPORT_STATUS_READY = "ready"
EXCLUDED_METRIC_KEYS = {"model_name"}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 score",
    "r2_score": "R² score",
    "mae": "MAE",
    "rmse": "RMSE",
    "mape": "MAPE",
    "test_size": "Test size",
    "target_mean": "Target mean",
    "target_min": "Target minimum",
    "target_max": "Target maximum",
}


def get_enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value

    return value


def get_cached_report(
    db: Session,
    analysis_id: int,
) -> AnalysisReport | None:
    return (
        db.query(AnalysisReport)
        .filter(AnalysisReport.analysis_id == analysis_id)
        .first()
    )


def require_completed_job(analysis_job: AnalysisJob) -> None:
    if get_enum_value(analysis_job.status) != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Report can only be generated for completed analysis jobs",
        )


def get_required_model_result(
    db: Session,
    analysis_job: AnalysisJob,
) -> ModelResult:
    model_result = (
        db.query(ModelResult)
        .filter(ModelResult.analysis_id == analysis_job.id)
        .first()
    )

    if model_result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model result not found for this analysis job",
        )

    return model_result


def get_optional_ai_explanation(
    db: Session,
    analysis_job: AnalysisJob,
) -> AIExplanation | None:
    return (
        db.query(AIExplanation)
        .filter(AIExplanation.analysis_id == analysis_job.id)
        .first()
    )


def get_report_or_404(
    db: Session,
    analysis_job: AnalysisJob,
) -> AnalysisReport:
    report = get_cached_report(db=db, analysis_id=analysis_job.id)

    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found for this analysis job",
        )

    return report


def get_safe_dataset_display_name(analysis_job: AnalysisJob) -> str:
    dataset_file_name = analysis_job.dataset_file_name

    if not dataset_file_name:
        return "Uploaded dataset"

    return dataset_file_name.split("/")[-1].split("\\")[-1]


def get_report_file_name(analysis_id: int) -> str:
    return f"basitanaliz_job_{analysis_id}_report.html"


def format_title(value: str) -> str:
    if value in METRIC_LABELS:
        return METRIC_LABELS[value]

    return value.replace("_", " ").title()


def render_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"

    if isinstance(value, (int, str, bool)):
        return str(value)

    if value is None:
        return "Not available"

    return str(value)


def render_metric_cards(values: dict[str, Any]) -> str:
    if not values:
        return "<p>Not available.</p>"

    cards = []
    for key, value in values.items():
        if key in EXCLUDED_METRIC_KEYS or isinstance(value, (dict, list)):
            continue
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{escape(format_title(str(key)))}</div>'
            f'<div class="metric-value">{escape(render_metric_value(value))}</div>'
            "</div>"
        )

    if not cards:
        return "<p>Not available.</p>"

    return '<div class="metric-grid">' + "".join(cards) + "</div>"


def render_text_list(values: list[Any]) -> str:
    if not values:
        return "<p>None.</p>"

    return '<ul class="clean-list">' + "".join(
        f"<li>{escape(str(value))}</li>" for value in values
    ) + "</ul>"


def render_setup_items(items: list[tuple[str, str]]) -> str:
    return "".join(
        '<div class="setup-row">'
        f'<span class="setup-label">{escape(label)}</span>'
        f'<span class="setup-value">{escape(value)}</span>'
        "</div>"
        for label, value in items
    )


def get_interpretation(model_result: ModelResult) -> dict[str, Any]:
    report_json = model_result.report_json

    if not isinstance(report_json, dict):
        return {}

    interpretation = report_json.get("interpretation")

    if isinstance(interpretation, dict):
        return interpretation

    return {}


def get_date_column(analysis_job: AnalysisJob, model_result: ModelResult) -> str | None:
    if get_enum_value(analysis_job.task_type) != TaskType.FORECASTING.value:
        return None

    report_json = model_result.report_json
    if isinstance(report_json, dict):
        date_column = report_json.get("date_column")
        if isinstance(date_column, str) and date_column.strip():
            return date_column

    config_json = analysis_job.config_json
    if isinstance(config_json, dict):
        date_column = config_json.get("date_column")
        if isinstance(date_column, str) and date_column.strip():
            return date_column

    return None


def render_report_html(
    analysis_job: AnalysisJob,
    model_result: ModelResult,
    ai_explanation: AIExplanation | None,
) -> str:
    interpretation = get_interpretation(model_result)
    date_column = get_date_column(analysis_job, model_result)
    generated_at = datetime.utcnow().isoformat(timespec="seconds")
    warnings = [
        warning.get("message", warning)
        if isinstance(warning, dict)
        else warning
        for warning in interpretation.get("warnings", [])
    ]
    recommended_actions = interpretation.get("recommended_actions", [])
    ai_explanation_html = (
        f"<p>{escape(ai_explanation.explanation_text)}</p>"
        if ai_explanation is not None
        else "<p>No local AI explanation was generated for this result.</p>"
    )
    setup_items = [
        ("Dataset", get_safe_dataset_display_name(analysis_job)),
        ("Dataset source", analysis_job.dataset_source_label),
        ("Task type", str(get_enum_value(analysis_job.task_type))),
        ("Target column", analysis_job.target_column),
    ]
    if date_column:
        setup_items.append(("Date column", date_column))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BasitAnaliz Report - Job {analysis_job.id}</title>
  <style>
    :root {{
      --background: #f6f0e8;
      --paper: #fffaf2;
      --ink: #1f2937;
      --muted: #6b7280;
      --border: #e7d8c6;
      --brand: #7c4a1e;
      --brand-soft: #f1dfc8;
      --card: #ffffff;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
      margin: 0;
      padding: 2rem;
      color: var(--ink);
      background: var(--background);
    }}
    .report-shell {{
      max-width: 980px;
      margin: 0 auto;
      background: var(--paper);
      border: 1px solid var(--border);
      border-radius: 20px;
      overflow: hidden;
      box-shadow: 0 18px 45px rgba(60, 40, 20, 0.10);
    }}
    .report-header {{
      padding: 2rem;
      background: linear-gradient(135deg, #7c4a1e, #b2783c);
      color: #fffaf2;
    }}
    .brand-kicker {{
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.78rem;
      font-weight: 700;
      opacity: 0.9;
      margin: 0 0 0.5rem;
    }}
    h1 {{
      margin: 0;
      font-size: 2rem;
      line-height: 1.15;
    }}
    .report-subtitle {{
      margin: 0.75rem 0 0;
      color: rgba(255, 250, 242, 0.86);
    }}
    main {{
      padding: 2rem;
    }}
    h2 {{
      color: #111827;
      font-size: 1.05rem;
      margin: 0 0 1rem;
      padding-bottom: 0.65rem;
      border-bottom: 1px solid var(--border);
    }}
    section {{
      margin-bottom: 1.25rem;
      padding: 1.25rem;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--border);
      border-radius: 16px;
    }}
    .muted {{
      color: var(--muted);
    }}
    .setup-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.75rem;
    }}
    .setup-row {{
      padding: 0.8rem;
      border-radius: 12px;
      background: #fff;
      border: 1px solid #f0e4d5;
    }}
    .setup-label {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.2rem;
    }}
    .setup-value {{
      font-weight: 700;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
    }}
    .metric-card {{
      background: var(--card);
      border: 1px solid #ecdcc9;
      border-radius: 14px;
      padding: 0.95rem;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric-value {{
      margin-top: 0.25rem;
      font-size: 1.25rem;
      font-weight: 800;
      color: var(--brand);
    }}
    .clean-list {{
      margin: 0;
      padding-left: 1.2rem;
    }}
    .clean-list li {{
      margin-bottom: 0.35rem;
    }}
    @media print {{
      body {{
        background: #fff;
        padding: 0;
      }}
      .report-shell {{
        box-shadow: none;
        border-radius: 0;
      }}
      section {{
        break-inside: avoid;
      }}
    }}
  </style>
</head>
<body>
  <div class="report-shell">
    <header class="report-header">
      <p class="brand-kicker">BasitAnaliz</p>
      <h1>Local Analysis Report</h1>
      <p class="report-subtitle">Generated at {escape(generated_at)} UTC · Job {analysis_job.id}</p>
    </header>
    <main>

  <section>
    <h2>Analysis Setup</h2>
    <div class="setup-grid">
      {render_setup_items(setup_items)}
    </div>
  </section>

  <section>
    <h2>Model</h2>
    <div class="setup-grid">
      {render_setup_items([
          ("Model name", model_result.model_name),
          ("Quality level", str(interpretation.get("quality_level", "Not available"))),
      ])}
    </div>
  </section>

  <section>
    <h2>Summary</h2>
    <p>{escape(str(interpretation.get("summary", "No summary available.")))}</p>
  </section>

  <section>
    <h2>Key Metrics</h2>
    {render_metric_cards(model_result.metrics or {})}
  </section>

  <section>
    <h2>Warnings</h2>
    {render_text_list(warnings)}
  </section>

  <section>
    <h2>Recommended Actions</h2>
    {render_text_list(recommended_actions)}
  </section>

  <section>
    <h2>AI Explanation</h2>
    {ai_explanation_html}
  </section>
    </main>
  </div>
</body>
</html>
"""


def create_analysis_report(
    db: Session,
    job_id: int,
    current_user: User,
) -> AnalysisReport:
    analysis_job = get_analysis_job(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )
    require_completed_job(analysis_job)

    model_result = get_required_model_result(db=db, analysis_job=analysis_job)
    cached_report = get_cached_report(db=db, analysis_id=analysis_job.id)

    if cached_report is not None:
        return cached_report

    ai_explanation = get_optional_ai_explanation(db=db, analysis_job=analysis_job)
    report = AnalysisReport(
        analysis_id=analysis_job.id,
        report_type=REPORT_TYPE_HTML,
        status=REPORT_STATUS_READY,
        file_name=get_report_file_name(analysis_job.id),
        html_content=render_report_html(
            analysis_job=analysis_job,
            model_result=model_result,
            ai_explanation=ai_explanation,
        ),
    )

    try:
        db.add(report)
        db.commit()
        db.refresh(report)
    except Exception:
        db.rollback()
        raise

    return report


def get_analysis_report(
    db: Session,
    job_id: int,
    current_user: User,
) -> AnalysisReport:
    analysis_job = get_analysis_job(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )

    return get_report_or_404(db=db, analysis_job=analysis_job)


def get_analysis_report_html(
    db: Session,
    job_id: int,
    current_user: User,
) -> AnalysisReport:
    return get_analysis_report(
        db=db,
        job_id=job_id,
        current_user=current_user,
    )
