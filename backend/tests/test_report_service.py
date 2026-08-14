from html import unescape

import pytest
from fastapi import HTTPException

from app.models.ai_explanation import AIExplanation
from app.models.analysis_job import AnalysisJob
from app.models.analysis_report import AnalysisReport
from app.models.dataset import Dataset
from app.models.enums import JobStatus, TaskType
from app.models.model_result import ModelResult
from app.models.user import User
from app.routers.analysis import download_analysis_report_endpoint
from app.services.report_service import (
    create_analysis_report,
    get_analysis_report,
)


def add_user(db_session, username: str) -> User:
    user = User(
        username=username,
        email=f"{username}@example.com",
        password_hash="not-used-in-tests",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def add_dataset(
    db_session,
    user: User,
    file_name: str = "dataset.csv",
    cleaned: bool = False,
) -> Dataset:
    dataset = Dataset(
        user_id=user.id,
        file_name=file_name,
        file_path="/server/private/uploads/dataset.csv",
        cleaned_file_path="/server/private/uploads/cleaned_dataset.csv" if cleaned else None,
        row_count=40,
        column_count=4,
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


def add_completed_job(
    db_session,
    user: User,
    task_type: TaskType,
    target_column: str,
    config_json: dict | None = None,
    file_name: str = "dataset.csv",
    cleaned: bool = False,
) -> AnalysisJob:
    dataset = add_dataset(
        db_session=db_session,
        user=user,
        file_name=file_name,
        cleaned=cleaned,
    )
    job = AnalysisJob(
        user_id=user.id,
        dataset_id=dataset.id,
        task_type=task_type,
        target_column=target_column,
        status=JobStatus.COMPLETED,
        config_json=config_json or {},
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    return job


def add_model_result(
    db_session,
    job: AnalysisJob,
    model_name: str = "RandomForestClassifier",
    metrics: dict | None = None,
    report_json: dict | None = None,
) -> ModelResult:
    model_result = ModelResult(
        analysis_id=job.id,
        model_name=model_name,
        metrics=metrics
        or {
            "accuracy": 0.91,
            "precision": 0.9,
            "recall": 0.89,
            "f1_score": 0.9,
            "model_name": model_name,
        },
        report_json=report_json
        or {
            "interpretation": {
                "summary": "The model shows good classification performance.",
                "quality_level": "good",
                "warnings": [
                    {
                        "code": "small_dataset",
                        "message": "The dataset is small.",
                    }
                ],
                "recommended_actions": [
                    "Add more rows before trusting the result.",
                ],
            }
        },
    )
    db_session.add(model_result)
    db_session.commit()
    db_session.refresh(model_result)
    return model_result


def add_ai_explanation(db_session, job: AnalysisJob, text: str) -> AIExplanation:
    explanation = AIExplanation(
        analysis_id=job.id,
        llm_model="gemma3:4b",
        explanation_text=text,
    )
    db_session.add(explanation)
    db_session.commit()
    db_session.refresh(explanation)
    return explanation


def get_section_html(html: str, heading: str) -> str:
    section_start = html.index(f"<h2>{heading}</h2>")
    next_section_start = html.find("<section>", section_start + 1)

    if next_section_start == -1:
        return html[section_start:]

    return html[section_start:next_section_start]


@pytest.mark.parametrize(
    "task_type,target_column,model_name,metrics,report_json,config_json",
    [
        (
            TaskType.CLASSIFICATION,
            "target",
            "RandomForestClassifier",
            {"accuracy": 0.91, "precision": 0.9, "recall": 0.89, "f1_score": 0.9},
            None,
            {},
        ),
        (
            TaskType.REGRESSION,
            "price",
            "RandomForestRegressor",
            {"mae": 5.2, "rmse": 7.8, "r2_score": 0.82},
            {
                "interpretation": {
                    "summary": "The model shows good regression performance.",
                    "quality_level": "good",
                    "warnings": [],
                    "recommended_actions": ["Review errors before decisions."],
                }
            },
            {},
        ),
        (
            TaskType.FORECASTING,
            "sales",
            "RandomForestRegressorForecasting",
            {"mae": 12.5, "rmse": 16.1, "mape": 7.4, "r2_score": 0.64},
            {
                "date_column": "date",
                "interpretation": {
                    "summary": "The model shows fair forecasting performance.",
                    "quality_level": "fair",
                    "warnings": [],
                    "recommended_actions": ["Add more historical rows."],
                },
            },
            {"date_column": "date"},
        ),
    ],
)
def test_completed_jobs_can_generate_reports(
    db_session,
    task_type: TaskType,
    target_column: str,
    model_name: str,
    metrics: dict,
    report_json: dict | None,
    config_json: dict,
) -> None:
    user = add_user(db_session, f"report_{task_type.value}")
    job = add_completed_job(
        db_session=db_session,
        user=user,
        task_type=task_type,
        target_column=target_column,
        config_json=config_json,
    )
    add_model_result(
        db_session=db_session,
        job=job,
        model_name=model_name,
        metrics=metrics,
        report_json=report_json,
    )

    report = create_analysis_report(db_session, job.id, user)

    assert report.analysis_id == job.id
    assert report.report_type == "html"
    assert report.status == "ready"
    assert report.file_name == f"basitanaliz_job_{job.id}_report.html"
    assert "BasitAnaliz" in report.html_content
    assert "Local Analysis Report" in report.html_content
    assert model_name in report.html_content
    assert target_column in report.html_content
    if task_type == TaskType.FORECASTING:
        assert "Date column" in report.html_content
        assert "date" in report.html_content


def test_cached_report_is_returned_without_regenerating(db_session) -> None:
    user = add_user(db_session, "cached_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")
    add_model_result(db_session, job)
    first_report = create_analysis_report(db_session, job.id, user)
    first_html = first_report.html_content

    second_report = create_analysis_report(db_session, job.id, user)

    assert second_report.id == first_report.id
    assert second_report.html_content == first_html
    assert (
        db_session.query(AnalysisReport)
        .filter(AnalysisReport.analysis_id == job.id)
        .count()
        == 1
    )


def test_get_missing_report_returns_404(db_session) -> None:
    user = add_user(db_session, "missing_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")

    with pytest.raises(HTTPException) as error:
        get_analysis_report(db_session, job.id, user)

    assert error.value.status_code == 404
    assert "Report not found" in error.value.detail


def test_non_owner_cannot_generate_get_or_download_report(db_session) -> None:
    owner = add_user(db_session, "report_owner")
    other_user = add_user(db_session, "report_other")
    job = add_completed_job(db_session, owner, TaskType.CLASSIFICATION, "target")
    add_model_result(db_session, job)

    with pytest.raises(HTTPException) as create_error:
        create_analysis_report(db_session, job.id, other_user)
    with pytest.raises(HTTPException) as get_error:
        get_analysis_report(db_session, job.id, other_user)

    assert create_error.value.status_code == 404
    assert get_error.value.status_code == 404


def test_non_completed_job_cannot_generate_report(db_session) -> None:
    user = add_user(db_session, "created_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")
    job.status = JobStatus.CREATED
    db_session.commit()
    add_model_result(db_session, job)

    with pytest.raises(HTTPException) as error:
        create_analysis_report(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "completed analysis jobs" in error.value.detail


def test_missing_model_result_blocks_report_generation(db_session) -> None:
    user = add_user(db_session, "missing_result_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")

    with pytest.raises(HTTPException) as error:
        create_analysis_report(db_session, job.id, user)

    assert error.value.status_code == 404
    assert "Model result not found" in error.value.detail


def test_generated_html_escapes_unsafe_values(db_session) -> None:
    user = add_user(db_session, "safe_report_user")
    job = add_completed_job(
        db_session=db_session,
        user=user,
        task_type=TaskType.CLASSIFICATION,
        target_column="<script>alert('target')</script>",
        file_name="<img src=x onerror=alert(1)>.csv",
    )
    add_model_result(
        db_session=db_session,
        job=job,
        report_json={
            "interpretation": {
                "summary": "<script>alert('summary')</script>",
                "quality_level": "good",
                "warnings": [
                    {"message": "<b>unsafe warning</b>"},
                ],
                "recommended_actions": [
                    "<i>unsafe action</i>",
                ],
            }
        },
    )
    add_ai_explanation(
        db_session,
        job,
        "<script>alert('ai')</script>",
    )

    report = create_analysis_report(db_session, job.id, user)
    html = report.html_content

    assert "<script>" not in html
    assert "<img src=x" not in html
    assert "<b>unsafe warning</b>" not in html
    assert "<i>unsafe action</i>" not in html
    assert "&lt;script&gt;alert" in html
    assert "/server/private/uploads" not in html


def test_report_includes_metrics_warnings_actions_and_ai_explanation(db_session) -> None:
    user = add_user(db_session, "content_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")
    add_model_result(db_session, job)
    add_ai_explanation(db_session, job, "The model looks good in this test.")

    report = create_analysis_report(db_session, job.id, user)
    html = unescape(report.html_content)

    assert "Accuracy" in html
    assert "0.91" in html
    assert "The dataset is small." in html
    assert "Add more rows before trusting the result." in html
    assert "The model looks good in this test." in html


def test_report_metrics_use_cards_and_exclude_metadata(db_session) -> None:
    user = add_user(db_session, "metric_cards_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")
    add_model_result(db_session, job)

    report = create_analysis_report(db_session, job.id, user)
    metrics_section = get_section_html(report.html_content, "Key Metrics")

    assert 'class="metric-grid"' in metrics_section
    assert 'class="metric-card"' in metrics_section
    assert "F1 score" in metrics_section
    assert "Model Name" not in metrics_section
    assert "RandomForestClassifier" not in metrics_section


def test_report_uses_branded_layout(db_session) -> None:
    user = add_user(db_session, "branded_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")
    add_model_result(db_session, job)

    report = create_analysis_report(db_session, job.id, user)
    html = report.html_content

    assert "report-shell" in html
    assert "report-header" in html
    assert "metric-grid" in html
    assert "BasitAnaliz" in html


def test_classification_report_setup_has_no_empty_date_column(db_session) -> None:
    user = add_user(db_session, "no_blank_setup_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")
    add_model_result(db_session, job)

    report = create_analysis_report(db_session, job.id, user)
    setup_section = get_section_html(report.html_content, "Analysis Setup")

    assert "Date column" not in setup_section


def test_report_generates_without_ai_explanation(db_session) -> None:
    user = add_user(db_session, "no_ai_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")
    add_model_result(db_session, job)

    report = create_analysis_report(db_session, job.id, user)

    assert "No local AI explanation was generated for this result." in (
        report.html_content
    )


def test_download_endpoint_returns_text_html(db_session) -> None:
    user = add_user(db_session, "download_report_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION, "target")
    add_model_result(db_session, job)
    report = create_analysis_report(db_session, job.id, user)

    response = download_analysis_report_endpoint(
        job_id=job.id,
        db=db_session,
        current_user=user,
    )

    assert response.media_type == "text/html"
    assert report.file_name in response.headers["content-disposition"]
    assert "Local Analysis Report" in response.body.decode()
