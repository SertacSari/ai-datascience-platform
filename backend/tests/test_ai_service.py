from datetime import datetime
from typing import Any

import pytest
from fastapi import HTTPException

from app.models.ai_explanation import AIExplanation
from app.models.analysis_job import AnalysisJob
from app.models.dataset import Dataset
from app.models.enums import JobStatus, TaskType
from app.models.model_result import ModelResult
from app.models.user import User
from app.services import ai_service


class OllamaCallRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt: str) -> dict[str, str]:
        self.calls.append({"prompt": prompt})
        return {"response": "This is a simple explanation."}


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


def add_dataset(db_session, user: User) -> Dataset:
    dataset = Dataset(
        user_id=user.id,
        file_name="../private/customer_upload.csv",
        file_path="/server/uploads/private/customer_upload.csv",
        cleaned_file_path="/server/uploads/private/customer_upload_cleaned.csv",
        row_count=100,
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
    target_column: str = "target",
    config_json: dict[str, Any] | None = None,
) -> AnalysisJob:
    dataset = add_dataset(db_session, user)
    job = AnalysisJob(
        user_id=user.id,
        dataset_id=dataset.id,
        task_type=task_type,
        target_column=target_column,
        status=JobStatus.COMPLETED,
        config_json=config_json or {},
        finished_at=datetime.utcnow(),
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    return job


def add_model_result(
    db_session,
    job: AnalysisJob,
    metrics: dict[str, Any] | None = None,
    report_json: dict[str, Any] | None = None,
) -> ModelResult:
    model_result = ModelResult(
        analysis_id=job.id,
        model_name="TestModel",
        metrics=metrics or {
            "accuracy": 0.82,
            "precision": 0.81,
            "recall": 0.8,
            "f1_score": 0.8,
        },
        report_json=report_json
        or {
            "interpretation": {
                "summary": "The model shows fair performance.",
                "quality_level": "fair",
                "warnings": [
                    {
                        "code": "low_recall",
                        "message": "The model may miss some real cases.",
                    }
                ],
                "metric_explanations": {
                    "accuracy": "Overall share of correct predictions.",
                },
                "recommended_actions": [
                    "Check false negatives; the model may miss real cases.",
                ],
            }
        },
    )
    db_session.add(model_result)
    db_session.commit()
    db_session.refresh(model_result)
    return model_result


def enable_ai(monkeypatch) -> OllamaCallRecorder:
    recorder = OllamaCallRecorder()
    monkeypatch.setattr(ai_service, "AI_EXPLANATION_ENABLED", True)
    monkeypatch.setattr(ai_service, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(ai_service, "OLLAMA_MODEL", "gemma3:4b")
    monkeypatch.setattr(ai_service, "AI_EXPLANATION_TIMEOUT_SECONDS", 30)
    monkeypatch.setattr(ai_service, "post_ollama_request", recorder)
    return recorder


@pytest.mark.parametrize(
    "task_type,target_column,config_json",
    [
        (TaskType.CLASSIFICATION, "target", {}),
        (TaskType.REGRESSION, "price", {}),
        (TaskType.FORECASTING, "sales", {"date_column": "date"}),
    ],
)
def test_ai_explanation_generation_for_completed_jobs(
    db_session,
    monkeypatch,
    task_type: TaskType,
    target_column: str,
    config_json: dict[str, Any],
) -> None:
    recorder = enable_ai(monkeypatch)
    user = add_user(db_session, f"{task_type.value}_ai_user")
    job = add_completed_job(
        db_session=db_session,
        user=user,
        task_type=task_type,
        target_column=target_column,
        config_json=config_json,
    )
    add_model_result(db_session, job)

    explanation = ai_service.create_ai_explanation(db_session, job.id, user)

    assert explanation.analysis_id == job.id
    assert explanation.llm_model == "gemma3:4b"
    assert explanation.explanation_text == "This is a simple explanation."
    assert len(recorder.calls) == 1


def test_cached_ai_explanation_does_not_call_ollama_again(
    db_session,
    monkeypatch,
) -> None:
    recorder = enable_ai(monkeypatch)
    user = add_user(db_session, "cached_ai_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION)
    add_model_result(db_session, job)
    cached_explanation = AIExplanation(
        analysis_id=job.id,
        llm_model="gemma3:4b",
        explanation_text="Cached explanation.",
    )
    db_session.add(cached_explanation)
    db_session.commit()
    db_session.refresh(cached_explanation)

    explanation = ai_service.create_ai_explanation(db_session, job.id, user)

    assert explanation.id == cached_explanation.id
    assert explanation.explanation_text == "Cached explanation."
    assert recorder.calls == []


def test_get_ai_explanation_returns_cached_explanation(db_session) -> None:
    user = add_user(db_session, "get_ai_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION)
    explanation = AIExplanation(
        analysis_id=job.id,
        llm_model="gemma3:4b",
        explanation_text="Cached explanation.",
    )
    db_session.add(explanation)
    db_session.commit()
    db_session.refresh(explanation)

    result = ai_service.get_ai_explanation(db_session, job.id, user)

    assert result.id == explanation.id


def test_get_ai_explanation_returns_404_when_none_exists(db_session) -> None:
    user = add_user(db_session, "missing_ai_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION)

    with pytest.raises(HTTPException) as error:
        ai_service.get_ai_explanation(db_session, job.id, user)

    assert error.value.status_code == 404
    assert "AI explanation not found" in error.value.detail


def test_non_owner_cannot_create_ai_explanation(
    db_session,
    monkeypatch,
) -> None:
    recorder = enable_ai(monkeypatch)
    owner = add_user(db_session, "ai_owner")
    other_user = add_user(db_session, "ai_other")
    job = add_completed_job(db_session, owner, TaskType.CLASSIFICATION)
    add_model_result(db_session, job)

    with pytest.raises(HTTPException) as error:
        ai_service.create_ai_explanation(db_session, job.id, other_user)

    assert error.value.status_code == 404
    assert recorder.calls == []


def test_non_completed_job_cannot_create_ai_explanation(
    db_session,
    monkeypatch,
) -> None:
    recorder = enable_ai(monkeypatch)
    user = add_user(db_session, "created_ai_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION)
    job.status = JobStatus.CREATED
    db_session.commit()
    add_model_result(db_session, job)

    with pytest.raises(HTTPException) as error:
        ai_service.create_ai_explanation(db_session, job.id, user)

    assert error.value.status_code == 400
    assert "completed analysis jobs" in error.value.detail
    assert recorder.calls == []


def test_missing_model_result_blocks_ai_explanation(
    db_session,
    monkeypatch,
) -> None:
    recorder = enable_ai(monkeypatch)
    user = add_user(db_session, "missing_result_ai_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION)

    with pytest.raises(HTTPException) as error:
        ai_service.create_ai_explanation(db_session, job.id, user)

    assert error.value.status_code == 404
    assert "Model result not found" in error.value.detail
    assert recorder.calls == []


def test_ollama_unavailable_returns_clean_503(
    db_session,
    monkeypatch,
) -> None:
    user = add_user(db_session, "ollama_unavailable_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION)
    add_model_result(db_session, job)
    monkeypatch.setattr(ai_service, "AI_EXPLANATION_ENABLED", True)

    def raise_provider_error(prompt: str) -> dict[str, str]:
        raise HTTPException(
            status_code=503,
            detail="AI explanation provider is unavailable",
        )

    monkeypatch.setattr(ai_service, "post_ollama_request", raise_provider_error)

    with pytest.raises(HTTPException) as error:
        ai_service.create_ai_explanation(db_session, job.id, user)

    assert error.value.status_code == 503
    assert "provider is unavailable" in error.value.detail


def test_disabled_ai_explanation_returns_clean_503(
    db_session,
    monkeypatch,
) -> None:
    user = add_user(db_session, "disabled_ai_user")
    job = add_completed_job(db_session, user, TaskType.CLASSIFICATION)
    add_model_result(db_session, job)
    monkeypatch.setattr(ai_service, "AI_EXPLANATION_ENABLED", False)

    with pytest.raises(HTTPException) as error:
        ai_service.create_ai_explanation(db_session, job.id, user)

    assert error.value.status_code == 503
    assert "disabled" in error.value.detail


def test_prompt_payload_excludes_unsafe_data(
    db_session,
    monkeypatch,
) -> None:
    recorder = enable_ai(monkeypatch)
    user = add_user(db_session, "safe_prompt_user")
    job = add_completed_job(
        db_session,
        user,
        TaskType.FORECASTING,
        target_column="sales",
        config_json={
            "date_column": "date",
            "unsafe_path": "/server/uploads/private/secret.csv",
        },
    )
    add_model_result(
        db_session,
        job,
        metrics={"mae": 5.0, "rmse": 8.0, "r2_score": 0.72},
        report_json={
            "date_column": "date",
            "prediction_sample": [
                {
                    "date": "2026-01-01",
                    "actual": 10,
                    "predicted": 11,
                    "raw_row": "secret,csv,content",
                }
            ],
            "file_path": "/server/uploads/private/customer_upload.csv",
            "cleaned_file_path": "/server/uploads/private/customer_upload_cleaned.csv",
            "interpretation": {
                "summary": "The forecast is fair.",
                "quality_level": "fair",
                "warnings": [],
                "metric_explanations": {"mae": "Average absolute error."},
                "recommended_actions": [],
            },
        },
    )

    ai_service.create_ai_explanation(db_session, job.id, user)

    prompt = recorder.calls[0]["prompt"]
    assert "Do not compare this model to other models" in prompt
    assert "Do not use classification-only terms like accuracy" in prompt
    assert "sales" in prompt
    assert "date_column" in prompt
    assert "customer_upload.csv" in prompt
    assert "/server/uploads" not in prompt
    assert "customer_upload_cleaned.csv" not in prompt
    assert "secret,csv,content" not in prompt
    assert "prediction_sample" not in prompt
    assert "unsafe_path" not in prompt
    assert "password" not in prompt
    assert "token" not in prompt
