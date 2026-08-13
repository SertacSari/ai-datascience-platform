from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.user import User
from app.schemas.dataset import (
    AnalysisRecommendationAlternative,
    AnalysisRecommendationResponse,
)
from app.services.dataset_service import get_owned_dataset, read_stored_dataset_file


TaskName = Literal["classification", "regression", "forecasting"]

MIN_FORECASTING_ROWS = 30
MAX_CLASSIFICATION_CLASSES = 20
MISSING_HEAVY_RATIO = 0.40
ID_LIKE_UNIQUE_RATIO = 0.90

ID_LIKE_NAMES = {
    "id",
    "user_id",
    "customer_id",
    "uuid",
    "index",
    "code",
    "zip",
    "postal",
    "postal_code",
}
DATE_NAME_HINTS = {"date", "datetime", "timestamp", "time", "day"}
FORECASTING_TARGET_HINTS = {
    "sales",
    "sale",
    "demand",
    "revenue",
    "amount",
    "value",
    "total",
    "volume",
    "orders",
    "quantity",
}
REGRESSION_TARGET_HINTS = {
    "sale_price",
    "price",
    "revenue",
    "sales",
    "amount",
    "cost",
    "value",
    "score",
    "total",
}
CLASSIFICATION_TARGET_HINTS = {
    "churned",
    "churn",
    "default",
    "approved",
    "fraud",
    "status",
    "label",
    "class",
    "outcome",
    "target",
}


@dataclass
class ColumnProfile:
    name: str
    normalized_name: str
    is_numeric: bool
    is_bool: bool
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    is_id_like: bool
    is_date_like: bool
    date_valid_ratio: float

    @property
    def is_constant(self) -> bool:
        return self.unique_count <= 1

    @property
    def is_missing_heavy(self) -> bool:
        return self.missing_ratio >= MISSING_HEAVY_RATIO

    @property
    def is_categorical_like(self) -> bool:
        return (
            self.is_bool
            or not self.is_numeric
            or self.unique_count <= MAX_CLASSIFICATION_CLASSES
        )


@dataclass
class Candidate:
    task_type: TaskName
    target_column: str
    score: int
    reason: str
    date_column: str | None = None
    date_score: int = 0


def normalize_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_").replace("-", "_")


def has_name_hint(normalized_name: str, hints: set[str]) -> bool:
    tokens = set(normalized_name.split("_"))
    return normalized_name in hints or bool(tokens & hints)


def is_id_like_name(normalized_name: str) -> bool:
    if normalized_name in ID_LIKE_NAMES:
        return True

    return (
        normalized_name.endswith("_id")
        or normalized_name.endswith("_uuid")
        or normalized_name.endswith("_code")
    )


def get_date_valid_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0

    non_missing = series.dropna()
    if non_missing.empty:
        return 0.0

    parsed_dates = pd.to_datetime(
        non_missing,
        errors="coerce",
        format="mixed",
    )
    return float(parsed_dates.notna().mean())


def build_column_profiles(df: pd.DataFrame) -> dict[str, ColumnProfile]:
    profiles = {}
    row_count = len(df)

    for column in df.columns:
        column_name = str(column)
        normalized_name = normalize_name(column_name)
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / row_count if row_count else 0.0
        missing_ratio = float(series.isna().mean()) if row_count else 1.0
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        date_valid_ratio = 1.0 if is_datetime else get_date_valid_ratio(series)
        is_date_hinted = has_name_hint(normalized_name, DATE_NAME_HINTS)
        is_date_like = is_datetime or (
            date_valid_ratio >= 0.90
            and (is_date_hinted or not is_numeric)
        )
        is_id_like = is_id_like_name(normalized_name) or (
            unique_ratio >= ID_LIKE_UNIQUE_RATIO
            and (
                is_id_like_name(normalized_name)
                or "id" in normalized_name.split("_")
            )
        )

        profiles[column_name] = ColumnProfile(
            name=column_name,
            normalized_name=normalized_name,
            is_numeric=is_numeric,
            is_bool=pd.api.types.is_bool_dtype(series),
            missing_ratio=missing_ratio,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            is_id_like=is_id_like,
            is_date_like=is_date_like,
            date_valid_ratio=date_valid_ratio,
        )

    return profiles


def is_usable_target(profile: ColumnProfile) -> bool:
    return (
        not profile.is_id_like
        and not profile.is_constant
        and not profile.is_missing_heavy
    )


def build_date_candidates(profiles: dict[str, ColumnProfile]) -> list[Candidate]:
    candidates = []

    for profile in profiles.values():
        if profile.is_id_like or profile.is_missing_heavy:
            continue

        if profile.is_date_like and profile.date_valid_ratio >= 0.90:
            score = 40
            if has_name_hint(profile.normalized_name, DATE_NAME_HINTS):
                score += 30
            if profile.unique_ratio >= 0.90:
                score += 10
            candidates.append(
                Candidate(
                    task_type="forecasting",
                    target_column=profile.name,
                    score=score,
                    reason=f"{profile.name} looks like a usable date column.",
                )
            )

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def build_classification_candidates(
    profiles: dict[str, ColumnProfile],
    row_count: int,
) -> list[Candidate]:
    candidates = []

    for profile in profiles.values():
        if not is_usable_target(profile) or not profile.is_categorical_like:
            continue

        if profile.unique_count < 2 or profile.unique_count > MAX_CLASSIFICATION_CLASSES:
            continue

        score = 30
        if has_name_hint(profile.normalized_name, CLASSIFICATION_TARGET_HINTS):
            score += 55
        if profile.unique_count == 2:
            score += 10
        if profile.missing_ratio:
            score -= 15
        if row_count >= 50:
            score += 5

        candidates.append(
            Candidate(
                task_type="classification",
                target_column=profile.name,
                score=score,
                reason=(
                    f"{profile.name} has {profile.unique_count} classes and can be "
                    "used for classification."
                ),
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def build_regression_candidates(
    profiles: dict[str, ColumnProfile],
    row_count: int,
) -> list[Candidate]:
    candidates = []

    for profile in profiles.values():
        if not is_usable_target(profile) or not profile.is_numeric:
            continue

        if profile.unique_count < 3:
            continue

        score = 30
        if has_name_hint(profile.normalized_name, REGRESSION_TARGET_HINTS):
            score += 55
        if profile.unique_count >= min(10, max(3, row_count // 3)):
            score += 10
        if profile.missing_ratio:
            score -= 15

        candidates.append(
            Candidate(
                task_type="regression",
                target_column=profile.name,
                score=score,
                reason=f"{profile.name} is numeric and looks suitable for regression.",
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def build_forecasting_candidates(
    profiles: dict[str, ColumnProfile],
    date_candidates: list[Candidate],
    row_count: int,
) -> list[Candidate]:
    if row_count < MIN_FORECASTING_ROWS or not date_candidates:
        return []

    candidates = []
    best_date = date_candidates[0]

    for profile in profiles.values():
        if profile.name == best_date.target_column:
            continue

        if not is_usable_target(profile) or not profile.is_numeric:
            continue

        if not has_name_hint(profile.normalized_name, FORECASTING_TARGET_HINTS):
            continue

        score = 45 + best_date.score
        if profile.unique_count >= 3:
            score += 10
        if profile.missing_ratio:
            score -= 15

        candidates.append(
            Candidate(
                task_type="forecasting",
                target_column=profile.name,
                date_column=best_date.target_column,
                score=score,
                date_score=best_date.score,
                reason=(
                    f"{profile.name} is numeric and {best_date.target_column} "
                    "looks like a date column, so forecasting is plausible."
                ),
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def get_best_candidate(candidates: list[Candidate]) -> Candidate:
    if candidates:
        return candidates[0]

    return Candidate(
        task_type="classification",
        target_column="",
        score=0,
        reason="No clearly suitable target column was found.",
    )


def choose_recommendation(
    forecasting_candidates: list[Candidate],
    regression_candidates: list[Candidate],
    classification_candidates: list[Candidate],
) -> Candidate:
    all_candidates = (
        forecasting_candidates
        + regression_candidates
        + classification_candidates
    )

    if not all_candidates:
        return get_best_candidate([])

    task_priority = {
        "forecasting": 2,
        "regression": 1,
        "classification": 0,
    }
    return sorted(
        all_candidates,
        key=lambda candidate: (
            candidate.score,
            task_priority[candidate.task_type],
        ),
        reverse=True,
    )[0]


def build_warnings(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    recommendation: Candidate,
    date_candidates: list[Candidate],
) -> list[str]:
    warnings = []
    row_count = len(df)
    duplicate_rows = int(df.duplicated().sum())
    missing_cell_ratio = float(df.isna().sum().sum() / df.size) if df.size else 1.0

    if row_count < 20:
        warnings.append("The dataset has very few rows, so ML results may be weak.")
    elif row_count < 50:
        warnings.append("The dataset is small, so results should be reviewed carefully.")

    if missing_cell_ratio > 0:
        warnings.append("Some values are missing, which can reduce model quality.")

    if duplicate_rows > 0:
        warnings.append("Duplicate rows were found and may affect analysis quality.")

    missing_heavy_columns = [
        profile.name
        for profile in profiles.values()
        if profile.is_missing_heavy
    ]
    if missing_heavy_columns:
        warnings.append(
            "Some columns have many missing values and were avoided as targets."
        )

    if not recommendation.target_column:
        warnings.append("No clearly suitable target column was found.")

    if recommendation.task_type == "forecasting" and not date_candidates:
        warnings.append("No reliable date column was found for forecasting.")

    if recommendation.target_column:
        feature_columns = [
            profile
            for profile in profiles.values()
            if profile.name != recommendation.target_column
            and not profile.is_missing_heavy
        ]
        if recommendation.date_column:
            feature_columns = [
                profile
                for profile in feature_columns
                if profile.name != recommendation.date_column
            ]
        if not feature_columns:
            warnings.append("The dataset has few usable feature columns for ML.")

    if recommendation.task_type == "classification" and recommendation.target_column:
        target_counts = df[recommendation.target_column].value_counts(dropna=True)
        if not target_counts.empty:
            largest_class_ratio = float(target_counts.iloc[0] / target_counts.sum())
            if largest_class_ratio > 0.80:
                warnings.append(
                    "The recommended classification target appears imbalanced."
                )

    return warnings


def calculate_health_score(
    df: pd.DataFrame,
    recommendation: Candidate,
    warnings: list[str],
) -> int:
    row_count = len(df)
    score = 100
    missing_cell_ratio = float(df.isna().sum().sum() / df.size) if df.size else 1.0
    duplicate_ratio = float(df.duplicated().sum() / row_count) if row_count else 1.0

    if row_count < 20:
        score -= 25
    elif row_count < 50:
        score -= 15
    elif row_count < 100:
        score -= 5

    if missing_cell_ratio >= 0.25:
        score -= 25
    elif missing_cell_ratio >= 0.10:
        score -= 15
    elif missing_cell_ratio > 0:
        score -= 5

    if duplicate_ratio >= 0.25:
        score -= 15
    elif duplicate_ratio > 0:
        score -= 5

    if not recommendation.target_column:
        score -= 25

    if recommendation.score < 50:
        score -= 15
    elif recommendation.score < 75:
        score -= 5

    if any("imbalanced" in warning for warning in warnings):
        score -= 10

    return max(0, min(100, int(score)))


def confidence_from_score(score: int, health_score: int) -> Literal["high", "medium", "low"]:
    if score >= 90 and health_score >= 75:
        return "high"

    if score >= 55 and health_score >= 50:
        return "medium"

    return "low"


def build_alternatives(
    recommendation: Candidate,
    candidates: list[Candidate],
) -> list[AnalysisRecommendationAlternative]:
    alternatives = []

    for candidate in candidates:
        if (
            candidate.task_type == recommendation.task_type
            and candidate.target_column == recommendation.target_column
            and candidate.date_column == recommendation.date_column
        ):
            continue

        alternatives.append(
            AnalysisRecommendationAlternative(
                task_type=candidate.task_type,
                target_column=candidate.target_column,
                date_column=candidate.date_column,
                reason=candidate.reason,
            )
        )

        if len(alternatives) == 3:
            break

    return alternatives


def build_reasons(
    recommendation: Candidate,
    health_score: int,
) -> list[str]:
    reasons = [recommendation.reason]

    if recommendation.task_type == "forecasting" and recommendation.date_column:
        reasons.append(
            f"{recommendation.date_column} will be used as the time column."
        )

    if health_score >= 75:
        reasons.append("The dataset health score is strong enough for a first ML run.")
    elif health_score >= 50:
        reasons.append("The dataset is usable, but the result should be reviewed.")
    else:
        reasons.append("The dataset needs cleanup or review before trusting ML results.")

    return reasons


def get_analysis_recommendation(
    db: Session,
    dataset_id: int,
    current_user: User,
) -> AnalysisRecommendationResponse:
    dataset = get_owned_dataset(db, dataset_id, current_user)

    if not Path(dataset.file_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found on server",
        )

    df = read_stored_dataset_file(dataset.file_path)
    profiles = build_column_profiles(df)
    row_count = len(df)
    date_candidates = build_date_candidates(profiles)
    forecasting_candidates = build_forecasting_candidates(
        profiles=profiles,
        date_candidates=date_candidates,
        row_count=row_count,
    )
    regression_candidates = build_regression_candidates(
        profiles=profiles,
        row_count=row_count,
    )
    classification_candidates = build_classification_candidates(
        profiles=profiles,
        row_count=row_count,
    )
    recommendation = choose_recommendation(
        forecasting_candidates=forecasting_candidates,
        regression_candidates=regression_candidates,
        classification_candidates=classification_candidates,
    )
    warnings = build_warnings(
        df=df,
        profiles=profiles,
        recommendation=recommendation,
        date_candidates=date_candidates,
    )
    health_score = calculate_health_score(
        df=df,
        recommendation=recommendation,
        warnings=warnings,
    )
    all_candidates = sorted(
        forecasting_candidates
        + regression_candidates
        + classification_candidates,
        key=lambda candidate: candidate.score,
        reverse=True,
    )

    return AnalysisRecommendationResponse(
        dataset_id=dataset.id,
        recommended_task_type=recommendation.task_type,
        recommended_target_column=recommendation.target_column,
        recommended_date_column=recommendation.date_column,
        confidence=confidence_from_score(recommendation.score, health_score),
        health_score=health_score,
        reasons=build_reasons(recommendation, health_score),
        warnings=warnings,
        alternatives=build_alternatives(recommendation, all_candidates),
    )
