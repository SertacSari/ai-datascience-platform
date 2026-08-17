import pandas as pd
import pytest
from fastapi import HTTPException

from app.models.dataset import Dataset
from app.models.user import User
from app.services.analysis_recommendation_service import get_analysis_recommendation


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


def add_dataset_with_frame(
    db_session,
    tmp_path,
    user: User,
    dataframe: pd.DataFrame,
    file_name: str = "dataset.csv",
) -> Dataset:
    dataset_path = tmp_path / file_name
    dataframe.to_csv(dataset_path, index=False)
    dataset = Dataset(
        user_id=user.id,
        file_name=file_name,
        file_path=str(dataset_path),
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
    )
    db_session.add(dataset)
    db_session.commit()
    db_session.refresh(dataset)
    return dataset


def house_price_frame(row_count: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "house_id": [f"home_{index}" for index in range(row_count)],
            "size_sqft": [700 + index * 12 for index in range(row_count)],
            "bedrooms": [2 + index % 4 for index in range(row_count)],
            "neighborhood": [
                "central" if index % 3 == 0 else "suburban"
                for index in range(row_count)
            ],
            "sale_price": [
                120_000 + index * 2_500 for index in range(row_count)
            ],
        }
    )


def daily_sales_frame(row_count: int = 60) -> pd.DataFrame:
    start_date = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        {
            "date": [
                (start_date + pd.Timedelta(days=index)).date().isoformat()
                for index in range(row_count)
            ],
            "promotion": [
                "yes" if index % 10 in {0, 1} else "no"
                for index in range(row_count)
            ],
            "sales": [200 + index * 4 for index in range(row_count)],
        }
    )


def churn_frame(row_count: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"customer_{index}" for index in range(row_count)],
            "plan": ["basic" if index % 2 else "pro" for index in range(row_count)],
            "support_tickets": [index % 5 for index in range(row_count)],
            "churned": ["yes" if index % 3 == 0 else "no" for index in range(row_count)],
        }
    )


def get_recommendation_for_frame(
    db_session,
    tmp_path,
    dataframe: pd.DataFrame,
    username: str,
):
    user = add_user(db_session, username)
    dataset = add_dataset_with_frame(db_session, tmp_path, user, dataframe)

    return get_analysis_recommendation(
        db=db_session,
        dataset_id=dataset.id,
        current_user=user,
    )


def guidance_by_column(recommendation) -> dict[str, object]:
    return {guidance.column: guidance for guidance in recommendation.column_guidance}


def test_house_price_dataset_recommends_regression_with_sale_price(
    db_session,
    tmp_path,
) -> None:
    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        house_price_frame(),
        "house_price_user",
    )

    assert recommendation.recommended_task_type == "regression"
    assert recommendation.recommended_target_column == "sale_price"
    assert recommendation.recommended_date_column is None
    assert recommendation.confidence in {"high", "medium"}
    assert recommendation.health_score >= 70
    assert any("sale_price is numeric" in reason for reason in recommendation.reasons)


def test_recommended_target_has_target_explanation(db_session, tmp_path) -> None:
    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        house_price_frame(),
        "target_explanation_user",
    )

    assert recommendation.target_explanation.column == "sale_price"
    assert "sale_price is recommended" in recommendation.target_explanation.message
    assert "Numeric column" in recommendation.target_explanation.strengths
    assert "Name suggests a prediction target" in (
        recommendation.target_explanation.strengths
    )
    assert recommendation.target_explanation.risks == []


def test_house_price_guidance_marks_target_and_features(db_session, tmp_path) -> None:
    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        house_price_frame(),
        "house_price_guidance_user",
    )
    guidance = guidance_by_column(recommendation)

    assert guidance["sale_price"].role == "recommended_target"
    assert guidance["sale_price"].severity == "low"
    assert guidance["size_sqft"].role == "useful_feature"
    assert guidance["size_sqft"].severity == "low"
    assert guidance["bedrooms"].role == "useful_feature"


def test_daily_sales_dataset_recommends_forecasting_with_sales_and_date(
    db_session,
    tmp_path,
) -> None:
    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        daily_sales_frame(),
        "daily_sales_user",
    )

    assert recommendation.recommended_task_type == "forecasting"
    assert recommendation.recommended_target_column == "sales"
    assert recommendation.recommended_date_column == "date"
    assert recommendation.confidence == "high"


def test_customer_churn_dataset_recommends_classification_with_churned(
    db_session,
    tmp_path,
) -> None:
    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        churn_frame(),
        "churn_user",
    )

    assert recommendation.recommended_task_type == "classification"
    assert recommendation.recommended_target_column == "churned"
    assert recommendation.confidence in {"high", "medium"}


def test_id_like_columns_are_not_selected_as_targets(db_session, tmp_path) -> None:
    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        house_price_frame(),
        "id_like_user",
    )

    assert recommendation.recommended_target_column not in {
        "house_id",
        "customer_id",
        "id",
    }
    assert guidance_by_column(recommendation)["house_id"].role == (
        "not_recommended_target"
    )


def test_missing_heavy_columns_reduce_confidence_and_health(
    db_session,
    tmp_path,
) -> None:
    dataframe = house_price_frame(row_count=50)
    dataframe.loc[:34, "sale_price"] = None

    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        dataframe,
        "missing_heavy_user",
    )

    assert recommendation.recommended_target_column != "sale_price"
    assert recommendation.confidence in {"medium", "low"}
    assert recommendation.health_score < 80
    assert any("missing" in warning.lower() for warning in recommendation.warnings)
    guidance = guidance_by_column(recommendation)
    assert guidance["sale_price"].role == "not_recommended_target"
    assert guidance["sale_price"].severity == "high"


def test_date_column_same_as_target_is_not_recommended(
    db_session,
    tmp_path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=40).astype(str),
            "category": ["A", "B"] * 20,
        }
    )

    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        dataframe,
        "date_target_user",
    )

    assert recommendation.recommended_target_column != recommendation.recommended_date_column


def test_date_column_is_marked_as_possible_date_column(db_session, tmp_path) -> None:
    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        daily_sales_frame(),
        "date_guidance_user",
    )

    guidance = guidance_by_column(recommendation)

    assert guidance["date"].role == "possible_date_column"
    assert guidance["date"].severity == "low"


def test_all_categorical_dataset_can_recommend_classification(
    db_session,
    tmp_path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "segment": ["A", "B", "C", "A"] * 15,
            "region": ["east", "west"] * 30,
            "status": ["approved", "rejected"] * 30,
        }
    )

    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        dataframe,
        "categorical_user",
    )

    assert recommendation.recommended_task_type == "classification"
    assert recommendation.recommended_target_column == "status"


def test_numeric_dataset_without_target_like_names_returns_low_or_medium_confidence(
    db_session,
    tmp_path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": range(60),
            "feature_b": [index * 2 for index in range(60)],
            "measurement": [100 + index * 3 for index in range(60)],
        }
    )

    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        dataframe,
        "numeric_generic_user",
    )

    assert recommendation.recommended_task_type == "regression"
    assert recommendation.recommended_target_column in dataframe.columns
    assert recommendation.confidence in {"low", "medium"}


def test_zip_postal_code_columns_get_caution(db_session, tmp_path) -> None:
    dataframe = house_price_frame()
    dataframe["zip"] = ["10001", "10002"] * 40

    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        dataframe,
        "zip_guidance_user",
    )
    guidance = guidance_by_column(recommendation)["zip"]

    assert guidance.role in {"useful_feature", "not_recommended_target"}
    assert guidance.severity == "medium"
    assert "reviewed carefully" in guidance.message


def test_leakage_like_column_creates_high_severity_guidance_and_warning(
    db_session,
    tmp_path,
) -> None:
    dataframe = churn_frame()
    dataframe["churn_reason"] = [
        "price" if value == "yes" else "none" for value in dataframe["churned"]
    ]

    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        dataframe,
        "leakage_guidance_user",
    )
    guidance = guidance_by_column(recommendation)["churn_reason"]

    assert recommendation.recommended_target_column == "churned"
    assert guidance.role == "useful_feature"
    assert guidance.severity == "high"
    assert "leak the answer" in guidance.message
    assert any("leak the answer" in warning for warning in recommendation.warnings)


def test_constant_column_gets_high_severity_not_recommended_target(
    db_session,
    tmp_path,
) -> None:
    dataframe = house_price_frame()
    dataframe["constant_flag"] = "same"

    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        dataframe,
        "constant_guidance_user",
    )
    guidance = guidance_by_column(recommendation)["constant_flag"]

    assert guidance.role == "not_recommended_target"
    assert guidance.severity == "high"
    assert "only one unique value" in guidance.message


def test_analysis_recommendation_enforces_dataset_ownership(
    db_session,
    tmp_path,
) -> None:
    owner = add_user(db_session, "recommendation_owner")
    other_user = add_user(db_session, "recommendation_other")
    dataset = add_dataset_with_frame(
        db_session,
        tmp_path,
        owner,
        house_price_frame(),
    )

    with pytest.raises(HTTPException) as error:
        get_analysis_recommendation(
            db=db_session,
            dataset_id=dataset.id,
            current_user=other_user,
        )

    assert error.value.status_code == 404
    assert error.value.detail == "Dataset not found"


def test_analysis_recommendation_response_shape_is_stable(
    db_session,
    tmp_path,
) -> None:
    recommendation = get_recommendation_for_frame(
        db_session,
        tmp_path,
        daily_sales_frame(),
        "shape_user",
    )

    assert set(recommendation.model_dump()) == {
        "dataset_id",
        "recommended_task_type",
        "recommended_target_column",
        "recommended_date_column",
        "confidence",
        "health_score",
        "reasons",
        "warnings",
        "alternatives",
        "target_explanation",
        "column_guidance",
    }
    assert recommendation.alternatives
    assert set(recommendation.alternatives[0].model_dump()) == {
        "task_type",
        "target_column",
        "date_column",
        "reason",
    }
    assert set(recommendation.target_explanation.model_dump()) == {
        "column",
        "message",
        "strengths",
        "risks",
    }
    assert set(recommendation.column_guidance[0].model_dump()) == {
        "column",
        "role",
        "severity",
        "message",
    }
