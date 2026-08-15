"""P_behavior의 데이터 가용성 및 파생 feature 계산 테스트."""

from datetime import datetime

from app.analyzers.behavior import BehaviorInput, analyze_behavior


REVIEW_DATE = datetime(2026, 8, 15, 12, 0)


def test_no_behavior_data_is_unavailable() -> None:
    result = analyze_behavior(BehaviorInput())

    assert result.available is False
    assert result.p_behavior is None
    assert result.unavailable_reasons == ("insufficient_behavior_evidence",)
    assert result.features.verified_purchase is None
    assert result.features.account_age_days is None
    assert result.features.reviews_written_today is None


def test_review_date_alone_is_not_sufficient() -> None:
    result = analyze_behavior(BehaviorInput(review_date=REVIEW_DATE))

    assert result.available is False
    assert result.p_behavior is None
    assert "insufficient_behavior_evidence" in result.unavailable_reasons


def test_verified_purchase_is_the_only_populated_feature_when_provided() -> None:
    result = analyze_behavior(BehaviorInput(verified_purchase=False))

    assert result.available is True
    assert result.p_behavior == 70.0
    assert result.features.verified_purchase is False
    assert result.features.account_age_days is None
    assert result.features.reviews_written_today is None
    assert result.reasons[0].message == "구매 인증 리뷰가 아님"


def test_verified_purchase_states_remain_distinct() -> None:
    missing = analyze_behavior(BehaviorInput(verified_purchase=None))
    verified = analyze_behavior(BehaviorInput(verified_purchase=True))
    not_verified = analyze_behavior(BehaviorInput(verified_purchase=False))

    assert missing.available is False
    assert missing.p_behavior is None
    assert missing.features.verified_purchase is None
    assert verified.available is True
    assert verified.p_behavior == 100.0
    assert verified.features.verified_purchase is True
    assert not_verified.available is True
    assert not_verified.p_behavior == 70.0
    assert not_verified.features.verified_purchase is False


def test_account_age_requires_both_dates() -> None:
    only_account_date = analyze_behavior(
        BehaviorInput(account_created_at=datetime(2026, 8, 1))
    )
    both_dates = analyze_behavior(
        BehaviorInput(
            review_date=REVIEW_DATE,
            account_created_at=datetime(2026, 8, 1),
        )
    )

    assert only_account_date.features.account_age_days is None
    assert both_dates.features.account_age_days == 14
    assert both_dates.available is False
    assert both_dates.p_behavior is None


def test_reviews_written_today_requires_stable_user_and_multiple_dates() -> None:
    review_dates = (
        datetime(2026, 8, 15, 8, 0),
        datetime(2026, 8, 15, 13, 0),
        datetime(2026, 8, 14, 9, 0),
    )

    without_user_id = analyze_behavior(
        BehaviorInput(review_date=REVIEW_DATE, user_review_dates=review_dates)
    )
    one_date = analyze_behavior(
        BehaviorInput(
            review_date=REVIEW_DATE,
            user_id="user-1",
            user_review_dates=(review_dates[0],),
        )
    )
    complete = analyze_behavior(
        BehaviorInput(
            review_date=REVIEW_DATE,
            user_id="user-1",
            user_review_dates=review_dates,
        )
    )

    assert without_user_id.features.reviews_written_today is None
    assert one_date.features.reviews_written_today is None
    assert complete.features.reviews_written_today == 2
    assert complete.available is True
    assert complete.p_behavior == 100.0


def test_missing_values_are_not_coerced_to_zero_or_false() -> None:
    result = analyze_behavior(BehaviorInput(review_date=REVIEW_DATE))

    assert result.available is False
    assert result.p_behavior is None
    assert result.features.verified_purchase is None
    assert result.features.account_age_days is None
    assert result.features.reviews_written_today is None
