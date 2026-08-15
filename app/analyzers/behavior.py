"""실제 리뷰·사용자 메타데이터로 P_behavior를 분석하는 모듈.

역할:
- 제공된 구매·계정·작성 이력에서 행동 feature와 P_behavior를 계산한다.
- 파생 feature는 필요한 원천 데이터가 모두 있을 때만 계산한다.
- 근거가 부족하면 0점이나 정상 점수 대신 unavailable을 반환한다.

수정 범위:
- [AI CORE]
- 행동 feature 또는 scoring 정책 변경은 AI 담당자와 협의한다.
- Spring/크롤링 연동을 위해 analyzer 내부 로직을 직접 수정하지 않는다.

주의:
- 없는 행동 데이터를 0이나 False로 보정하거나 임의 생성하지 않는다.
- 현재 30점·15점 감점은 Ground Truth로 검증되지 않은 v0 heuristic이다.
- crawler를 직접 호출하지 않는다.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class BehaviorInput:
    """Spring이 실제 확보한 경우에만 값을 채우는 행동 분석 입력."""

    review_date: datetime | None = None
    user_id: str | None = None
    # None은 구매 인증 데이터 미제공, True는 구매 인증 확인,
    # False는 구매 인증 리뷰가 아님을 각각 의미한다.
    verified_purchase: bool | None = None
    account_created_at: datetime | None = None
    user_review_dates: tuple[datetime, ...] | None = None


@dataclass(frozen=True, slots=True)
class BehaviorFeatures:
    """관측값 또는 충분한 원천값에서 계산된 행동 feature."""

    verified_purchase: bool | None = None
    account_age_days: int | None = None
    reviews_written_today: int | None = None


@dataclass(frozen=True, slots=True)
class BehaviorReason:
    """P_behavior 점수에 영향을 준 판단 근거."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class BehaviorAnalysisResult:
    """P_behavior 점수 또는 명시적인 unavailable 상태."""

    available: bool
    p_behavior: float | None
    features: BehaviorFeatures
    reasons: tuple[BehaviorReason, ...]
    unavailable_reasons: tuple[str, ...]


def analyze_behavior(data: BehaviorInput) -> BehaviorAnalysisResult:
    """사용 가능한 행동 feature만 계산하고 0~100 범위의 P_behavior를 반환한다."""

    account_age_days: int | None = None
    unavailable_reasons: list[str] = []

    if data.account_created_at is not None and data.review_date is not None:
        calculated_age = (data.review_date.date() - data.account_created_at.date()).days
        if calculated_age >= 0:
            account_age_days = calculated_age
        else:
            unavailable_reasons.append("account_created_after_review")

    reviews_written_today: int | None = None
    has_stable_user_id = bool(data.user_id and data.user_id.strip())
    if (
        has_stable_user_id
        and data.review_date is not None
        and data.user_review_dates is not None
        and len(data.user_review_dates) >= 2
    ):
        review_day = data.review_date.date()
        reviews_written_today = sum(
            timestamp.date() == review_day for timestamp in data.user_review_dates
        )

    features = BehaviorFeatures(
        verified_purchase=data.verified_purchase,
        account_age_days=account_age_days,
        reviews_written_today=reviews_written_today,
    )

    has_scoreable_feature = (
        data.verified_purchase is not None or reviews_written_today is not None
    )
    if not has_scoreable_feature:
        unavailable_reasons.append("insufficient_behavior_evidence")
        return BehaviorAnalysisResult(
            available=False,
            p_behavior=None,
            features=features,
            reasons=(),
            unavailable_reasons=tuple(unavailable_reasons),
        )

    # TODO: verified_purchase 하나만으로도 100점이 될 수 있는 v0 한계가 있다.
    # Meta-Scorer 연동 시 evidence_count와 confidence를 함께 고려해야 한다.
    score = 100.0
    reasons: list[BehaviorReason] = []

    if data.verified_purchase is False:
        score -= 30.0
        reasons.append(
            BehaviorReason(
                code="PURCHASE_NOT_VERIFIED",
                message="구매 인증 리뷰가 아님",
            )
        )

    if reviews_written_today is not None and reviews_written_today >= 3:
        score -= 15.0
        reasons.append(
            BehaviorReason(
                code="MULTIPLE_REVIEWS_SAME_DAY",
                message="동일 사용자의 같은 날짜 다수 리뷰 작성",
            )
        )

    return BehaviorAnalysisResult(
        available=True,
        p_behavior=max(score, 0.0),
        features=features,
        reasons=tuple(reasons),
        unavailable_reasons=tuple(unavailable_reasons),
    )
