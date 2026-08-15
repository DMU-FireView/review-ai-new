"""상품 리뷰를 analyzer와 Meta-Scorer에 연결하는 분석 orchestration 모듈.

역할:
- Spring/연동 계층의 정규화된 데이터를 analyzer 입력으로 변환한다.
- P_text, P_behavior, P_network를 실행하고 Meta-Scorer에 전달한다.
- 리뷰별 RTI 결과와 구조화된 판단 근거를 조립한다.

수정 범위:
- [AI ORCHESTRATION]
- 입력 연결 방식 변경 시 수정할 수 있으나 analyzer 점수 의미는 변경하지 않는다.
- 구조 변경 전 AI 담당자와 협의한다.

책임 흐름:
Crawler/Normalization → Spring/연동 계층 → AI Schema → Analysis Service
→ P_text/P_behavior/P_network → Meta-Scorer → RTI.
크롤링 필드 변환은 schema/API/별도 integration 계층에서 수행하며, crawler 코드를
analyzer 내부에 넣지 않는다. 실제 원천 데이터에 없는 값도 생성하지 않는다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

from app.analyzers.behavior import BehaviorInput, analyze_behavior
from app.analyzers.network import (
    NetworkReview,
    SimilarityAdapter,
    analyze_network,
)
from app.analyzers.text import SentimentAnalyzer, analyze_text
from app.scoring.meta_scorer import (
    MetaScoreInput,
    RTILevel,
    ScoreSignal,
    calculate_meta_score,
)


@dataclass(frozen=True, slots=True)
class ReviewAnalysisInput:
    """Spring API schema와 분리된 리뷰 분석 서비스의 최소 입력."""

    review_id: str
    product_id: str
    content: str
    user_id: str | None = None
    review_date: datetime | None = None
    verified_purchase: bool | None = None
    account_created_at: datetime | None = None
    user_review_dates: tuple[datetime, ...] | None = None


@dataclass(frozen=True, slots=True)
class AnalysisSignal:
    """최종 결과에서 확인할 수 있는 analyzer 신호 상태."""

    available: bool
    score: float | None
    unavailable_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalysisSignals:
    """리뷰 하나의 P_text, P_behavior, P_network 상태."""

    text: AnalysisSignal
    behavior: AnalysisSignal
    network: AnalysisSignal


@dataclass(frozen=True, slots=True)
class AnalysisReason:
    """출처를 보존한 analyzer 판단 근거."""

    source: Literal["text", "behavior", "network"]
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ReviewAnalysisResult:
    """리뷰별 analyzer 신호와 최종 RTI 결과."""

    review_id: str
    product_id: str
    available: bool
    rti: float | None
    level: RTILevel | None
    signals: AnalysisSignals
    used_signals: tuple[str, ...]
    unavailable_signals: tuple[str, ...]
    applied_weights: dict[str, float]
    used_signal_count: int
    reasons: tuple[AnalysisReason, ...]
    unavailable_reason: str | None


def analyze_product_reviews(
    product_id: str,
    reviews: Sequence[ReviewAnalysisInput],
    *,
    sentiment_analyzer: SentimentAnalyzer | None = None,
    similarity_adapter: SimilarityAdapter | None = None,
) -> tuple[ReviewAnalysisResult, ...]:
    """동일 상품 리뷰를 입력 순서대로 분석해 리뷰별 최종 결과를 반환한다."""

    _validate_product_input(product_id, reviews)
    network_reviews = tuple(
        NetworkReview(
            review_id=review.review_id,
            product_id=review.product_id,
            content=review.content,
        )
        for review in reviews
    )

    results: list[ReviewAnalysisResult] = []
    for review, network_target in zip(reviews, network_reviews, strict=True):
        text_result = analyze_text(
            review.content,
            sentiment_analyzer=sentiment_analyzer,
        )
        behavior_result = analyze_behavior(
            BehaviorInput(
                review_date=review.review_date,
                user_id=review.user_id,
                verified_purchase=review.verified_purchase,
                account_created_at=review.account_created_at,
                user_review_dates=review.user_review_dates,
            )
        )
        comparisons = tuple(
            candidate
            for candidate in network_reviews
            if candidate.review_id != network_target.review_id
        )
        network_result = analyze_network(
            network_target,
            comparisons,
            similarity_adapter=similarity_adapter,
        )

        signals = AnalysisSignals(
            text=AnalysisSignal(available=True, score=text_result.p_text),
            behavior=AnalysisSignal(
                available=behavior_result.available,
                score=behavior_result.p_behavior,
                unavailable_reasons=behavior_result.unavailable_reasons,
            ),
            network=AnalysisSignal(
                available=network_result.available,
                score=network_result.p_network,
                unavailable_reasons=(network_result.unavailable_reason,)
                if network_result.unavailable_reason is not None
                else (),
            ),
        )
        meta_result = calculate_meta_score(
            MetaScoreInput(
                text=_to_score_signal(signals.text),
                behavior=_to_score_signal(signals.behavior),
                network=_to_score_signal(signals.network),
            )
        )
        reasons = (
            tuple(
                AnalysisReason("text", reason.code, reason.message)
                for reason in text_result.reasons
            )
            + tuple(
                AnalysisReason("behavior", reason.code, reason.message)
                for reason in behavior_result.reasons
            )
            + tuple(
                AnalysisReason("network", reason.code, reason.message)
                for reason in network_result.reasons
            )
        )
        results.append(
            ReviewAnalysisResult(
                review_id=review.review_id,
                product_id=review.product_id,
                available=meta_result.available,
                rti=meta_result.rti,
                level=meta_result.level,
                signals=signals,
                used_signals=meta_result.used_signals,
                unavailable_signals=meta_result.unavailable_signals,
                applied_weights=meta_result.applied_weights,
                used_signal_count=meta_result.used_signal_count,
                reasons=reasons,
                unavailable_reason=meta_result.unavailable_reason,
            )
        )

    return tuple(results)


def _to_score_signal(signal: AnalysisSignal) -> ScoreSignal:
    return ScoreSignal(available=signal.available, score=signal.score)


def _validate_product_input(
    product_id: str,
    reviews: Sequence[ReviewAnalysisInput],
) -> None:
    if not product_id.strip():
        raise ValueError("product_id must not be blank")

    seen_review_ids: set[str] = set()
    for index, review in enumerate(reviews):
        if review.product_id != product_id:
            raise ValueError(
                f"reviews[{index}].product_id must match {product_id!r}"
            )
        if not review.review_id.strip():
            raise ValueError(f"reviews[{index}].review_id must not be blank")
        if review.review_id in seen_review_ids:
            raise ValueError(f"duplicate review_id: {review.review_id!r}")
        seen_review_ids.add(review.review_id)
