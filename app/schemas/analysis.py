"""분석 결과를 Spring으로 돌려주는 응답 schema 모듈.

역할:
- `ReviewAnalysisResult`를 직렬화 가능한 응답 구조로 옮긴다.
- crawler 원본 식별자(platform, review_id, product_id)를 함께 실어 Spring이 원본과 대조할 수 있게 한다.

수정 범위:
- [INTEGRATION]
- 응답 형태 변경은 Spring 담당자와 협의한다.
- analyzer 점수 의미나 RTI 등급 기준은 이 파일에서 변경하지 않는다.

주의:
- unavailable 신호를 0점·100점·중립값으로 채우지 않고 null과 사유를 그대로 노출한다.
"""

from pydantic import BaseModel, Field

from app.services.analysis import AnalysisSignal, ReviewAnalysisResult


class SignalResponse(BaseModel):
    """analyzer 신호 하나의 사용 가능 여부와 점수."""

    available: bool
    score: float | None = Field(default=None, description="0~100, unavailable이면 null")
    unavailable_reasons: list[str] = Field(default_factory=list)


class SignalsResponse(BaseModel):
    """리뷰 하나의 P_text, P_behavior, P_network 상태."""

    text: SignalResponse
    behavior: SignalResponse
    network: SignalResponse


class ReasonResponse(BaseModel):
    """출처를 보존한 analyzer 판단 근거."""

    source: str = Field(description="text | behavior | network")
    code: str
    message: str


class ReviewAnalysisResponse(BaseModel):
    """리뷰 한 건의 analyzer 신호와 최종 RTI 결과."""

    platform: str
    review_id: str = Field(description="platform 내 원본 리뷰 ID")
    product_id: str = Field(description="platform 내 원본 상품 ID")
    analysis_review_id: str = Field(description="분석에 사용한 platform 스코프 리뷰 키")
    product_key: str = Field(description="분석 단위를 묶은 상품 키")

    available: bool
    rti: float | None = Field(default=None, description="신호가 하나도 없으면 null")
    level: str | None = Field(default=None, description="safe | warn | danger")
    signals: SignalsResponse
    used_signals: list[str]
    used_signal_count: int
    unavailable_signals: list[str]
    applied_weights: dict[str, float]
    reasons: list[ReasonResponse]
    unavailable_reason: str | None = None


class ProductAnalysisResponse(BaseModel):
    """상품 하나에 대한 리뷰별 분석 결과 묶음."""

    product_key: str
    review_count: int
    results: list[ReviewAnalysisResponse]


def to_review_response(
    result: ReviewAnalysisResult,
    *,
    platform: str,
    review_id: str,
    product_id: str,
) -> ReviewAnalysisResponse:
    """분석 결과에 crawler 원본 식별자를 붙여 응답 구조로 옮긴다."""

    return ReviewAnalysisResponse(
        platform=platform,
        review_id=review_id,
        product_id=product_id,
        analysis_review_id=result.review_id,
        product_key=result.product_id,
        available=result.available,
        rti=result.rti,
        level=result.level.value if result.level is not None else None,
        signals=SignalsResponse(
            text=_to_signal_response(result.signals.text),
            behavior=_to_signal_response(result.signals.behavior),
            network=_to_signal_response(result.signals.network),
        ),
        used_signals=list(result.used_signals),
        used_signal_count=result.used_signal_count,
        unavailable_signals=list(result.unavailable_signals),
        applied_weights=dict(result.applied_weights),
        reasons=[
            ReasonResponse(source=reason.source, code=reason.code, message=reason.message)
            for reason in result.reasons
        ],
        unavailable_reason=result.unavailable_reason,
    )


def _to_signal_response(signal: AnalysisSignal) -> SignalResponse:
    return SignalResponse(
        available=signal.available,
        score=signal.score,
        unavailable_reasons=list(signal.unavailable_reasons),
    )
