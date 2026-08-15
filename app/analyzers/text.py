"""Re:view의 리뷰 텍스트 신호(P_text)를 분석하는 모듈.

역할:
- 리뷰 본문에서 규칙 기반 feature와 현재 v0 P_text를 계산한다.
- 외부 감성 결과는 실제 사용 가능한 경우에만 보조 신호로 사용한다.
- 향후 학습 모델 출력은 ``features``와 별도 판단 근거로 확장한다.

수정 범위:
- [AI CORE]
- Spring/크롤링 연동을 위해 직접 수정하지 않는다.
- feature 또는 scoring 정책 변경은 AI 담당자와 협의한다.

주의:
- 일부 규칙은 Ground Truth로 검증된 최종 모델이 아닌 MVP용 v0 heuristic이다.
- crawler를 직접 호출하거나 확보되지 않은 입력 데이터를 생성하지 않는다.
"""

from dataclasses import dataclass
from typing import Protocol

from app.integrations.sentiment import SentimentResult, SentimentStatus


REPETITIVE_KEYWORDS: tuple[str, ...] = (
    "최고",
    "대박",
    "강력추천",
    "완전",
    "무조건",
    "좋아요",
    "만족",
)


class SentimentAnalyzer(Protocol):
    """P_text가 사용하는 외부 감성 분석 adapter의 최소 인터페이스."""

    def analyze(self, content: str) -> SentimentResult:
        """본문의 감성 분석 결과를 반환한다."""


@dataclass(frozen=True, slots=True)
class TextReason:
    """P_text 판단에 영향을 준 규칙과 설명."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class TextFeatures:
    """실제 리뷰 본문에서 직접 계산한 규칙 기반 feature."""

    character_count: int
    exclamation_count: int
    repeated_keywords: dict[str, int]


@dataclass(frozen=True, slots=True)
class TextAnalysisResult:
    """Spring 응답 스키마로 변환할 수 있는 P_text 분석 결과."""

    p_text: float
    features: TextFeatures
    reasons: tuple[TextReason, ...]
    sentiment: SentimentResult | None


def analyze_text(
    content: str,
    *,
    sentiment_analyzer: SentimentAnalyzer | None = None,
) -> TextAnalysisResult:
    """리뷰 본문을 분석해 0~100 범위의 P_text와 판단 근거를 반환한다.

    ``sentiment_analyzer``가 없으면 외부 감성 분석을 요청하지 않는다. adapter가
    unavailable을 반환한 경우 그 상태를 그대로 노출하고 점수에는 반영하지 않는다.
    """

    if not isinstance(content, str):
        raise TypeError("content must be a string")

    stripped_content = content.strip()
    repeated_keywords = {
        keyword: count
        for keyword in REPETITIVE_KEYWORDS
        if (count := content.count(keyword)) >= 2
    }
    exclamation_count = content.count("!")
    features = TextFeatures(
        character_count=len(stripped_content),
        exclamation_count=exclamation_count,
        repeated_keywords=repeated_keywords,
    )

    score = 100.0
    reasons: list[TextReason] = []

    for keyword, count in repeated_keywords.items():
        score -= 15.0
        reasons.append(
            TextReason(
                code="REPETITIVE_KEYWORD",
                message=f"반복 표현 탐지: '{keyword}' {count}회",
            )
        )

    if features.character_count < 20:
        score -= 25.0
        reasons.append(
            TextReason(
                code="SHORT_REVIEW",
                message="리뷰 내용이 지나치게 짧음",
            )
        )

    if exclamation_count >= 3:
        score -= 10.0
        reasons.append(
            TextReason(
                code="EXCESSIVE_EXCLAMATION",
                message="과도한 느낌표 사용",
            )
        )

    sentiment = (
        sentiment_analyzer.analyze(content) if sentiment_analyzer is not None else None
    )
    if (
        sentiment is not None
        and sentiment.status is SentimentStatus.AVAILABLE
        and sentiment.score is not None
        and sentiment.magnitude is not None
        and sentiment.score > 0.8
        and sentiment.magnitude > 1.5
    ):
        score -= 5.0
        reasons.append(
            TextReason(
                code="OVERLY_POSITIVE_SENTIMENT",
                message="과도하게 긍정적인 감성 표현 탐지",
            )
        )

    return TextAnalysisResult(
        p_text=max(score, 0.0),
        features=features,
        reasons=tuple(reasons),
        sentiment=sentiment,
    )
