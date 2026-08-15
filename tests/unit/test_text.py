"""P_text 규칙과 외부 sentiment 상태 처리에 대한 단위 테스트."""

from app.analyzers.text import analyze_text
from app.integrations.sentiment import SentimentResult, SentimentStatus


class AvailableSentimentStub:
    """사용 가능한 외부 감성 결과를 제공하는 테스트 대역."""

    def analyze(self, content: str) -> SentimentResult:
        return SentimentResult(
            status=SentimentStatus.AVAILABLE,
            score=0.9,
            magnitude=2.0,
        )


class UnavailableSentimentStub:
    """외부 감성 서비스 실패 결과를 제공하는 테스트 대역."""

    def analyze(self, content: str) -> SentimentResult:
        return SentimentResult(
            status=SentimentStatus.UNAVAILABLE,
            unavailable_reason="service_error:TimeoutError",
        )


def test_rule_based_features_and_reasons() -> None:
    result = analyze_text("최고 최고!!!")

    assert result.p_text == 50.0
    assert result.features.character_count == 8
    assert result.features.exclamation_count == 3
    assert result.features.repeated_keywords == {"최고": 2}
    assert [reason.code for reason in result.reasons] == [
        "REPETITIVE_KEYWORD",
        "SHORT_REVIEW",
        "EXCESSIVE_EXCLAMATION",
    ]
    assert result.sentiment is None


def test_available_sentiment_can_adjust_score() -> None:
    result = analyze_text(
        "충분히 긴 리뷰 본문으로 감성 분석 보조 신호를 확인합니다.",
        sentiment_analyzer=AvailableSentimentStub(),
    )

    assert result.p_text == 95.0
    assert result.sentiment is not None
    assert result.sentiment.status is SentimentStatus.AVAILABLE
    assert result.reasons[-1].code == "OVERLY_POSITIVE_SENTIMENT"


def test_unavailable_sentiment_is_explicit_and_does_not_change_score() -> None:
    result = analyze_text(
        "충분히 긴 리뷰 본문으로 서비스 실패 상태를 확인합니다.",
        sentiment_analyzer=UnavailableSentimentStub(),
    )

    assert result.p_text == 100.0
    assert result.sentiment is not None
    assert result.sentiment.status is SentimentStatus.UNAVAILABLE
    assert result.sentiment.score is None
    assert result.sentiment.magnitude is None
    assert result.sentiment.unavailable_reason == "service_error:TimeoutError"
