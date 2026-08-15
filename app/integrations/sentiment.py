"""외부 감성 분석 서비스를 P_text에 연결하는 adapter 모듈.

수정 범위:
- [MODEL / EXTERNAL INTEGRATION]
- 외부 감성 모델이나 서비스 교체 시 수정할 수 있다.
- P_text scoring 정책은 이 파일에서 변경하지 않는다.

외부 결과가 없거나 실패하면 고정 점수로 대체하지 않고 unavailable을 반환한다.
"""

import os
from dataclasses import dataclass
from enum import Enum


class SentimentStatus(str, Enum):
    """외부 감성 분석 결과의 사용 가능 상태."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """점수 조작 없이 외부 감성 결과 또는 unavailable 사유를 표현한다."""

    status: SentimentStatus
    score: float | None = None
    magnitude: float | None = None
    unavailable_reason: str | None = None


class GoogleCloudSentimentAdapter:
    """Google Cloud Natural Language의 한국어 감성 분석 adapter."""

    def analyze(self, content: str) -> SentimentResult:
        """감성 결과를 반환하며, 사용할 수 없으면 그 사유를 명시한다."""

        if not content.strip():
            return SentimentResult(
                status=SentimentStatus.UNAVAILABLE,
                unavailable_reason="empty_content",
            )

        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            return SentimentResult(
                status=SentimentStatus.UNAVAILABLE,
                unavailable_reason="credentials_not_configured",
            )

        try:
            from google.cloud import language_v1
        except ImportError:
            return SentimentResult(
                status=SentimentStatus.UNAVAILABLE,
                unavailable_reason="dependency_not_installed",
            )

        try:
            client = language_v1.LanguageServiceClient()
            document = language_v1.Document(
                content=content,
                type_=language_v1.Document.Type.PLAIN_TEXT,
                language="ko",
            )
            response = client.analyze_sentiment(
                request={
                    "document": document,
                    "encoding_type": language_v1.EncodingType.UTF8,
                }
            )
        except Exception as exc:
            return SentimentResult(
                status=SentimentStatus.UNAVAILABLE,
                unavailable_reason=f"service_error:{type(exc).__name__}",
            )

        sentiment = response.document_sentiment
        return SentimentResult(
            status=SentimentStatus.AVAILABLE,
            score=float(sentiment.score),
            magnitude=float(sentiment.magnitude),
        )
