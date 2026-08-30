"""ai-review-crawler가 제공하는 리뷰 JSON을 그대로 받는 요청 schema 모듈.

역할:
- crawler의 `Review` 표준 스키마를 필드 그대로 표현한다.
- 이미 수집된 payload 요청과 수집 서비스 호출 요청을 함께 정의한다.
- 단발 수집 요청과 스트림 수집 요청이 같은 필드 규약을 공유하도록 base model로 묶는다.

수정 범위:
- [INTEGRATION]
- crawler 계약(`review_crawler.core.models.Review`)이 바뀌면 이 파일을 함께 갱신한다.
- analyzer 필드 의미·타입 변경은 AI 담당자와 협의한다.

주의:
- crawler가 제공하지 않는 값(구매 인증, 계정 생성일 등)을 이 계층에서 만들지 않는다.
- crawler 패키지를 import하지 않는다. 필드 계약만 복제해 저장소 간 결합을 만들지 않는다.
- 작업 식별자 필드명은 두 서버 사이에서 `job_id`로 통일한다. `task_id` 등 다른 이름을 쓰지 않는다.

`extra="ignore"` vs `extra="forbid"` 구분 기준:
- **응답성 모델(크롤러 서버가 우리에게 주는 것)은 `ignore`.** 크롤러 서버가 필드를 추가해도
  이 서버가 깨지지 않아야 한다. 두 저장소는 따로 배포되므로 크롤러가 먼저 나가는 상황을
  정상으로 본다. 예: `CrawlerReview`.
- **요청 모델(이 서버가 밖에서 받는 것)은 `forbid`.** 호출 측 오타(`platfrom`, `productId`)를
  조용히 무시하면 기본값으로 엉뚱한 상품을 분석하게 된다. 422로 즉시 알려주는 편이 낫다.
  예: `CrawlerAnalysisRequest`, `CollectAnalysisRequest`, `CollectStreamRequest`.
- 새 모델을 추가할 때 "이 값을 우리가 읽는가(ignore), 남이 우리에게 시키는가(forbid)"로 판단한다.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CrawlerReview(BaseModel):
    """crawler `Review` 한 건의 원본 표현."""

    # crawler가 필드를 추가해도 요청이 깨지지 않도록 미지의 키는 무시한다.
    model_config = ConfigDict(extra="ignore")

    platform: str = Field(description="플랫폼 식별자 (collector 폴더명과 동일)")
    product_id: str = Field(description="이 리뷰가 달린 상품의 platform 내 ID")
    review_id: str = Field(description="platform 내 원본 리뷰 ID")
    content: str

    rating: float | None = Field(default=None, description="5점 만점 환산")
    author: str | None = None
    written_at: datetime | None = None
    option: str | None = Field(default=None, description="구매 옵션 (색상/사이즈 등)")
    images: list[str] = Field(default_factory=list)
    helpful_count: int | None = Field(default=None, description="도움돼요 수")
    collected_at: datetime | None = None


class CrawlerAnalysisRequest(BaseModel):
    """이미 수집된 crawler 리뷰 묶음에 대한 분석 요청."""

    model_config = ConfigDict(extra="forbid")

    reviews: list[CrawlerReview] = Field(
        min_length=1,
        description="같은 (platform, product_id)에 속한 crawler 리뷰 목록",
    )
    product_key: str | None = Field(
        default=None,
        description=(
            "분석 단위를 묶는 상품 키. 생략하면 리뷰의 (platform, product_id)에서 도출한다. "
            "여러 플랫폼 리뷰를 한 상품으로 묶는 경우 호출 측이 직접 지정한다."
        ),
    )


class CollectRequestBase(BaseModel):
    """수집 요청이 공유하는 필드 규약.

    단발 수집(`CollectAnalysisRequest`)과 스트림 수집(`CollectStreamRequest`)이
    "무엇을 얼마나 수집할지"를 같은 이름·같은 제약으로 표현하게 만든다. 한쪽에만
    필드가 늘어나 두 경로의 요청 형태가 갈라지는 일을 막는 것이 이 base의 목적이다.

    직접 인스턴스로 쓰지 않는다. 라우터는 항상 하위 클래스를 받는다.
    """

    model_config = ConfigDict(extra="forbid")

    platform: str = Field(min_length=1, description="수집 서비스에 등록된 플랫폼 식별자")
    product_id: str = Field(min_length=1, description="해당 플랫폼 내 원본 상품 ID")
    limit: int = Field(default=50, ge=1, le=500, description="수집할 최대 리뷰 수")
    product_key: str | None = Field(
        default=None,
        description="분석 단위를 묶는 상품 키. 생략하면 (platform, product_id)에서 도출한다.",
    )


class CollectAnalysisRequest(CollectRequestBase):
    """수집 서비스에서 리뷰를 가져와 분석하는 요청.

    수집이 끝날 때까지 응답을 붙잡는 단발 경로다. 중간 진행 상황이 필요하면
    `CollectStreamRequest`를 쓴다.
    """


class CollectStreamRequest(CollectRequestBase):
    """수집 서비스에서 리뷰를 SSE로 흘려받는 요청.

    크롤러 서버는 이 요청 하나에 `review`/`progress`/`done`/`error`/`heartbeat`
    이벤트를 흘려보낸다. 이벤트 모델 자체는 이 파일이 아니라 stream schema가 정의한다.
    """

    job_id: str | None = Field(
        default=None,
        description=(
            "이 수집 작업의 식별자. 생략하면 크롤러 서버가 발급한 값을 응답에서 받는다. "
            "호출 측이 재연결·중복 요청을 스스로 구분해야 할 때만 직접 지정한다."
        ),
    )


class PlatformsResponse(BaseModel):
    """수집 서비스가 제공하는 플랫폼 현황."""

    available: list[str] = Field(description="수집 요청을 보낼 수 있는 플랫폼")
    failed: list[str] = Field(
        default_factory=list,
        description="수집 서비스에서 로드에 실패한 플랫폼 (의존성 누락 등)",
    )
