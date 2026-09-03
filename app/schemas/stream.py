"""수집 서비스가 SSE로 흘려보내는 이벤트의 schema 모듈.

역할:
- 크롤러 서버가 `text/event-stream`으로 보내는 5개 이벤트(`review`, `progress`,
  `done`, `error`, `heartbeat`)의 이름 상수와 payload 모델을 정의한다.
- 이벤트 이름을 payload 모델로 잇는 표를 제공해, 수신 계층이 이름만 보고 검증하게 한다.

수정 범위:
- [INTEGRATION]
- 크롤러 서버의 SSE 이벤트 계약이 바뀌면 이 파일을 함께 갱신한다.
- 전송 방식(httpx 호출, 프레이밍 파싱, 재연결)은 여기가 아니라
  `app.integrations.crawler_stream`에서 다룬다.

주의:
- `review` 이벤트의 payload는 `app.schemas.crawler.CrawlerReview`를 상속해 재사용한다.
  필드를 이 파일에 복제하지 않는다. 크롤러 리뷰 계약은 한 곳에만 있어야 한다.
- 미지의 event 이름 처리 방침: **죽이지 않고 `UnknownEvent`로 감싸 원문을 보존한다.**
  크롤러 서버가 이벤트를 먼저 추가하고 AI 서버가 나중에 따라가는 배포 순서를 허용해야 하므로,
  모르는 이름은 오류가 아니라 "아직 해석하지 않는 값"으로 본다. 소비 측은 이를 무시하거나
  로그로 남기면 된다. 반대로 **아는 이름인데 payload가 계약과 다르면 조용히 넘기지 않는다.**
  그건 배포 순서 문제가 아니라 계약 위반이고, 수신 계층이 오류로 올린다.
- 공통 식별자 필드명은 `job_id`로 통일한다. 다른 이름을 새로 만들지 않는다.
"""

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.crawler import CrawlerReview


class StreamEventName(StrEnum):
    """크롤러 SSE가 보내는 event 이름. 이 5가지가 계약의 전부다."""

    REVIEW = "review"
    PROGRESS = "progress"
    DONE = "done"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class ReviewEvent(CrawlerReview):
    """`review` 이벤트. data는 `CrawlerReview` JSON object 한 건 그대로다.

    `CrawlerReview`를 상속하므로 adapter(`app.integrations.crawler.to_analysis_inputs`)에
    그대로 넘길 수 있다.
    """


class ProgressEvent(BaseModel):
    """`progress` 이벤트. 수집 진행 상황을 알린다."""

    # 크롤러가 진행 필드를 추가해도 수신이 깨지지 않도록 미지의 키는 무시한다.
    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(description="수집 작업 식별자")
    collected: int = Field(ge=0, description="지금까지 흘려보낸 리뷰 수")
    target: int | None = Field(
        default=None,
        description="목표 리뷰 수. 크롤러가 총량을 모르면 null이며, 0으로 보정하지 않는다.",
    )


class DoneEvent(BaseModel):
    """`done` 이벤트. 이 이벤트가 스트림의 정상 종료 신호다."""

    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(description="수집 작업 식별자")
    collected: int = Field(ge=0, description="최종적으로 흘려보낸 리뷰 수")


class ErrorEvent(BaseModel):
    """`error` 이벤트. 크롤러가 수집 실패를 보고한다."""

    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(description="수집 작업 식별자")
    detail: str = Field(description="크롤러가 밝힌 실패 사유")
    retryable: bool = Field(description="같은 요청을 다시 시도해볼 만한 실패인지 여부")


class HeartbeatEvent(BaseModel):
    """`heartbeat` 이벤트. data는 빈 object이며 연결 유지 외의 의미가 없다."""

    model_config = ConfigDict(extra="ignore")


class UnknownEvent(BaseModel):
    """계약에 없는 event 이름. 해석하지 않고 원문을 그대로 보존한다."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="크롤러가 보낸 event 이름")
    data: str = Field(description="파싱하지 않은 data 원문")


CrawlerStreamEvent = (
    ReviewEvent | ProgressEvent | DoneEvent | ErrorEvent | HeartbeatEvent | UnknownEvent
)
"""수신 계층이 소비 측에 넘기는 이벤트 union."""


EVENT_PAYLOAD_MODELS: Mapping[str, type[BaseModel]] = {
    StreamEventName.REVIEW: ReviewEvent,
    StreamEventName.PROGRESS: ProgressEvent,
    StreamEventName.DONE: DoneEvent,
    StreamEventName.ERROR: ErrorEvent,
    StreamEventName.HEARTBEAT: HeartbeatEvent,
}
"""event 이름 → payload 모델. 표에 없는 이름이 미지의 이벤트다."""


def event_payload_model(name: str) -> type[BaseModel] | None:
    """event 이름에 대응하는 payload 모델을 준다. 모르는 이름이면 None."""

    return EVENT_PAYLOAD_MODELS.get(name)
