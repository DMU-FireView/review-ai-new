"""수집 서비스(ai-review-crawler)의 SSE 스트림을 구독하는 gateway 모듈.

역할:
- 크롤러 서버가 수집하는 동안 흘려보내는 리뷰를 끝까지 기다리지 않고 이어서 받는다.
- `text/event-stream` 프레이밍을 직접 파싱해 `app.schemas.stream` 이벤트로 옮긴다.
- 스트림 오류를 이 서비스가 이미 쓰는 예외로 옮긴다.

수정 범위:
- [MODEL / EXTERNAL INTEGRATION]
- 크롤러 서버의 SSE 경로·프레이밍·오류 규약이 바뀌면 이 파일을 수정한다.
- base URL·timeout·재시도 상한은 `app.config`에서 읽는다. 여기서 다시 정의하지 않는다.
- 이벤트 payload 계약은 `app.schemas.stream`에서, analyzer 점수 의미나 scoring 정책은
  이 파일에서 변경하지 않는다.

주의:
- 수집 서비스를 이 저장소의 패키지로 가져오지 않는다. 결합점은 base URL 하나로 유지한다.
- 수집 실패를 빈 리뷰 목록으로 감추지 않는다. `error` 이벤트는 예외로 올린다.
  `done` 없이 끊긴 스트림도 "지금까지 받은 것"으로 성공 처리하지 않는다.
- SSE 라이브러리를 새로 들이지 않는다. 프레이밍은 `SseFrameDecoder`가 직접 다루고,
  이 디코더는 네트워크를 모르는 순수 상태 기계라 단위 테스트로 검증한다.
- 예외 종류는 `crawler_client`의 것을 import해 재사용한다. 같은 사건에 두 벌의 예외를
  두지 않는다.
- 주입받은 `httpx.AsyncClient`는 이 client가 닫지 않는다. 만든 쪽이 닫는다.
  동기 gateway와 같은 수명 규칙이다.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from types import TracebackType

import httpx
from pydantic import ValidationError

from app.config import CrawlerSettings, load_crawler_settings
from app.integrations.crawler_client import CrawlerRequestError, CrawlerUnavailableError
from app.schemas.stream import (
    CrawlerStreamEvent,
    DoneEvent,
    ErrorEvent,
    UnknownEvent,
    event_payload_model,
)

__all__ = [
    "CrawlerStreamClient",
    "SseFrame",
    "SseFrameDecoder",
    "decode_sse_frame",
    "iter_sse_frames",
    "aiter_sse_frames",
    "stream_reviews_path",
]

DEFAULT_CONNECT_TIMEOUT = 10.0
"""연결까지의 상한. 스트림이 얼마나 오래 열려 있든 연결 자체는 금방 맺어져야 한다.

`app.config`에는 이 값에 해당하는 항목이 없다. 스트림 수명과 달리 운영 중에
조절할 이유가 없는 값이라 여기 상수로 둔다.
"""

DEFAULT_RECONNECT_DELAY = 1.0
"""첫 재연결까지의 대기. 서버가 `retry:`를 보내면 그 값을 우선한다."""

MAX_RECONNECT_DELAY = 10.0
"""재연결 대기 상한. 서버가 보낸 `retry:`도 이 값으로 자른다."""

DEFAULT_EVENT_NAME = "message"
"""SSE 표준에서 `event:` 필드가 없는 프레임의 이름."""


def stream_reviews_path(platform: str, product_id: str) -> str:
    """상품 리뷰 SSE 경로를 만든다. 단발 조회 경로에 `/stream`을 붙인 형태다."""

    return f"/{platform}/products/{product_id}/reviews/stream"


@dataclass(frozen=True, slots=True)
class SseFrame:
    """빈 줄로 종료된 SSE 프레임 하나. 아직 payload를 해석하기 전 상태다."""

    event: str
    data: str
    last_event_id: str | None = None
    retry_ms: int | None = None


class SseFrameDecoder:
    r"""SSE 라인을 받아 프레임 경계를 잡는 상태 기계.

    네트워크를 모르고 라인만 다루므로 동기·비동기 소비 양쪽에서 같은 객체를 쓴다.

    프레이밍 규칙(WHATWG SSE):
    - `:`로 시작하는 줄은 주석이라 버린다.
    - `field: value`에서 값 앞의 공백은 한 칸만 제거한다. 두 칸이면 한 칸이 값에 남는다.
    - 콜론이 없는 줄은 필드 이름만 있고 값은 빈 문자열인 줄로 본다.
    - `data`가 여러 줄이면 `\n`으로 잇는다.
    - 빈 줄이 프레임 경계다. 이때 `event`는 초기화되고 `id`/`retry`는 유지된다.
    - **data가 한 줄도 없으면 프레임을 내보내지 않는다.** 표준 그대로다. `heartbeat`는
      계약상 항상 `{}`를 싣고 오므로 이 규칙에 걸리지 않고, 걸리는 프레임은 어차피
      해석할 내용이 없다.
    - 빈 줄 없이 스트림이 끝나면 모아둔 data는 버린다. 잘린 프레임을 완성된 것처럼
      올리지 않는다.
    """

    def __init__(self) -> None:
        self._event: str | None = None
        self._data_lines: list[str] = []
        self._last_event_id: str | None = None
        self._retry_ms: int | None = None

    def feed_line(self, line: str) -> SseFrame | None:
        """라인 하나를 넣는다. 프레임이 끝났으면 그 프레임을, 아니면 None을 준다."""

        line = line.rstrip("\n").rstrip("\r")

        if not line:
            return self._dispatch()

        if line.startswith(":"):
            # 주석 라인. 프록시가 연결을 끊지 않게 하는 용도라 버린다.
            return None

        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]

        match field:
            case "event":
                self._event = value
            case "data":
                self._data_lines.append(value)
            case "id":
                # 표준상 NULL이 든 id는 무시한다.
                if "\x00" not in value:
                    self._last_event_id = value
            case "retry":
                if value.isdigit():
                    self._retry_ms = int(value)
            case _:
                # 모르는 필드는 버린다. 프레임을 깨뜨리지 않는다.
                pass

        return None

    def _dispatch(self) -> SseFrame | None:
        if not self._data_lines:
            self._event = None
            return None

        frame = SseFrame(
            event=self._event or DEFAULT_EVENT_NAME,
            data="\n".join(self._data_lines),
            last_event_id=self._last_event_id,
            retry_ms=self._retry_ms,
        )
        self._event = None
        self._data_lines = []
        return frame


def iter_sse_frames(lines: Iterable[str]) -> Iterator[SseFrame]:
    """라인 이터러블에서 SSE 프레임을 뽑는 순수 제너레이터."""

    decoder = SseFrameDecoder()
    for line in lines:
        frame = decoder.feed_line(line)
        if frame is not None:
            yield frame


async def aiter_sse_frames(lines: AsyncIterator[str]) -> AsyncIterator[SseFrame]:
    """비동기 라인 스트림에서 SSE 프레임을 뽑는다. 규칙은 동기판과 같다."""

    decoder = SseFrameDecoder()
    async for line in lines:
        frame = decoder.feed_line(line)
        if frame is not None:
            yield frame


def decode_sse_frame(frame: SseFrame) -> CrawlerStreamEvent:
    """프레임 payload를 계약 모델로 옮긴다.

    모르는 event 이름은 `UnknownEvent`로 감싼다. 아는 이름인데 payload가 계약과 다르면
    `CrawlerRequestError`로 올린다. 계약 위반을 조용히 넘기지 않는다.
    """

    model = event_payload_model(frame.event)
    if model is None:
        return UnknownEvent(name=frame.event, data=frame.data)

    try:
        payload = json.loads(frame.data)
    except json.JSONDecodeError as exc:
        raise CrawlerRequestError(
            f"수집 서비스의 '{frame.event}' 이벤트 data가 JSON이 아닙니다: {frame.data!r}",
            status_code=502,
        ) from exc

    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise CrawlerRequestError(
            f"수집 서비스의 '{frame.event}' 이벤트가 계약과 다릅니다: {exc}",
            status_code=502,
        ) from exc


class CrawlerStreamClient:
    """크롤러 서버의 SSE 엔드포인트를 구독하는 client.

    `done` 이벤트를 받으면 그 이벤트까지 내보내고 정상 종료한다. `error` 이벤트는
    예외로 올린다. 스트림이 `done` 없이 끊기면 재연결 상한까지 다시 붙어보고,
    그래도 못 끝내면 `CrawlerUnavailableError`로 올린다.

    재연결 상한은 `app.config`의 `max_retries`(`CRAWLER_MAX_RETRIES`)를 그대로 쓴다.
    근거: 재연결은 전송이 끊긴 경우만 구제할 뿐, 크롤러가 실제로 실패한 경우는 구제하지
    못한다. 무한 재시도는 죽은 작업을 살아 있는 것처럼 보이게 하고 크롤러에 부하만 준다.
    상한을 넘기면 실패로 올려 호출 측이 판단하게 한다.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        stream_timeout: float | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        max_reconnects: int | None = None,
        settings: CrawlerSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings if settings is not None else load_crawler_settings()
        resolved = base_url if base_url is not None else self.settings.base_url
        self.base_url = resolved.rstrip("/")
        self.stream_timeout = (
            stream_timeout if stream_timeout is not None else self.settings.stream_timeout
        )
        self.connect_timeout = connect_timeout
        self.max_reconnects = (
            max_reconnects if max_reconnects is not None else self.settings.max_retries
        )
        self._client = client
        # 주입받은 client는 만든 쪽이 닫는다. 직접 만든 client만 이 client가 닫는다.
        self._owns_client = client is None

    def stream_reviews(
        self,
        platform: str,
        product_id: str,
        *,
        limit: int,
    ) -> AsyncIterator[CrawlerStreamEvent]:
        """상품 리뷰 수집 스트림을 구독한다."""

        return self.subscribe(
            stream_reviews_path(platform, product_id),
            params={"limit": limit},
        )

    async def subscribe(
        self,
        path: str,
        *,
        params: dict | None = None,
    ) -> AsyncIterator[CrawlerStreamEvent]:
        """SSE 경로를 구독해 이벤트를 순서대로 내보낸다.

        `heartbeat`와 `UnknownEvent`도 그대로 내보낸다. 소비 측이 무시하면 되고,
        여기서 걸러 버리면 연결이 살아 있다는 사실조차 위로 전해지지 않는다.

        재연결 시 마지막으로 본 `id`를 `Last-Event-ID` 헤더로 보낸다. 크롤러가 이 헤더를
        무시하면 이미 받은 리뷰가 다시 올 수 있으므로, 중복 제거는 소비 측이
        `review_id` 기준으로 한다. 여기서 임의로 버리면 어디까지 받았는지를
        수신 계층이 단독으로 판단하는 셈이 된다.
        """

        attempt = 0
        delay = DEFAULT_RECONNECT_DELAY
        last_event_id: str | None = None

        while True:
            cause: Exception | None = None
            try:
                async for frame in self._iter_frames(path, params, last_event_id):
                    if frame.last_event_id is not None:
                        last_event_id = frame.last_event_id
                    if frame.retry_ms is not None:
                        delay = min(frame.retry_ms / 1000, MAX_RECONNECT_DELAY)

                    event = decode_sse_frame(frame)
                    if isinstance(event, ErrorEvent):
                        raise _error_event_to_exception(event)

                    yield event
                    if isinstance(event, DoneEvent):
                        return
            except httpx.TimeoutException as exc:
                cause = exc
                reason = (
                    f"수집 스트림이 {self.stream_timeout}초 동안 아무 것도 보내지 않았습니다"
                )
            except httpx.HTTPError as exc:
                cause = exc
                reason = f"수집 스트림 연결이 끊겼습니다 ({type(exc).__name__})"
            else:
                reason = "수집 스트림이 done 이벤트 없이 끝났습니다"

            attempt += 1
            if attempt > self.max_reconnects:
                raise CrawlerUnavailableError(
                    f"{reason}: {self.base_url}{path} "
                    f"(재연결 {self.max_reconnects}회 모두 실패)"
                ) from cause

            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_RECONNECT_DELAY)

    async def aclose(self) -> None:
        """이 client가 만든 httpx client를 닫는다. 주입받은 것은 건드리지 않는다."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
        if self._owns_client:
            self._client = None

    async def __aenter__(self) -> "CrawlerStreamClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _ensure_client(self) -> httpx.AsyncClient:
        """client를 처음 쓸 때 만들고, 재연결은 같은 연결 풀을 재사용한다."""

        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url)
        return self._client

    async def _iter_frames(
        self,
        path: str,
        params: dict | None,
        last_event_id: str | None,
    ) -> AsyncIterator[SseFrame]:
        """한 번의 연결에서 나오는 SSE 프레임을 내보낸다."""

        headers = {"Accept": "text/event-stream", "Cache-Control": "no-store"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id

        # httpx의 read timeout은 청크 사이 간격에 걸린다. 총 수집 시간이 아니라
        # 침묵 시간을 재므로, 크롤러가 heartbeat를 보내는 동안에는 계속 갱신된다.
        timeout = httpx.Timeout(self.stream_timeout, connect=self.connect_timeout)

        async with self._ensure_client().stream(
            "GET",
            path,
            params=params,
            headers=headers,
            timeout=timeout,
        ) as response:
            if response.is_error:
                await response.aread()
                raise CrawlerRequestError(
                    _error_detail(response),
                    status_code=response.status_code,
                )

            media_type = response.headers.get("content-type", "").split(";")[0].strip()
            if media_type != "text/event-stream":
                raise CrawlerRequestError(
                    f"수집 서비스가 SSE가 아닌 '{media_type or '알 수 없음'}'을 반환했습니다.",
                    status_code=502,
                )

            async for frame in aiter_sse_frames(response.aiter_lines()):
                yield frame


def _error_event_to_exception(event: ErrorEvent) -> Exception:
    """`error` 이벤트를 기존 예외 체계로 옮긴다.

    `retryable`은 크롤러가 "다시 시도해볼 만하다"고 알려준 값이므로 일시적 장애로 보고
    `CrawlerUnavailableError`에 태운다. 그렇지 않은 실패는 요청 자체가 성립하지 않은
    경우라 `CrawlerRequestError`로 올린다. HTTP 응답 자체는 200이었으므로 상태 코드는
    upstream 오류를 뜻하는 502를 쓴다.
    """

    message = f"수집 작업 {event.job_id}이(가) 실패했습니다: {event.detail}"
    if event.retryable:
        return CrawlerUnavailableError(message)
    return CrawlerRequestError(message, status_code=502)


def _error_detail(response: httpx.Response) -> str:
    """수집 서비스가 내려준 오류 사유를 최대한 그대로 옮긴다."""

    try:
        payload = response.json()
    except ValueError:
        return response.text or f"수집 서비스 오류 (HTTP {response.status_code})"

    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)
