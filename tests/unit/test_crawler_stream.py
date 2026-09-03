"""수집 서비스 SSE 수신 계층 테스트.

프레이밍 파서는 순수 함수라 라인 목록만으로 검증하고, 스트림 소비는 httpx MockTransport로
검증한다. 실제 네트워크는 쓰지 않는다.

pytest-asyncio를 쓰지 않는다. 새 의존성 없이 `asyncio.run`으로 async 이터레이터를 돌린다.
"""

import asyncio
import json

import httpx
import pytest

from app.integrations.crawler_client import CrawlerRequestError, CrawlerUnavailableError
from app.integrations.crawler_stream import (
    CrawlerStreamClient,
    decode_sse_frame,
    iter_sse_frames,
)
from app.schemas.stream import (
    DoneEvent,
    HeartbeatEvent,
    ProgressEvent,
    ReviewEvent,
    UnknownEvent,
)

BASE_URL = "http://crawler.test"

REVIEW_PAYLOAD = {
    "platform": "elevenst",
    "product_id": "1831255717",
    "review_id": "545961223",
    "content": "묵직한데 깔끔하고 빨대까지 포함되어 있어 좋아요.",
    "rating": 5.0,
    "author": "가나다라01",
    "written_at": "2026-06-09T00:00:00",
    "option": "색상:모카그레이",
    "images": [],
    "helpful_count": 2,
    "collected_at": "2026-08-04T20:41:18.802538",
}


def sse_block(event: str, data: object, *, event_id: str | None = None) -> str:
    """이벤트 하나를 SSE 텍스트 블록으로 만든다."""

    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    return "\n".join(lines) + "\n\n"


def as_lines(payload: str) -> list[str]:
    """SSE 텍스트를 파서가 받는 라인 목록으로 자른다."""

    return payload.split("\n")


# --- 프레이밍 파서 ---------------------------------------------------------


def test_parses_event_and_data_fields() -> None:
    frames = list(iter_sse_frames(as_lines("event: progress\ndata: {\"a\": 1}\n\n")))

    assert len(frames) == 1
    assert frames[0].event == "progress"
    assert frames[0].data == '{"a": 1}'


def test_multiline_data_is_joined_with_newline() -> None:
    frames = list(iter_sse_frames(["event: review", "data: {", 'data:   "x": 1', "data: }", ""]))

    assert len(frames) == 1
    # 값 앞 공백은 한 칸만 제거하므로 두 번째 줄에는 한 칸이 남는다.
    assert frames[0].data == '{\n  "x": 1\n}'


def test_comment_lines_and_id_retry_fields_are_handled() -> None:
    lines = as_lines(
        ": 프록시 유지용 주석\nid: 42\nretry: 2500\nevent: heartbeat\ndata: {}\n\n"
    )

    frames = list(iter_sse_frames(lines))

    assert len(frames) == 1
    assert frames[0].event == "heartbeat"
    assert frames[0].last_event_id == "42"
    assert frames[0].retry_ms == 2500


def test_blank_line_separates_events_and_resets_event_name() -> None:
    payload = "event: progress\ndata: 1\n\ndata: 2\n\n"

    frames = list(iter_sse_frames(as_lines(payload)))

    assert [(frame.event, frame.data) for frame in frames] == [
        ("progress", "1"),
        # event 필드는 프레임 경계에서 초기화되므로 기본 이름으로 돌아간다.
        ("message", "2"),
    ]


def test_id_persists_across_frames() -> None:
    payload = "id: 7\nevent: review\ndata: 1\n\nevent: review\ndata: 2\n\n"

    frames = list(iter_sse_frames(as_lines(payload)))

    assert [frame.last_event_id for frame in frames] == ["7", "7"]


def test_truncated_frame_is_discarded() -> None:
    # 빈 줄 없이 끊긴 프레임은 완성된 것처럼 올리지 않는다.
    frames = list(iter_sse_frames(["event: review", 'data: {"partial": tr']))

    assert frames == []


def test_frame_without_data_is_not_dispatched() -> None:
    frames = list(iter_sse_frames(["event: heartbeat", "", "event: review", "data: 1", ""]))

    assert [frame.event for frame in frames] == ["review"]


def test_field_without_colon_is_read_as_empty_value() -> None:
    frames = list(iter_sse_frames(["event: review", "data", ""]))

    assert len(frames) == 1
    assert frames[0].data == ""


def test_unknown_event_name_is_preserved_not_raised() -> None:
    frames = list(iter_sse_frames(as_lines('event: rate_limited\ndata: {"wait": 5}\n\n')))
    event = decode_sse_frame(frames[0])

    assert isinstance(event, UnknownEvent)
    assert event.name == "rate_limited"
    assert event.data == '{"wait": 5}'


def test_known_event_with_broken_payload_is_raised() -> None:
    frames = list(iter_sse_frames(["event: progress", "data: not-json", ""]))

    with pytest.raises(CrawlerRequestError, match="JSON이 아닙니다"):
        decode_sse_frame(frames[0])


def test_known_event_violating_contract_is_raised() -> None:
    frames = list(iter_sse_frames(['event: done', 'data: {"job_id": "j1"}', ""]))

    with pytest.raises(CrawlerRequestError, match="계약과 다릅니다"):
        decode_sse_frame(frames[0])


# --- 스트림 소비 -----------------------------------------------------------


def sse_response(body: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        content=body.encode(),
    )


def collect(handler, **kwargs) -> list:
    """MockTransport를 물린 client로 스트림을 끝까지 소비한다. 네트워크로 나가지 않는다."""

    kwargs.setdefault("max_reconnects", 0)

    async def run() -> list:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(base_url=BASE_URL, transport=transport) as http:
            client = CrawlerStreamClient(BASE_URL, client=http, **kwargs)
            return [
                event
                async for event in client.stream_reviews("elevenst", "1", limit=10)
            ]

    return asyncio.run(run())


def test_stream_yields_events_until_done() -> None:
    body = (
        ": 연결 확인\n\n"
        + sse_block("heartbeat", {})
        + sse_block("review", REVIEW_PAYLOAD)
        + sse_block("progress", {"job_id": "j1", "collected": 1, "target": 10})
        + sse_block("unheard_of", {"x": 1})
        + sse_block("done", {"job_id": "j1", "collected": 1})
    )
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["accept"] = request.headers.get("accept")
        return sse_response(body)

    events = collect(handler)

    assert [type(event) for event in events] == [
        HeartbeatEvent,
        ReviewEvent,
        ProgressEvent,
        UnknownEvent,
        DoneEvent,
    ]
    assert events[1].review_id == "545961223"
    assert events[2].target == 10
    assert events[4].collected == 1
    assert captured["accept"] == "text/event-stream"
    assert captured["url"] == (
        "http://crawler.test/elevenst/products/1/reviews/stream?limit=10"
    )


def test_progress_target_stays_none_when_crawler_does_not_know_it() -> None:
    body = sse_block("progress", {"job_id": "j1", "collected": 3, "target": None}) + sse_block(
        "done", {"job_id": "j1", "collected": 3}
    )

    events = collect(lambda request: sse_response(body))

    # 총량을 모른다는 사실을 0으로 보정하지 않는다.
    assert events[0].target is None


def test_error_event_is_raised_not_swallowed() -> None:
    body = (
        sse_block("review", REVIEW_PAYLOAD)
        + sse_block(
            "error",
            {"job_id": "j1", "detail": "'nope' collector 가 없습니다.", "retryable": False},
        )
    )

    with pytest.raises(CrawlerRequestError) as exc_info:
        collect(lambda request: sse_response(body))

    assert exc_info.value.status_code == 502
    assert "collector 가 없습니다" in str(exc_info.value)


def test_retryable_error_event_becomes_unavailable() -> None:
    body = sse_block(
        "error", {"job_id": "j1", "detail": "플랫폼 응답이 느립니다.", "retryable": True}
    )

    with pytest.raises(CrawlerUnavailableError, match="플랫폼 응답이 느립니다"):
        collect(lambda request: sse_response(body))


def test_stream_ending_without_done_is_a_failure() -> None:
    # 중간에 끊긴 스트림을 "지금까지 받은 것"으로 성공 처리하지 않는다.
    body = sse_block("review", REVIEW_PAYLOAD)

    with pytest.raises(CrawlerUnavailableError, match="done 이벤트 없이"):
        collect(lambda request: sse_response(body))


def test_connection_failure_becomes_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(CrawlerUnavailableError, match="연결이 끊겼습니다"):
        collect(handler)


def test_read_timeout_becomes_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(CrawlerUnavailableError, match="보내지 않았습니다"):
        collect(handler)


def test_http_error_detail_is_preserved() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "'nope' collector 가 없습니다."})

    with pytest.raises(CrawlerRequestError) as exc_info:
        collect(handler)

    assert exc_info.value.status_code == 404
    assert "collector 가 없습니다" in str(exc_info.value)


def test_non_sse_content_type_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[REVIEW_PAYLOAD])

    with pytest.raises(CrawlerRequestError, match="SSE가 아닌"):
        collect(handler)


def test_reconnect_resumes_with_last_event_id(monkeypatch) -> None:
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr("app.integrations.crawler_stream.asyncio.sleep", fake_sleep)

    attempts: list[str | None] = []
    first = sse_block("review", REVIEW_PAYLOAD, event_id="1")
    second = sse_block("done", {"job_id": "j1", "collected": 1})

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.headers.get("last-event-id"))
        return sse_response(first if len(attempts) == 1 else second)

    events = collect(handler, max_reconnects=1)

    assert [type(event) for event in events] == [ReviewEvent, DoneEvent]
    # 첫 연결에는 재개 지점이 없고, 재연결에는 마지막으로 본 id를 실어 보낸다.
    assert attempts == [None, "1"]
    assert slept == [1.0]


def test_reconnect_gives_up_at_the_limit(monkeypatch) -> None:
    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("app.integrations.crawler_stream.asyncio.sleep", fake_sleep)

    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("connection refused")

    with pytest.raises(CrawlerUnavailableError, match="재연결 2회 모두 실패"):
        collect(handler, max_reconnects=2)

    # 최초 연결 1회 + 재연결 2회.
    assert len(attempts) == 3


def test_settings_supply_base_url_and_reconnect_limit(monkeypatch) -> None:
    monkeypatch.setenv("CRAWLER_BASE_URL", "http://crawler.internal:9000/")
    monkeypatch.setenv("CRAWLER_MAX_RETRIES", "5")

    client = CrawlerStreamClient()

    assert client.base_url == "http://crawler.internal:9000"
    assert client.max_reconnects == 5
