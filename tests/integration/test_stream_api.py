"""수집 스트림 분석 API 통합 테스트.

크롤러 서버는 httpx MockTransport로 흉내낸다. 실제 네트워크로 나가지 않는다.
검증 대상은 "크롤러 SSE → 이 서버 SSE" 번역이다: 이벤트 이름, 순서, 그리고 실패를
HTTP 상태로 알릴지 `error` 이벤트로 알릴지의 경계.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.analysis import get_crawler_stream_client
from app.integrations.crawler_stream import CrawlerStreamClient, iter_sse_frames
from app.main import app

BASE_URL = "http://crawler.test"

LONG_CONTENT = "묵직한데 깔끔하고 빨대까지 포함되어 있어 좋았습니다. 손잡이 분리로 세척도 편합니다."


def crawler_review(review_id: str, content: str = LONG_CONTENT) -> dict:
    return {
        "platform": "elevenst",
        "product_id": "1831255717",
        "review_id": review_id,
        "content": content,
    }


def sse_block(event: str, data: object) -> str:
    """크롤러가 보내는 이벤트 하나를 SSE 텍스트로 만든다."""

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def crawler_response(body: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        content=body.encode(),
    )


def sse_events(response) -> list[tuple[str, dict]]:
    """이 서버가 내보낸 SSE 본문을 (event 이름, data) 목록으로 만든다."""

    return [
        (frame.event, json.loads(frame.data))
        for frame in iter_sse_frames(response.text.split("\n"))
    ]


@pytest.fixture
def client():
    """크롤러를 흉내내는 MockTransport를 물린 테스트 client를 만든다."""

    def build(handler, **kwargs) -> TestClient:
        kwargs.setdefault("max_reconnects", 0)
        stream_client = CrawlerStreamClient(
            BASE_URL,
            client=httpx.AsyncClient(
                base_url=BASE_URL, transport=httpx.MockTransport(handler)
            ),
            **kwargs,
        )
        app.dependency_overrides[get_crawler_stream_client] = lambda: stream_client
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


def post_stream(test_client: TestClient, **overrides: object):
    payload: dict = {"platform": "elevenst", "product_id": "1831255717"}
    payload.update(overrides)
    return test_client.post("/analysis/collect/stream", json=payload)


def test_stream_reports_progress_then_final_result(client) -> None:
    body = (
        sse_block("heartbeat", {})
        + sse_block("review", crawler_review("1"))
        + sse_block("progress", {"job_id": "j1", "collected": 1, "target": 2})
        + sse_block("review", crawler_review("2"))
        + sse_block("done", {"job_id": "j1", "collected": 2})
    )
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return crawler_response(body)

    response = post_stream(client(handler), limit=30)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert captured["url"] == (
        "http://crawler.test/elevenst/products/1831255717/reviews/stream?limit=30"
    )

    events = sse_events(response)
    assert [name for name, _ in events] == [
        "heartbeat",
        "progress",
        "progress",
        "progress",
        "result",
    ]

    # 첫 리뷰 시점에는 크롤러가 job_id도 target도 알려준 적이 없다. 지어내지 않는다.
    assert events[1][1] == {"job_id": None, "collected": 1, "target": None}
    # 크롤러 progress를 받은 뒤에는 그 값을 반영한다. collected는 우리가 실제로 받은 수다.
    assert events[2][1] == {"job_id": "j1", "collected": 1, "target": 2}
    assert events[3][1] == {"job_id": "j1", "collected": 2, "target": 2}

    result = events[4][1]
    assert result["product_key"] == "elevenst:1831255717"
    assert result["review_count"] == 2
    assert [item["analysis_review_id"] for item in result["results"]] == [
        "elevenst:1",
        "elevenst:2",
    ]
    assert result["results"][0]["level"] in {"safe", "warn", "danger"}


def test_stream_passes_job_id_from_request_until_crawler_reports_one(client) -> None:
    body = sse_block("review", crawler_review("1")) + sse_block(
        "done", {"job_id": "crawler-job", "collected": 1}
    )

    response = post_stream(client(lambda request: crawler_response(body)), job_id="mine")

    events = sse_events(response)
    assert events[0] == ("progress", {"job_id": "mine", "collected": 1, "target": None})


def test_stream_surfaces_crawler_error_event_as_error_event(client) -> None:
    body = sse_block("review", crawler_review("1")) + sse_block(
        "error",
        {"job_id": "j1", "detail": "'nope' collector 가 없습니다.", "retryable": False},
    )

    response = post_stream(client(lambda request: crawler_response(body)))

    # 헤더는 이미 200으로 나갔으므로 실패는 이벤트로 알린다.
    assert response.status_code == 200
    events = sse_events(response)
    assert [name for name, _ in events] == ["progress", "error"]
    assert events[1][1]["status"] == 502
    assert "collector 가 없습니다" in events[1][1]["detail"]
    # 실패했으므로 result는 나가지 않는다.
    assert "result" not in {name for name, _ in events}


def test_stream_maps_retryable_crawler_error_to_gateway_timeout_event(client) -> None:
    body = sse_block("review", crawler_review("1")) + sse_block(
        "error", {"job_id": "j1", "detail": "플랫폼 응답이 느립니다.", "retryable": True}
    )

    response = post_stream(client(lambda request: crawler_response(body)))

    events = sse_events(response)
    assert events[-1][0] == "error"
    assert events[-1][1]["status"] == 504


def test_stream_does_not_report_partial_collection_as_success(client) -> None:
    # done 없이 끊긴 스트림을 받은 리뷰만으로 성공 처리하지 않는다.
    body = sse_block("review", crawler_review("1"))

    response = post_stream(client(lambda request: crawler_response(body)))

    events = sse_events(response)
    assert [name for name, _ in events] == ["progress", "error"]
    assert events[1][1]["status"] == 504
    assert "done 이벤트 없이" in events[1][1]["detail"]


def test_stream_reports_empty_collection_as_error_event(client) -> None:
    body = sse_block("done", {"job_id": "j1", "collected": 0})

    response = post_stream(client(lambda request: crawler_response(body)))

    assert response.status_code == 200
    events = sse_events(response)
    assert [name for name, _ in events] == ["error"]
    assert events[0][1]["status"] == 404
    assert "수집된 리뷰가 없습니다" in events[0][1]["detail"]


def test_stream_maps_unreachable_crawler_to_gateway_timeout(client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    response = post_stream(client(handler))

    # 첫 이벤트 전에 실패했으므로 아직 아무 바이트도 보내지 않았다. HTTP 상태로 알린다.
    assert response.status_code == 504
    assert "연결이 끊겼습니다" in response.json()["detail"]


def test_stream_reports_missing_crawler_endpoint_as_not_found(client) -> None:
    # 크롤러 서버에 아직 SSE 경로가 없는 현재 상태가 그대로 드러나야 한다.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    response = post_stream(client(handler))

    assert response.status_code == 404
    assert "Not Found" in response.json()["detail"]


def test_stream_rejects_invalid_limit(client) -> None:
    response = post_stream(client(lambda request: crawler_response("")), limit=0)

    assert response.status_code == 422


def test_stream_ignores_unknown_crawler_events(client) -> None:
    body = (
        sse_block("rate_limited", {"wait": 5})
        + sse_block("review", crawler_review("1"))
        + sse_block("done", {"job_id": "j1", "collected": 1})
    )

    response = post_stream(client(lambda request: crawler_response(body)))

    # 계약에 없는 이름은 하류로 전파하지 않는다. 대신 스트림을 죽이지도 않는다.
    assert [name for name, _ in sse_events(response)] == ["progress", "result"]
