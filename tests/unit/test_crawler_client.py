"""수집 서비스 HTTP gateway 테스트.

동기·async gateway, `app.config` 설정 로딩, 수집 요청 스키마 통일을 함께 덮는다.
네트워크를 실제로 타지 않는다. 모든 HTTP 호출은 `httpx.MockTransport`로 가로챈다.
"""

import httpx
import pytest
from pydantic import ValidationError

from app.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_STREAM_TIMEOUT,
    DEFAULT_TIMEOUT,
    CrawlerSettings,
    load_crawler_settings,
)
from app.integrations.crawler_client import (
    AsyncHttpCrawlerGateway,
    CrawlerRequestError,
    CrawlerUnavailableError,
    HttpCrawlerGateway,
)
from app.schemas.crawler import (
    CollectAnalysisRequest,
    CollectRequestBase,
    CollectStreamRequest,
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


@pytest.fixture
def patch_httpx(monkeypatch):
    created: list[httpx.Client] = []

    def apply(handler):
        transport = httpx.MockTransport(handler)
        original = httpx.Client

        def factory(*args: object, **kwargs: object) -> httpx.Client:
            kwargs["transport"] = transport
            client = original(*args, **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr("app.integrations.crawler_client.httpx.Client", factory)
        return HttpCrawlerGateway(BASE_URL)

    # 만들어진 client 수를 세어 연결 재사용을 검증한다.
    apply.created = created
    return apply


@pytest.fixture
def patch_async_httpx(monkeypatch):
    created: list[httpx.AsyncClient] = []

    def apply(handler):
        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            client = original(*args, **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(
            "app.integrations.crawler_client.httpx.AsyncClient", factory
        )
        return AsyncHttpCrawlerGateway(BASE_URL)

    apply.created = created
    return apply


def test_fetch_reviews_parses_crawler_payload(patch_httpx) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=[REVIEW_PAYLOAD])

    gateway = patch_httpx(handler)
    reviews = gateway.fetch_reviews("elevenst", "1831255717", limit=50)

    assert len(reviews) == 1
    assert reviews[0].review_id == "545961223"
    assert reviews[0].author == "가나다라01"
    assert captured["url"] == (
        "http://crawler.test/elevenst/products/1831255717/reviews?limit=50"
    )


def test_list_platforms_reports_available_and_failed(patch_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/platforms"
        return httpx.Response(
            200,
            json={"available": ["elevenst", "oliveyoung"], "failed": ["gmarket"]},
        )

    gateway = patch_httpx(handler)
    available, failed = gateway.list_platforms()

    assert available == ("elevenst", "oliveyoung")
    assert failed == ("gmarket",)


def test_crawler_error_detail_is_preserved(patch_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "'nope' collector 가 없습니다."})

    gateway = patch_httpx(handler)
    with pytest.raises(CrawlerRequestError) as exc_info:
        gateway.fetch_reviews("nope", "1", limit=10)

    assert exc_info.value.status_code == 404
    assert "collector 가 없습니다" in str(exc_info.value)


def test_connection_failure_becomes_unavailable(patch_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    gateway = patch_httpx(handler)
    with pytest.raises(CrawlerUnavailableError, match="연결하지 못했습니다"):
        gateway.list_platforms()


def test_timeout_becomes_unavailable(patch_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    gateway = patch_httpx(handler)
    with pytest.raises(CrawlerUnavailableError, match="오지 않았습니다"):
        gateway.list_platforms()


def test_unexpected_payload_shape_is_rejected(patch_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    gateway = patch_httpx(handler)
    with pytest.raises(CrawlerRequestError, match="리뷰 목록이 아닌"):
        gateway.fetch_reviews("elevenst", "1", limit=10)


def test_base_url_comes_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CRAWLER_BASE_URL", "http://crawler.internal:9000/")

    assert HttpCrawlerGateway().base_url == "http://crawler.internal:9000"


def test_timeout_falls_back_on_invalid_environment(monkeypatch) -> None:
    monkeypatch.setenv("CRAWLER_TIMEOUT", "not-a-number")

    assert HttpCrawlerGateway().timeout == 120.0


# --- HTTP client 수명 ---------------------------------------------------------


def test_client_is_reused_across_calls(patch_httpx) -> None:
    """호출마다 새 연결을 맺지 않는다."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[REVIEW_PAYLOAD])

    gateway = patch_httpx(handler)
    gateway.fetch_reviews("elevenst", "1", limit=10)
    gateway.fetch_reviews("elevenst", "2", limit=10)

    assert len(patch_httpx.created) == 1


def test_close_releases_the_owned_client(patch_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[REVIEW_PAYLOAD])

    gateway = patch_httpx(handler)
    gateway.fetch_reviews("elevenst", "1", limit=10)
    gateway.close()

    assert patch_httpx.created[0].is_closed


def test_context_manager_closes_the_client(patch_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[REVIEW_PAYLOAD])

    with patch_httpx(handler) as gateway:
        gateway.fetch_reviews("elevenst", "1", limit=10)

    assert patch_httpx.created[0].is_closed


def test_injected_client_is_not_closed_by_the_gateway() -> None:
    """밖에서 준 client는 만든 쪽이 닫는다."""

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=[REVIEW_PAYLOAD])
    )
    client = httpx.Client(base_url=BASE_URL, transport=transport)
    gateway = HttpCrawlerGateway(BASE_URL, client=client)

    reviews = gateway.fetch_reviews("elevenst", "1", limit=10)
    gateway.close()

    assert len(reviews) == 1
    assert not client.is_closed
    client.close()


# --- async gateway ------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_fetch_reviews_parses_crawler_payload(patch_async_httpx) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=[REVIEW_PAYLOAD])

    gateway = patch_async_httpx(handler)
    async with gateway:
        reviews = await gateway.fetch_reviews("elevenst", "1831255717", limit=50)

    assert len(reviews) == 1
    assert reviews[0].review_id == "545961223"
    assert captured["url"] == (
        "http://crawler.test/elevenst/products/1831255717/reviews?limit=50"
    )
    assert patch_async_httpx.created[0].is_closed


@pytest.mark.asyncio
async def test_async_list_platforms_reports_available_and_failed(
    patch_async_httpx,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/platforms"
        return httpx.Response(
            200, json={"available": ["elevenst"], "failed": ["gmarket"]}
        )

    gateway = patch_async_httpx(handler)
    available, failed = await gateway.list_platforms()
    await gateway.aclose()

    assert available == ("elevenst",)
    assert failed == ("gmarket",)


@pytest.mark.asyncio
async def test_async_client_is_reused_across_calls(patch_async_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[REVIEW_PAYLOAD])

    gateway = patch_async_httpx(handler)
    await gateway.fetch_reviews("elevenst", "1", limit=10)
    await gateway.fetch_reviews("elevenst", "2", limit=10)
    await gateway.aclose()

    assert len(patch_async_httpx.created) == 1


@pytest.mark.asyncio
async def test_async_crawler_error_detail_is_preserved(patch_async_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "수집기가 응답하지 않습니다."})

    gateway = patch_async_httpx(handler)
    with pytest.raises(CrawlerRequestError) as exc_info:
        await gateway.fetch_reviews("elevenst", "1", limit=10)
    await gateway.aclose()

    assert exc_info.value.status_code == 503
    assert "응답하지 않습니다" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_connection_failure_becomes_unavailable(patch_async_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    gateway = patch_async_httpx(handler)
    with pytest.raises(CrawlerUnavailableError, match="연결하지 못했습니다"):
        await gateway.list_platforms()
    await gateway.aclose()


@pytest.mark.asyncio
async def test_async_timeout_becomes_unavailable(patch_async_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    gateway = patch_async_httpx(handler)
    with pytest.raises(CrawlerUnavailableError, match="오지 않았습니다"):
        await gateway.list_platforms()
    await gateway.aclose()


@pytest.mark.asyncio
async def test_async_unexpected_payload_shape_is_rejected(patch_async_httpx) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    gateway = patch_async_httpx(handler)
    with pytest.raises(CrawlerRequestError, match="리뷰 목록이 아닌"):
        await gateway.fetch_reviews("elevenst", "1", limit=10)
    await gateway.aclose()


@pytest.mark.asyncio
async def test_async_injected_client_is_not_closed_by_the_gateway() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"available": [], "failed": []})
    )
    client = httpx.AsyncClient(base_url=BASE_URL, transport=transport)
    gateway = AsyncHttpCrawlerGateway(BASE_URL, client=client)

    await gateway.list_platforms()
    await gateway.aclose()

    assert not client.is_closed
    await client.aclose()


def test_async_gateway_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("CRAWLER_BASE_URL", "http://crawler.internal:9000/")
    monkeypatch.setenv("CRAWLER_TIMEOUT", "5")

    gateway = AsyncHttpCrawlerGateway()

    assert gateway.base_url == "http://crawler.internal:9000"
    assert gateway.timeout == 5.0


# --- 설정 로딩 ----------------------------------------------------------------


def test_settings_defaults_are_used_without_environment() -> None:
    settings = load_crawler_settings(env={})

    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.timeout == DEFAULT_TIMEOUT
    assert settings.stream_timeout == DEFAULT_STREAM_TIMEOUT
    assert settings.max_retries == DEFAULT_MAX_RETRIES


def test_settings_read_every_supported_environment_variable() -> None:
    settings = load_crawler_settings(
        env={
            "CRAWLER_BASE_URL": "http://crawler.internal:9000/",
            "CRAWLER_TIMEOUT": "30",
            "CRAWLER_STREAM_TIMEOUT": "900",
            "CRAWLER_MAX_RETRIES": "4",
        }
    )

    assert settings.base_url == "http://crawler.internal:9000"
    assert settings.timeout == 30.0
    assert settings.stream_timeout == 900.0
    assert settings.max_retries == 4


@pytest.mark.parametrize(
    ("env", "field", "expected"),
    [
        ({"CRAWLER_TIMEOUT": "not-a-number"}, "timeout", DEFAULT_TIMEOUT),
        ({"CRAWLER_TIMEOUT": "0"}, "timeout", DEFAULT_TIMEOUT),
        ({"CRAWLER_TIMEOUT": "-5"}, "timeout", DEFAULT_TIMEOUT),
        (
            {"CRAWLER_STREAM_TIMEOUT": "nope"},
            "stream_timeout",
            DEFAULT_STREAM_TIMEOUT,
        ),
        ({"CRAWLER_MAX_RETRIES": "2.5"}, "max_retries", DEFAULT_MAX_RETRIES),
        ({"CRAWLER_MAX_RETRIES": "-1"}, "max_retries", DEFAULT_MAX_RETRIES),
        ({"CRAWLER_BASE_URL": "   "}, "base_url", DEFAULT_BASE_URL),
    ],
)
def test_invalid_environment_values_fall_back_to_defaults(
    env: dict, field: str, expected: object, caplog
) -> None:
    """잘못된 설정 값은 예외 대신 기본값 + WARNING 로그로 처리한다 (app/config.py docstring 참고)."""

    with caplog.at_level("WARNING", logger="app.config"):
        settings = load_crawler_settings(env=env)

    assert getattr(settings, field) == expected
    # 조용히 넘기지 않는다. 운영자가 로그에서 잘못된 값을 알아볼 수 있어야 한다.
    assert caplog.records


def test_settings_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        CrawlerSettings(base_url=BASE_URL, timeoutt=1.0)


def test_settings_reject_non_positive_timeout() -> None:
    with pytest.raises(ValidationError):
        CrawlerSettings(timeout=0)


def test_gateway_accepts_injected_settings() -> None:
    settings = CrawlerSettings(
        base_url="http://injected.test/", timeout=7.0, stream_timeout=70.0
    )
    gateway = HttpCrawlerGateway(settings=settings)

    assert gateway.base_url == "http://injected.test"
    assert gateway.timeout == 7.0
    # 스트림 timeout은 gateway가 쓰지 않지만, 스트림 호출 측이 같은 설정에서 읽어 간다.
    assert gateway.settings.stream_timeout == 70.0


def test_explicit_arguments_win_over_settings() -> None:
    settings = CrawlerSettings(base_url="http://injected.test", timeout=7.0)
    gateway = HttpCrawlerGateway(BASE_URL, timeout=1.5, settings=settings)

    assert gateway.base_url == BASE_URL
    assert gateway.timeout == 1.5


# --- 수집 요청 스키마 통일 ------------------------------------------------------


def test_collect_requests_share_the_same_field_contract() -> None:
    """단발 수집과 스트림 수집이 같은 필드 규약을 쓴다."""

    shared = {"platform", "product_id", "limit", "product_key"}

    assert issubclass(CollectAnalysisRequest, CollectRequestBase)
    assert issubclass(CollectStreamRequest, CollectRequestBase)
    assert set(CollectAnalysisRequest.model_fields) == shared
    assert set(CollectStreamRequest.model_fields) == shared | {"job_id"}


def test_collect_stream_request_defaults_job_id_to_none() -> None:
    request = CollectStreamRequest(platform="elevenst", product_id="1831255717")

    assert request.job_id is None
    assert request.limit == 50


def test_collect_stream_request_keeps_caller_supplied_job_id() -> None:
    request = CollectStreamRequest(
        platform="elevenst", product_id="1831255717", job_id="job-1"
    )

    assert request.job_id == "job-1"


@pytest.mark.parametrize(
    "model", [CollectAnalysisRequest, CollectStreamRequest]
)
def test_collect_requests_reject_unknown_fields(model) -> None:
    """요청 모델은 forbid다. 호출 측 오타를 조용히 무시하지 않는다."""

    with pytest.raises(ValidationError):
        model(platform="elevenst", product_id="1", productKey="x")


@pytest.mark.parametrize(
    "model", [CollectAnalysisRequest, CollectStreamRequest]
)
def test_collect_requests_share_the_same_limit_bounds(model) -> None:
    with pytest.raises(ValidationError):
        model(platform="elevenst", product_id="1", limit=0)
    with pytest.raises(ValidationError):
        model(platform="elevenst", product_id="1", limit=501)
