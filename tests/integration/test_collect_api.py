"""수집 서비스를 거치는 분석 API 통합 테스트."""

import pytest
from fastapi.testclient import TestClient

from app.api.analysis import get_crawler_gateway
from app.integrations.crawler_client import CrawlerRequestError, CrawlerUnavailableError
from app.main import app
from app.schemas.crawler import CrawlerReview

LONG_CONTENT = "묵직한데 깔끔하고 빨대까지 포함되어 있어 좋았습니다. 손잡이 분리로 세척도 편합니다."


def crawler_review(review_id: str, content: str = LONG_CONTENT, **overrides: object) -> CrawlerReview:
    payload: dict = {
        "platform": "elevenst",
        "product_id": "1831255717",
        "review_id": review_id,
        "content": content,
    }
    payload.update(overrides)
    return CrawlerReview.model_validate(payload)


class FakeGateway:
    """수집 서비스를 대신하는 테스트용 async gateway.

    라우터가 `AsyncCrawlerGateway`를 쓰므로 fake도 async로 맞춘다. 동기 fake를 두면
    라우터가 동기 gateway로 되돌아가도 테스트가 통과해 버린다.
    """

    def __init__(self, reviews=(), *, error: Exception | None = None) -> None:
        self.reviews = tuple(reviews)
        self.error = error
        self.calls: list[tuple[str, str, int]] = []

    async def list_platforms(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self.error is not None:
            raise self.error
        return ("elevenst", "oliveyoung"), ("gmarket",)

    async def fetch_reviews(self, platform: str, product_id: str, *, limit: int):
        self.calls.append((platform, product_id, limit))
        if self.error is not None:
            raise self.error
        return self.reviews


@pytest.fixture
def client():
    """gateway를 교체할 수 있는 테스트 client를 만든다."""

    def build(gateway: FakeGateway) -> TestClient:
        app.dependency_overrides[get_crawler_gateway] = lambda: gateway
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


def test_collect_fetches_then_analyzes(client) -> None:
    gateway = FakeGateway([crawler_review("1"), crawler_review("2")])

    response = client(gateway).post(
        "/analysis/collect",
        json={"platform": "elevenst", "product_id": "1831255717", "limit": 30},
    )

    assert response.status_code == 200
    assert gateway.calls == [("elevenst", "1831255717", 30)]

    body = response.json()
    assert body["product_key"] == "elevenst:1831255717"
    assert body["review_count"] == 2
    assert body["results"][0]["analysis_review_id"] == "elevenst:1"
    assert body["results"][0]["level"] in {"safe", "warn", "danger"}


def test_collect_uses_default_limit(client) -> None:
    gateway = FakeGateway([crawler_review("1")])

    client(gateway).post(
        "/analysis/collect",
        json={"platform": "elevenst", "product_id": "1831255717"},
    )

    assert gateway.calls == [("elevenst", "1831255717", 50)]


def test_collect_reports_empty_result_as_not_found(client) -> None:
    response = client(FakeGateway([])).post(
        "/analysis/collect",
        json={"platform": "elevenst", "product_id": "1831255717"},
    )

    assert response.status_code == 404
    assert "수집된 리뷰가 없습니다" in response.json()["detail"]


def test_collect_surfaces_crawler_client_error(client) -> None:
    gateway = FakeGateway(
        error=CrawlerRequestError("'nope' collector 가 없습니다.", status_code=404)
    )

    response = client(gateway).post(
        "/analysis/collect",
        json={"platform": "nope", "product_id": "1"},
    )

    assert response.status_code == 404
    assert "collector 가 없습니다" in response.json()["detail"]


def test_collect_maps_crawler_server_error_to_bad_gateway(client) -> None:
    gateway = FakeGateway(error=CrawlerRequestError("수집 중 오류", status_code=500))

    response = client(gateway).post(
        "/analysis/collect",
        json={"platform": "elevenst", "product_id": "1"},
    )

    assert response.status_code == 502


def test_collect_maps_unreachable_crawler_to_gateway_timeout(client) -> None:
    gateway = FakeGateway(error=CrawlerUnavailableError("수집 서비스에 연결하지 못했습니다"))

    response = client(gateway).post(
        "/analysis/collect",
        json={"platform": "elevenst", "product_id": "1"},
    )

    assert response.status_code == 504


def test_collect_rejects_invalid_limit(client) -> None:
    response = client(FakeGateway([])).post(
        "/analysis/collect",
        json={"platform": "elevenst", "product_id": "1", "limit": 0},
    )

    assert response.status_code == 422


def test_platforms_lists_available_and_failed(client) -> None:
    response = client(FakeGateway()).get("/analysis/platforms")

    assert response.status_code == 200
    assert response.json() == {
        "available": ["elevenst", "oliveyoung"],
        "failed": ["gmarket"],
    }


def test_platforms_reports_unreachable_crawler(client) -> None:
    gateway = FakeGateway(error=CrawlerUnavailableError("연결 실패"))

    response = client(gateway).get("/analysis/platforms")

    assert response.status_code == 504
