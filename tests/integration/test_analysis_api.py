"""crawler 리뷰 분석 API 통합 테스트."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/analysis/crawler-reviews"

LONG_CONTENT = "묵직한데 깔끔하고 빨대까지 포함되어 있어 좋았습니다. 손잡이 분리로 세척도 편합니다."


def crawler_review(review_id: str, content: str = LONG_CONTENT, **overrides: object) -> dict:
    payload: dict = {
        "platform": "elevenst",
        "product_id": "1831255717",
        "review_id": review_id,
        "content": content,
        "rating": 5.0,
        "author": None,
        "written_at": None,
        "option": None,
        "images": [],
        "helpful_count": None,
        "collected_at": "2026-08-04T20:41:18.802538",
    }
    payload.update(overrides)
    return payload


def test_analyzes_crawler_reviews_and_echoes_source_ids() -> None:
    response = client.post(ENDPOINT, json={"reviews": [crawler_review("545961223")]})

    assert response.status_code == 200
    body = response.json()
    assert body["product_key"] == "elevenst:1831255717"
    assert body["review_count"] == 1

    result = body["results"][0]
    assert result["platform"] == "elevenst"
    assert result["review_id"] == "545961223"
    assert result["product_id"] == "1831255717"
    assert result["analysis_review_id"] == "elevenst:545961223"
    assert result["available"] is True
    assert result["level"] in {"safe", "warn", "danger"}


def test_behavior_signal_is_unavailable_without_crawler_evidence() -> None:
    response = client.post(ENDPOINT, json={"reviews": [crawler_review("1")]})

    signals = response.json()["results"][0]["signals"]
    assert signals["text"]["available"] is True
    assert signals["behavior"]["available"] is False
    assert signals["behavior"]["score"] is None
    assert "insufficient_behavior_evidence" in signals["behavior"]["unavailable_reasons"]


def test_duplicate_content_raises_network_signal() -> None:
    response = client.post(
        ENDPOINT,
        json={
            "reviews": [
                crawler_review("1"),
                crawler_review("2"),
            ]
        },
    )

    results = response.json()["results"]
    network = results[0]["signals"]["network"]
    assert network["available"] is True
    assert "network" in results[0]["used_signals"]
    assert any(reason["source"] == "network" for reason in results[0]["reasons"])


def test_mixed_products_without_product_key_are_rejected() -> None:
    response = client.post(
        ENDPOINT,
        json={
            "reviews": [
                crawler_review("1"),
                crawler_review("2", platform="kurly", product_id="p-2"),
            ]
        },
    )

    assert response.status_code == 422
    assert "must share one" in response.json()["detail"]


def test_explicit_product_key_groups_multiple_platforms() -> None:
    response = client.post(
        ENDPOINT,
        json={
            "product_key": "review-product-1",
            "reviews": [
                crawler_review("1"),
                crawler_review("2", platform="kurly", product_id="p-2"),
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["product_key"] == "review-product-1"
    assert {result["platform"] for result in body["results"]} == {"elevenst", "kurly"}


def test_empty_review_list_is_rejected() -> None:
    response = client.post(ENDPOINT, json={"reviews": []})

    assert response.status_code == 422
