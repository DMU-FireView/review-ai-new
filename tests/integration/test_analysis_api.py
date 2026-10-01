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
    assert body["platform"] == "elevenst"
    assert body["product_id"] == "1831255717"
    assert body["review_count"] == 1

    result = body["results"][0]
    assert result["review_id"] == "545961223"
    assert result["text_score"] == 100
    assert result["level"] in {"safe", "warn", "danger"}


def test_behavior_signal_is_unavailable_without_crawler_evidence() -> None:
    response = client.post(ENDPOINT, json={"reviews": [crawler_review("1")]})

    result = response.json()["results"][0]
    assert result["text_score"] == 100
    assert result["behavior_score"] == -1
    assert result["network_score"] == -1
    assert "signals" not in result



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
    assert results[0]["network_score"] == 9.1
    assert "NETWORK_SIMILAR_REVIEW_PATTERN" in results[0]["reasons"]



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


def test_v05_rejects_multiple_platforms_even_with_product_key() -> None:
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

    assert response.status_code == 422
    assert "must share one" in response.json()["detail"]



def test_empty_review_list_is_rejected() -> None:
    response = client.post(ENDPOINT, json={"reviews": []})

    assert response.status_code == 422
