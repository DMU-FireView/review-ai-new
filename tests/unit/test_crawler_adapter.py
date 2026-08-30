"""crawler 리뷰 → 분석 service 입력 변환 adapter 테스트."""

from datetime import datetime

import pytest

from app.integrations.crawler import (
    resolve_product_key,
    to_analysis_inputs,
    to_scoped_key,
)
from app.schemas.crawler import CrawlerReview


def crawler_review(
    review_id: str,
    content: str = "충분히 긴 일반 리뷰 본문입니다. 만족스럽게 사용했습니다.",
    *,
    platform: str = "elevenst",
    product_id: str = "1831255717",
    author: str | None = None,
    written_at: datetime | None = None,
) -> CrawlerReview:
    return CrawlerReview(
        platform=platform,
        product_id=product_id,
        review_id=review_id,
        content=content,
        author=author,
        written_at=written_at,
    )


def test_maps_crawler_fields_to_analysis_input() -> None:
    written_at = datetime(2026, 6, 9)
    review = crawler_review("545961223", author="가나다라01", written_at=written_at)

    result = to_analysis_inputs((review,))[0]

    assert result.review_id == "elevenst:545961223"
    assert result.product_id == "elevenst:1831255717"
    assert result.content == review.content
    assert result.user_id == "가나다라01"
    assert result.review_date == written_at


def test_missing_crawler_fields_stay_none() -> None:
    result = to_analysis_inputs((crawler_review("1"),))[0]

    assert result.verified_purchase is None
    assert result.account_created_at is None
    assert result.user_id is None
    assert result.review_date is None
    assert result.user_review_dates is None


def test_parses_raw_crawler_json_payload() -> None:
    payload = {
        "platform": "elevenst",
        "product_id": "1831255717",
        "review_id": "545961223",
        "content": "묵직한데 깔끔하고 빨대까지 포함되어 있어 좋아요.",
        "rating": 5.0,
        "author": "가나다라01",
        "written_at": "2026-06-09T00:00:00",
        "option": "색상:모카그레이",
        "images": ["https://cdn.011st.com/a.jpg"],
        "helpful_count": 2,
        "collected_at": "2026-08-04T20:41:18.802538",
    }

    result = to_analysis_inputs((CrawlerReview.model_validate(payload),))[0]

    assert result.review_id == "elevenst:545961223"
    assert result.review_date == datetime(2026, 6, 9)


def test_unknown_crawler_fields_are_ignored() -> None:
    review = CrawlerReview.model_validate(
        {
            "platform": "kurly",
            "product_id": "p-1",
            "review_id": "r-1",
            "content": "본문",
            "future_field": "무시되어야 합니다",
        }
    )

    assert review.review_id == "r-1"


def test_derives_user_review_dates_within_request() -> None:
    first = datetime(2026, 6, 9, 10)
    second = datetime(2026, 6, 9, 12)
    reviews = (
        crawler_review("1", author="같은사람", written_at=second),
        crawler_review("2", author="같은사람", written_at=first),
        crawler_review("3", author="다른사람", written_at=first),
    )

    results = to_analysis_inputs(reviews)

    assert results[0].user_review_dates == (first, second)
    assert results[1].user_review_dates == (first, second)
    assert results[2].user_review_dates == (first,)


def test_derivation_can_be_disabled() -> None:
    reviews = (
        crawler_review("1", author="같은사람", written_at=datetime(2026, 6, 9)),
        crawler_review("2", author="같은사람", written_at=datetime(2026, 6, 9)),
    )

    results = to_analysis_inputs(reviews, derive_user_review_dates=False)

    assert all(result.user_review_dates is None for result in results)


def test_blank_author_is_not_grouped() -> None:
    reviews = (
        crawler_review("1", author="   ", written_at=datetime(2026, 6, 9)),
        crawler_review("2", author="", written_at=datetime(2026, 6, 9)),
    )

    results = to_analysis_inputs(reviews)

    assert all(result.user_review_dates is None for result in results)


def test_resolve_product_key_requires_single_origin() -> None:
    reviews = (
        crawler_review("1"),
        crawler_review("2", platform="kurly", product_id="p-2"),
    )

    with pytest.raises(ValueError, match="must share one"):
        resolve_product_key(reviews)


def test_explicit_product_key_groups_multiple_platforms() -> None:
    reviews = (
        crawler_review("1"),
        crawler_review("2", platform="kurly", product_id="p-2"),
    )

    results = to_analysis_inputs(reviews, product_key="review-product-1")

    assert {result.product_id for result in results} == {"review-product-1"}
    assert results[1].review_id == "kurly:2"


def test_blank_product_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        to_analysis_inputs((crawler_review("1"),), product_key="  ")


def test_empty_reviews_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        to_analysis_inputs(())


def test_scoped_key_format() -> None:
    assert to_scoped_key("elevenst", "545961223") == "elevenst:545961223"
