"""Map crawler fields without treating display author names as user IDs."""
from collections.abc import Sequence

from app.schemas.crawler import CrawlerReview
from app.services.analysis import ReviewAnalysisInput

KEY_SEPARATOR = ":"
"""platform과 원본 ID를 잇는 구분자. 예: `elevenst:545961223`."""


def to_scoped_key(platform: str, native_id: str) -> str:
    """platform 스코프 ID를 저장소 전역에서 구분되는 키로 만든다."""

    return f"{platform}{KEY_SEPARATOR}{native_id}"


def resolve_product_key(reviews: Sequence[CrawlerReview]) -> str:
    """리뷰가 모두 같은 상품일 때만 그 상품 키를 반환한다."""

    if not reviews:
        raise ValueError("reviews must not be empty")

    origins = {(review.platform, review.product_id) for review in reviews}
    if len(origins) > 1:
        listed = ", ".join(
            sorted(to_scoped_key(platform, product_id) for platform, product_id in origins)
        )
        raise ValueError(
            "reviews must share one (platform, product_id); "
            f"pass product_key explicitly to group them: {listed}"
        )

    platform, product_id = origins.pop()
    return to_scoped_key(platform, product_id)


def to_analysis_inputs(
    reviews: Sequence[CrawlerReview],
    *,
    product_key: str | None = None,
    derive_user_review_dates: bool = True,
) -> tuple[ReviewAnalysisInput, ...]:
    """crawler 리뷰를 입력 순서 그대로 분석 service 입력으로 변환한다."""

    resolved_key = product_key if product_key is not None else resolve_product_key(reviews)
    if not resolved_key.strip():
        raise ValueError("product_key must not be blank")



    return tuple(
        ReviewAnalysisInput(
            review_id=to_scoped_key(review.platform, review.review_id),
            product_id=resolved_key,
            content=review.content,
            user_id=None,  # No stable ID in the crawler contract.
            review_date=review.written_at,
            # crawler `Review`에 대응 필드가 없어 관측되지 않은 값이다.
            verified_purchase=None,
            account_created_at=None,
            user_review_dates=None,
        )
        for review in reviews
    )
