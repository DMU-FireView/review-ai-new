"""crawler 리뷰를 분석 service 입력으로 변환하는 adapter 모듈.

역할:
- crawler 원본 필드를 `ReviewAnalysisInput` 필드로 매핑한다.
- platform 스코프인 원본 ID를 저장소 전역에서 충돌하지 않는 키로 만든다.
- 같은 요청에 담긴 리뷰에서만 동일 author의 작성 시각을 모아 준다.

수정 범위:
- [INTEGRATION]
- crawler 계약이 바뀌면 이 파일의 매핑을 갱신한다.
- analyzer 점수 의미나 scoring 정책은 이 파일에서 변경하지 않는다.

주의:
- crawler가 제공하지 않는 값은 None으로 남긴다. 0·False·중립값으로 보정하지 않는다.
  현재 crawler `Review`에는 verified_purchase와 account_created_at에 해당하는 필드가 없다.
- `user_review_dates`는 요청에 담긴 리뷰에서만 모은 관측값이므로 실제 작성 이력의 하한값이다.
  P_behavior의 `reviews_written_today`는 임계값 이상일 때만 감점하므로, 하한값을 쓰면
  과소 집계로 관대해질 뿐 없는 근거로 감점하지는 않는다.
- crawler 패키지를 import하지 않는다.

책임 흐름:
Crawler → CrawlerReview schema → 이 adapter → Analysis Service → analyzers → Meta-Scorer.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime

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

    dates_by_author = (
        _collect_written_at_by_author(reviews) if derive_user_review_dates else {}
    )

    return tuple(
        ReviewAnalysisInput(
            review_id=to_scoped_key(review.platform, review.review_id),
            product_id=resolved_key,
            content=review.content,
            user_id=review.author,
            review_date=review.written_at,
            # crawler `Review`에 대응 필드가 없어 관측되지 않은 값이다.
            verified_purchase=None,
            account_created_at=None,
            user_review_dates=dates_by_author.get(_author_key(review)),
        )
        for review in reviews
    )


def _collect_written_at_by_author(
    reviews: Iterable[CrawlerReview],
) -> dict[str, tuple[datetime, ...]]:
    """author별 written_at을 요청 안에서만 모아 준다."""

    collected: dict[str, list[datetime]] = defaultdict(list)
    for review in reviews:
        author = _author_key(review)
        if author is None or review.written_at is None:
            continue
        collected[author].append(review.written_at)

    return {author: tuple(sorted(dates)) for author, dates in collected.items()}


def _author_key(review: CrawlerReview) -> str | None:
    """동일 사용자 판별에 쓸 수 있는 author만 키로 인정한다."""

    if review.author is None:
        return None

    author = review.author.strip()
    return author or None
