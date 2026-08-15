"""리뷰 간 similarity 계산을 P_network에 제공하는 adapter 모듈.

수정 범위:
- [MODEL / EXTERNAL INTEGRATION]
- 향후 한국어 embedding 또는 sentence-transformers 모델로 교체할 수 있다.
- P_network scoring 정책은 이 파일에서 변경하지 않는다.

현재는 최소 문자열 정규화를 사용하며 계산 불가 결과를 임의 수치로 만들지 않는다.
"""

import re


_WHITESPACE_PATTERN = re.compile(r"\s+")


class NormalizedTextSimilarityAdapter:
    """최소 정규화 후 본문 동일 여부를 0 또는 1의 유사도로 반환한다."""

    def calculate(self, left: str, right: str) -> float | None:
        """빈 본문은 None, 정규화 후 동일하면 1.0, 다르면 0.0을 반환한다."""

        normalized_left = normalize_text(left)
        normalized_right = normalize_text(right)
        if not normalized_left or not normalized_right:
            return None
        return 1.0 if normalized_left == normalized_right else 0.0


def normalize_text(content: str) -> str:
    """앞뒤·연속 공백과 영문 대소문자만 최소한으로 정규화한다."""

    return _WHITESPACE_PATTERN.sub(" ", content.strip()).casefold()
