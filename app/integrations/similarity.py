"""리뷰 간 similarity 계산을 P_network에 제공하는 adapter 모듈.

수정 범위:
- [AI 연동]
- 향후 한국어 embedding 또는 sentence-transformers 모델로 교체할 수 있다.
- P_network scoring 정책은 이 파일에서 변경하지 않는다.

현재는 최소 문자열 정규화를 사용하며 계산 불가 결과를 임의 수치로 만들지 않는다.
"""

import re
import unicodedata
from collections import Counter
from math import log, sqrt

NGRAM_RANGE = (2, 4)
MIN_INFORMATION_CHARS = 10
HYBRID_TFIDF_WEIGHT = 0.40
HYBRID_BIGRAM_DICE_WEIGHT = 0.60


def tfidf_vectors(texts):
    """Sparse char n-gram TF-IDF, fitted once per batch; no external model.

    Smoothed IDF = 1 + log((1 + N)/(1 + df)); raw term frequency,
    L2 normalization. Empty/non-alphanumeric text is not scoreable.
    """
    counts = []
    for text in texts:
        text = normalize_text(text)
        if not any(c.isalnum() for c in text):
            counts.append(Counter())
            continue
        counts.append(Counter(text[i:i+n] for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1]+1)
                              for i in range(len(text)-n+1)))
    df = Counter(term for row in counts for term in row)
    size = sum(bool(row) for row in counts)
    vectors = []
    for row in counts:
        weighted = {term: count * (1 + log((1+size)/(1+df[term]))) for term, count in row.items()}
        norm = sqrt(sum(value*value for value in weighted.values()))
        vectors.append({term: value/norm for term, value in weighted.items()} if norm else {})
    return vectors


def sparse_cosine(left, right):
    if not left or not right:
        return None
    if len(left) > len(right):
        left, right = right, left
    return min(1.0, max(0.0, sum(value * right.get(term, 0.0) for term, value in left.items())))


def canonical_text(content: str) -> str:
    """Return a comparison-only representation without changing source content."""

    normalized = unicodedata.normalize("NFKC", content).casefold()
    return "".join(character for character in normalized if character.isalnum())


def is_informative_text(content: str) -> bool:
    """Protect short/common reviews from becoming network evidence."""

    return len(canonical_text(content)) >= MIN_INFORMATION_CHARS


def bigram_dice(left: str, right: str) -> float | None:
    """Multiset character-bigram Dice similarity for canonical strings."""

    left_counts = Counter(left[index:index + 2] for index in range(len(left) - 1))
    right_counts = Counter(right[index:index + 2] for index in range(len(right) - 1))
    denominator = sum(left_counts.values()) + sum(right_counts.values())
    if denominator == 0:
        return None
    overlap = sum((left_counts & right_counts).values())
    return 2.0 * overlap / denominator


def hybrid_similarity_rows(texts):
    """Return per-row Hybrid A similarities, excluding short evidence.

    Hybrid A is 0.40 batch TF-IDF cosine + 0.60 canonical bigram Dice.
    The function keeps only O(N) sparse vectors and per-row evidence lists.
    """

    canonical = [canonical_text(text) for text in texts]
    informative = [len(text) >= MIN_INFORMATION_CHARS for text in canonical]
    vectors = tfidf_vectors(canonical)
    rows = [[] for _ in canonical]
    for left_index in range(len(canonical)):
        if not informative[left_index]:
            continue
        for right_index in range(left_index + 1, len(canonical)):
            if not informative[right_index]:
                continue
            if canonical[left_index] == canonical[right_index]:
                similarity = 1.0
            else:
                cosine = sparse_cosine(vectors[left_index], vectors[right_index])
                dice = bigram_dice(canonical[left_index], canonical[right_index])
                if cosine is None or dice is None:
                    continue
                similarity = (
                    HYBRID_TFIDF_WEIGHT * cosine
                    + HYBRID_BIGRAM_DICE_WEIGHT * dice
                )
            rows[left_index].append(similarity)
            rows[right_index].append(similarity)
    return rows


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
