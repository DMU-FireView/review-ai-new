"""P_text/P_behavior/P_network를 종합해 최종 RTI를 계산하는 모듈.

역할:
- 사용 가능한 신호만 가중 합산하고 가중치를 재정규화한다.
- RTI와 safe/warn/danger 등급 및 계산 메타데이터를 반환한다.

수정 범위:
- [AI CORE]
- RTI 의미에 직접 영향을 주는 핵심 파일이다.
- 연동 목적으로 weight 또는 safe/warn/danger threshold를 수정하지 않는다.
- 정책 변경은 AI 담당자와 협의한다.

주의:
- unavailable 신호를 0점·100점·중립값으로 대체하지 않는다.
- 현재 weight와 threshold는 Ground Truth로 검증된 최종값이 아닌 v0 policy다.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


SAFE_THRESHOLD = 80.0
WARN_THRESHOLD = 50.0


class RTILevel(str, Enum):
    """RTI가 높을수록 신뢰도가 높은 v0 등급."""

    SAFE = "safe"
    WARN = "warn"
    DANGER = "danger"


@dataclass(frozen=True, slots=True)
class ScoreSignal:
    """Analyzer 구현과 독립적인 단일 점수 신호."""

    available: bool
    score: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool")
        if self.available and self.score is None:
            raise ValueError("available signal must have a score")
        if not self.available and self.score is not None:
            raise ValueError("unavailable signal must not have a score")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise TypeError("score must be a number")
            if not isfinite(self.score):
                raise ValueError("score must be finite")
            if not 0.0 <= self.score <= 100.0:
                raise ValueError("score must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class MetaScoreInput:
    """Meta-Scorer에 전달되는 세 분석 신호."""

    text: ScoreSignal
    behavior: ScoreSignal
    network: ScoreSignal


@dataclass(frozen=True, slots=True)
class MetaScoreWeights:
    """교체 가능한 Meta-Scorer 기본 가중치 정책."""

    text: float = 0.50
    behavior: float = 0.30
    network: float = 0.20

    def __post_init__(self) -> None:
        raw_weights = {
            "text": self.text,
            "behavior": self.behavior,
            "network": self.network,
        }
        for name, weight in raw_weights.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"{name} weight must be a number")
            if not isfinite(weight) or weight <= 0.0:
                raise ValueError(f"{name} weight must be finite and greater than zero")

    def as_dict(self) -> dict[str, float]:
        """신호 이름별 기본 가중치를 반환한다."""

        return {
            "text": float(self.text),
            "behavior": float(self.behavior),
            "network": float(self.network),
        }


# Ground Truth 검증 전까지 사용하는 교체 가능한 v0 policy weight다.
DEFAULT_WEIGHTS = MetaScoreWeights()


@dataclass(frozen=True, slots=True)
class MetaScoringMetadata:
    """RTI 계산을 재현할 수 있는 기본 가중치와 가용 가중치 합."""

    base_weights: dict[str, float]
    available_base_weight_sum: float


@dataclass(frozen=True, slots=True)
class MetaScoreResult:
    """최종 RTI와 계산에 사용된 신호·가중치 정보."""

    available: bool
    rti: float | None
    level: RTILevel | None
    used_signals: tuple[str, ...]
    used_signal_count: int
    unavailable_signals: tuple[str, ...]
    applied_weights: dict[str, float]
    metadata: MetaScoringMetadata
    unavailable_reason: str | None


def calculate_meta_score(
    signals: MetaScoreInput,
    *,
    weights: MetaScoreWeights = DEFAULT_WEIGHTS,
) -> MetaScoreResult:
    """사용 가능한 신호의 기본 가중치를 재정규화해 RTI를 계산한다."""

    named_signals = {
        "text": signals.text,
        "behavior": signals.behavior,
        "network": signals.network,
    }
    base_weights = weights.as_dict()
    used_signals = tuple(
        name for name, signal in named_signals.items() if signal.available
    )
    unavailable_signals = tuple(
        name for name, signal in named_signals.items() if not signal.available
    )
    available_weight_sum = sum(base_weights[name] for name in used_signals)
    metadata = MetaScoringMetadata(
        base_weights=base_weights,
        available_base_weight_sum=available_weight_sum,
    )

    if not used_signals:
        return MetaScoreResult(
            available=False,
            rti=None,
            level=None,
            used_signals=(),
            used_signal_count=len(used_signals),
            unavailable_signals=unavailable_signals,
            applied_weights={},
            metadata=metadata,
            unavailable_reason="no_available_signals",
        )

    applied_weights = {
        name: base_weights[name] / available_weight_sum for name in used_signals
    }
    rti = sum(
        _available_score(named_signals[name]) * applied_weights[name]
        for name in used_signals
    )

    return MetaScoreResult(
        available=True,
        rti=rti,
        level=classify_rti(rti),
        used_signals=used_signals,
        used_signal_count=len(used_signals),
        unavailable_signals=unavailable_signals,
        applied_weights=applied_weights,
        metadata=metadata,
        unavailable_reason=None,
    )


def classify_rti(rti: float) -> RTILevel:
    """v0 threshold에 따라 높은 RTI를 더 신뢰 가능한 등급으로 분류한다."""

    if isinstance(rti, bool) or not isinstance(rti, (int, float)):
        raise TypeError("rti must be a number")
    if not isfinite(rti) or not 0.0 <= rti <= 100.0:
        raise ValueError("rti must be finite and between 0 and 100")
    if rti >= SAFE_THRESHOLD:
        return RTILevel.SAFE
    if rti >= WARN_THRESHOLD:
        return RTILevel.WARN
    return RTILevel.DANGER


def _available_score(signal: ScoreSignal) -> float:
    """검증 완료된 available 신호의 점수를 정적 타입에 맞게 좁힌다."""

    if signal.score is None:  # ScoreSignal 검증상 도달할 수 없는 방어 코드
        raise ValueError("available signal must have a score")
    return float(signal.score)
