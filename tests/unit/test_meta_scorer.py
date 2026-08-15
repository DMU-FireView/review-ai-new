"""Meta-Scorer의 입력 검증, 가중치 재정규화 및 RTI 등급 테스트."""

import pytest

from app.scoring.meta_scorer import (
    MetaScoreInput,
    MetaScoreWeights,
    RTILevel,
    ScoreSignal,
    calculate_meta_score,
    classify_rti,
)


def available(score: float) -> ScoreSignal:
    return ScoreSignal(available=True, score=score)


def unavailable() -> ScoreSignal:
    return ScoreSignal(available=False, score=None)


def test_all_signals_use_default_weights() -> None:
    result = calculate_meta_score(
        MetaScoreInput(
            text=available(80),
            behavior=available(60),
            network=available(90),
        )
    )

    assert result.rti == pytest.approx(76.0)
    assert result.applied_weights == {
        "text": pytest.approx(0.5),
        "behavior": pytest.approx(0.3),
        "network": pytest.approx(0.2),
    }


def test_behavior_unavailable_renormalizes_text_and_network() -> None:
    result = calculate_meta_score(
        MetaScoreInput(
            text=available(80),
            behavior=unavailable(),
            network=available(70),
        )
    )

    assert result.available is True
    assert result.used_signals == ("text", "network")
    assert result.unavailable_signals == ("behavior",)
    assert result.used_signal_count == 2
    assert result.applied_weights["text"] == pytest.approx(0.5 / 0.7)
    assert result.applied_weights["network"] == pytest.approx(0.2 / 0.7)
    assert sum(result.applied_weights.values()) == pytest.approx(1.0)
    assert result.rti == pytest.approx(80 * (0.5 / 0.7) + 70 * (0.2 / 0.7))


def test_network_unavailable_renormalizes_text_and_behavior() -> None:
    result = calculate_meta_score(
        MetaScoreInput(
            text=available(70),
            behavior=available(90),
            network=unavailable(),
        )
    )

    assert result.applied_weights["text"] == pytest.approx(0.5 / 0.8)
    assert result.applied_weights["behavior"] == pytest.approx(0.3 / 0.8)
    assert result.rti == pytest.approx(70 * (0.5 / 0.8) + 90 * (0.3 / 0.8))


def test_text_only_receives_full_applied_weight() -> None:
    result = calculate_meta_score(
        MetaScoreInput(
            text=available(72),
            behavior=unavailable(),
            network=unavailable(),
        )
    )

    assert result.applied_weights == {"text": pytest.approx(1.0)}
    assert result.rti == pytest.approx(72.0)


def test_all_signals_unavailable_does_not_create_rti() -> None:
    result = calculate_meta_score(
        MetaScoreInput(
            text=unavailable(),
            behavior=unavailable(),
            network=unavailable(),
        )
    )

    assert result.available is False
    assert result.rti is None
    assert result.level is None
    assert result.used_signal_count == 0
    assert result.applied_weights == {}
    assert result.unavailable_reason == "no_available_signals"


def test_unavailable_signal_is_not_treated_as_zero() -> None:
    result = calculate_meta_score(
        MetaScoreInput(
            text=available(100),
            behavior=unavailable(),
            network=available(100),
        )
    )

    assert result.rti == pytest.approx(100.0)
    assert "behavior" not in result.applied_weights


@pytest.mark.parametrize("score", [-0.01, -1.0])
def test_negative_score_is_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        available(score)


@pytest.mark.parametrize("score", [100.01, 101.0])
def test_score_above_one_hundred_is_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        available(score)


def test_available_signal_without_score_is_rejected() -> None:
    with pytest.raises(ValueError, match="must have a score"):
        ScoreSignal(available=True, score=None)


def test_unavailable_signal_with_score_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not have a score"):
        ScoreSignal(available=False, score=50)


def test_applied_weights_sum_to_one() -> None:
    result = calculate_meta_score(
        MetaScoreInput(
            text=available(10),
            behavior=unavailable(),
            network=available(20),
        )
    )

    assert sum(result.applied_weights.values()) == pytest.approx(1.0)


def test_invalid_policy_weight_is_rejected() -> None:
    with pytest.raises(TypeError, match="text weight must be a number"):
        MetaScoreWeights(text=True)


@pytest.mark.parametrize(
    ("rti", "expected"),
    [
        # Ground Truth로 검증된 threshold가 아닌 MVP용 v0 policy 경계다.
        (49.99, RTILevel.DANGER),
        (50.0, RTILevel.WARN),
        (79.99, RTILevel.WARN),
        (80.0, RTILevel.SAFE),
    ],
)
def test_rti_level_boundaries(rti: float, expected: RTILevel) -> None:
    assert classify_rti(rti) is expected
