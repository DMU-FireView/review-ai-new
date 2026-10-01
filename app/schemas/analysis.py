"""Flat Result Contract v0.5. Missing scores are -1; missing level is null."""

from typing import Annotated, Literal
from math import isfinite
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator, model_validator

from app.services.analysis import ReviewAnalysisResult
from app.scoring.meta_scorer import classify_rti


def validate_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be numeric, never null")
    if not isfinite(value) or (value != -1 and not 0 <= value <= 100):
        raise ValueError("score must be -1 or between 0 and 100")
    return float(value)


Score = Annotated[float, BeforeValidator(validate_score), Field(ge=-1, le=100)]


class ReviewAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    rti: Score
    level: Literal["safe", "warn", "danger"] | None
    text_score: Score
    behavior_score: Score
    network_score: Score
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_level(self):
        missing = all(score == -1 for score in (self.text_score, self.behavior_score, self.network_score))
        if (self.rti == -1) != missing:
            raise ValueError("RTI must be unavailable exactly when all signals are unavailable")
        expected = None if self.rti == -1 else classify_rti(self.rti).value
        if self.level != expected:
            raise ValueError("level must match RTI; unavailable RTI has null level")
        return self


class ProductAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str
    product_id: str
    review_count: int = Field(ge=0)
    results: list[ReviewAnalysisResponse]

    @model_validator(mode="after")
    def consistent_count(self):
        if self.review_count != len(self.results):
            raise ValueError("review_count must equal results length")
        return self


def to_review_response(result: ReviewAnalysisResult, *, platform: str,
                       review_id: str, product_id: str) -> ReviewAnalysisResponse:
    rti = round(result.rti, 1) if result.rti is not None else -1
    return ReviewAnalysisResponse(
        review_id=review_id,
        rti=rti,
        level=classify_rti(rti).value if rti != -1 else None,
        text_score=result.signals.text.score if result.signals.text.available else -1,
        behavior_score=result.signals.behavior.score if result.signals.behavior.available else -1,
        network_score=result.signals.network.score if result.signals.network.available else -1,
        reasons=list(dict.fromkeys(f"{reason.source.upper()}_{reason.code}" for reason in result.reasons)),
    )
