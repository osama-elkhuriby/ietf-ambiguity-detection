"""Validated, recorded pilot settings; confidence is not yet human calibrated."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    backend: str = Field(default="semantic", pattern="^(semantic|lexical)$")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision: str | None = None
    batch_size: int = Field(default=32, ge=1)
    window_overlap: int = Field(default=32, ge=0, le=64)
    local_top_k: int = Field(default=10, ge=1)
    global_top_k: int = Field(default=5, ge=1)
    section_window: int = Field(default=1, ge=0)
    candidate_floor: float = Field(default=0.35, ge=0, le=1)
    match_floor: float = Field(default=0.62, ge=0, le=1)
    high_score: float = Field(default=0.88, ge=0, le=1)
    high_margin: float = Field(default=0.12, ge=0, le=1)
    uncertain_margin: float = Field(default=0.025, ge=0, le=1)
    max_group: int = Field(default=3, ge=2, le=4)
    group_coverage: float = Field(default=0.75, ge=0, le=1)
    member_coverage: float = Field(default=0.55, ge=0, le=1)
    group_penalty: float = Field(default=0.12, ge=0)
    group_min_tokens: int = Field(default=4, ge=1)
    solver_seconds: float = Field(default=30, gt=0)
    high_audit_fraction: float = Field(default=0.05, ge=0, le=1)
    review_sample_size: int = Field(default=150, ge=1)
    seed: int = 17
    semantic_weight: float = Field(default=0.38, ge=0)
    lexical_weight: float = Field(default=0.38, gt=0)
    structural_weight: float = Field(default=0.12, ge=0)
    position_weight: float = Field(default=0.04, ge=0)
    block_weight: float = Field(default=0.04, ge=0)
    protected_weight: float = Field(default=0.04, ge=0)

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if not self.candidate_floor <= self.match_floor <= self.high_score:
            raise ValueError("Require candidate_floor <= match_floor <= high_score")
        if self.uncertain_margin > self.high_margin:
            raise ValueError("uncertain_margin must not exceed high_margin")
        return self


def load_config(path: Path | None = None, **overrides: object) -> Config:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path else {}
    return Config.model_validate(
        {**(values or {}), **{k: v for k, v in overrides.items() if v is not None}}
    )
