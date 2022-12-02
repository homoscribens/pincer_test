from dataclasses import dataclass, field

@dataclass(frozen=True)
class MaskedPattern:
    masked_sentence: str
    masked_len: int
    correct: bool
    score_diff: float
    origin_pred: float
    masked_pred: float

@dataclass(frozen=True)
class Examples(MaskedPattern):
    keywords: list
    examples: list
