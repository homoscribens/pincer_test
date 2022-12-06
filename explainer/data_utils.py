from dataclasses import dataclass, field


@dataclass
class MaskedPattern:
    masked_sentence: str
    masked_len: int
    correct: bool
    score_diff: float
    origin_pred: float
    masked_pred: float


@dataclass
class Examples(MaskedPattern):
    keywords: list
    examples_idx: list
    generality: float = None
