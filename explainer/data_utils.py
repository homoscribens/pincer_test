from dataclasses import dataclass, field


@dataclass
class MaskedPattern:
    masked_sentence: str
    masked_len: int
    correct: bool
    score_diff: float
    origin_pred: int
    masked_pred: int


@dataclass
class Examples(MaskedPattern):
    keywords: list
    examples_idx: list
    example_preds: list = None
    generality: float = None
    
