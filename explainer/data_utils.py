from dataclasses import dataclass, field


@dataclass
class MaskedPattern:
    masked_sentence: str
    masked_len: int
    correct: bool
    score_diff: float
    origin_pred: int
    masked_pred: int
    gold_label: int


@dataclass
class Examples(MaskedPattern):
    keywords: list
    examples_idx_ood: list
    examples_idx_iid: list = None
    isTruncated: bool = False
    example_preds_iid: list = None
    example_preds_ood: list = None
    example_f1_iid: float = None
    example_f1_ood: float = None
    f1_diff: float = None
    generality: float = None
    iid_acc: float = None
