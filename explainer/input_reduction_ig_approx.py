import pathlib
import itertools
import argparse
import logging
import pickle

from dataclasses import dataclass, field

from datasets import load_from_disk, load_dataset

import numpy as np
import torch

from models.model import BertClassifier

from transformers import AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

from tqdm import tqdm

from .data_utils import MaskedPattern


logging.basicConfig(format='[%(asctime)s] [%(levelname)s] <%(funcName)s> %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_dir, IF):
    if IF:
        dataset = load_from_disk(data_dir / 'reduced_train')
    else:
        dataset = load_dataset('glue', 'sst2', split='validation')
        dataset = dataset.shuffle(seed=42).select(range(500))
    return dataset

def load_model(model_path, device="cuda"):
    logger.info(f"Loading model from {model_path}")
    model = BertClassifier.from_pretrained(model_path, num_labels=2)
    model = model.to(device)
    return model

def create_erased_inputs(x, s, cls_explainer, tokenizer):
    word_attributions = cls_explainer(s)

    erased = list()
    mod_input_ids = x.input_ids[0].clone()

    word_attributions = [(w, sc) for w, sc in word_attributions if w != ""]

    if len(word_attributions) != mod_input_ids.shape[0]:
        return None

    for i, _ in sorted(enumerate(word_attributions), key=lambda x: x[1][1]):
        if i == 0 or i == x.input_ids[0].shape[0]-1:
            continue

        mod_input_ids[i] = tokenizer.mask_token_id
        erased.append(mod_input_ids.clone())
    
    return torch.stack(erased[:-1])

def enumerate_hypothesis(s, cls_explainer, model, tokenizer):
    x = tokenizer(s, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model(input_ids=x['input_ids'])
        origin_logits = output.logits[0].cpu().numpy()

        erased_inputs = create_erased_inputs(x, s, cls_explainer, tokenizer)

        if erased_inputs is None:
            return None

        output = model(input_ids=erased_inputs)

    predictions = list()

    for logit, masked_x, s in zip(output.logits, erased_inputs, tokenizer.batch_decode(erased_inputs)):
        logit = logit.cpu().softmax(dim=0).numpy()

        predictions.append(MaskedPattern(masked_sentence=s,
                            masked_len= len([t for t in masked_x if t == tokenizer.mask_token_id]),
                            correct=origin_logits.argmax() == logit.argmax(),
                            score_diff=origin_logits.max() - logit.max(),
                            origin_pred=origin_logits.argmax().item(),
                            masked_pred=logit.argmax(0).item(),
                            ))
        
    predictions.sort(key=lambda x: (-int(x.correct), -x.masked_len, -x.score_diff))
    return predictions

def main(args):
    model = load_model(args.model_path, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    cls_explainer = SequenceClassificationExplainer(
        model,
        tokenizer)
    
    pattern_data = load_data(args.data_dir, args.IF)
    if args.IF:
        inf_file_list = list(args.if_dir.glob('influence_test_idx_*.pkl'))

        masked_inf = []
        for file in inf_file_list:
            with file.open('rb') as f:
                inf = pickle.load(f)
            topk_idx = np.flip(np.argsort(inf))[:args.topk]
            topk = [pattern_data[i] for i in topk_idx]

            for i, s in enumerate(topk):
                masked = enumerate_hypothesis(s, cls_explainer, model, tokenizer)
                masked_inf.append(masked)
    else:
        logger.info("Iterating reductive maksing...")
        masked_pattern = [enumerate_hypothesis(data['sentence'], cls_explainer, model, tokenizer)[0] for data in tqdm(pattern_data)]
        
        logger.info(f"Saving to {args.output_dir}")
        with open(args.output_dir / 'masked_pattern.pkl', 'wb') as f:
            pickle.dump(masked_pattern, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--IF", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=str, default='SA')
    parser.add_argument("--model_name", type=str, default='bert-base-uncased')
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent
    args.model_path = BASE_DIR / 'output' / 'SA' / 'epoch=4'

    args.data_dir = BASE_DIR / 'data' / args.task

    output_dir = BASE_DIR / 'explainer' / 'pattern' / args.task
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    args.output_dir = output_dir

    main(args)
