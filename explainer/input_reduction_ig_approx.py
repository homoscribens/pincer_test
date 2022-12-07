import re
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer

from tqdm import tqdm

from .data_utils import MaskedPattern


logging.basicConfig(format='[%(asctime)s] [%(levelname)s] <%(funcName)s> %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment'


def load_data(data_dir, IF):
    if IF:
        dataset = load_from_disk(data_dir / 'reduced_train')
    else:
        dataset = load_dataset('tweet_eval', 'sentiment', split='test')
        dataset = dataset.shuffle(seed=42).select(range(500))
    return dataset

def load_model(model_path, device='cuda'):
    logger.info(f'Loading model from {model_path}')
    model = BertClassifier.from_pretrained(model_path, num_labels=2)
    model = model.to(device)
    return model

def clean_text(text):
        text = re.sub('([.,!?()])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        return text

def create_erased_inputs(encoded, sentence, cls_explainer, tokenizer):
    word_attributions = cls_explainer(sentence)

    erased = []
    mod_input_ids = encoded.input_ids[0].clone()

    word_attributions = [(word, score) for word, score in word_attributions if word != '']
    
    # format input_ids to the same format as transformers_interpret
    for i, token in enumerate(tokenizer.convert_ids_to_tokens(mod_input_ids)):
        if token == 'Ġ':
            # delete the ith token
            mod_input_ids = torch.cat([mod_input_ids[:i], mod_input_ids[i+1:]])

    assert len(word_attributions) == len(mod_input_ids), f'{word_attributions} != {tokenizer.convert_ids_to_tokens(mod_input_ids)}'

    for i, _ in sorted(enumerate(word_attributions), key=lambda x: x[1][1]):
        if i == 0 or i == encoded.input_ids[0].shape[0]-1:
            continue
        mod_input_ids[i] = tokenizer.mask_token_id
        
        erased.append(mod_input_ids.clone())
    
    # if there is only one token in sentence, return None
    if len(erased) == 1:
        return None
    
    return torch.stack(erased[:-1])

def enumerate_hypothesis(sentence, cls_explainer, model, tokenizer):
    sentence = clean_text(sentence)
    encoded = tokenizer(sentence, return_tensors='pt').to('cuda')

    with torch.inference_mode():
        output = model(input_ids=encoded['input_ids'])
        origin_logits = output.logits[0].cpu().numpy()

        erased_inputs = create_erased_inputs(encoded, sentence, cls_explainer, tokenizer)

        if erased_inputs is None:
            return None

        output = model(input_ids=erased_inputs)

    predictions = []

    for logit, masked_sentence, masked_sentence_seq in zip(output.logits, erased_inputs, tokenizer.batch_decode(erased_inputs)):
        logit = logit.cpu().softmax(dim=0).numpy()
        
        predictions.append(MaskedPattern(masked_sentence=masked_sentence_seq,
                                         masked_len= len([token for token in masked_sentence if token == tokenizer.mask_token_id]),
                                         correct=origin_logits.argmax() == logit.argmax(),
                                         score_diff=origin_logits.max() - logit.max(),
                                         origin_pred=origin_logits.argmax().item(),
                                         masked_pred=logit.argmax(0).item(),
                                         ))
        
    predictions.sort(key=lambda x: (-int(x.correct), -x.masked_len, -x.score_diff))
    return predictions

def main(args):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    # model = load_model(args.model_path, device=args.device)
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
        logger.info('Iterating reductive maksing...')
        masked_pattern = []
        none = 0
        for data in tqdm(pattern_data):
            pattern = enumerate_hypothesis(data['text'], cls_explainer, model, tokenizer)
            if pattern is not None:
                masked_pattern.append(pattern[0])
            else:
                none += 1
        logger.info(f'Including None: {none}')
        
        logger.info(f'Saving to {args.output_dir}')
        with open(args.output_dir / 'masked_pattern.pkl', 'wb') as f:
            pickle.dump(masked_pattern, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--IF', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task', type=str, default='SA')
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent
    args.model_path = BASE_DIR / 'output' / args.task / 'epoch=4'

    args.data_dir = BASE_DIR / 'data' / args.task

    output_dir = BASE_DIR / 'explainer' / 'pattern' / args.task
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    args.output_dir = output_dir

    main(args)
