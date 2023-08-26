import re
import pathlib
import itertools
import argparse
import pickle
from logging import getLogger, basicConfig, INFO

from dataclasses import dataclass, field

from datasets import load_from_disk, load_dataset

import numpy as np
import torch

from models.model import BertClassifier

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer, PairwiseSequenceClassificationExplainer

from tqdm import tqdm

from .data_utils import MaskedPattern


basicConfig(format='[%(asctime)s] [%(levelname)s] <%(funcName)s> %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S',
                    level=INFO)
logger = getLogger(__name__)

def relabel(example):
    if example['label'] == 0:
        example['label'] = 2
    elif example['label'] == 2:
        example['label'] = 0
    return example
    
def load_data(task, data_size=1000):
    if task == 'SA':
        dataset = load_dataset('tweet_eval', 'sentiment', split='test')
    elif task == 'SA_train':
        dataset = load_dataset('tweet_eval', 'sentiment', split='train')
    elif task == 'NLI':
        dataset = load_dataset('glue', 'mnli', split='validation_matched')
        dataset = dataset.map(relabel, batched=False)
    elif task == 'NLI_train':
        dataset = load_dataset('glue', 'mnli', split='train')
        dataset = dataset.map(relabel, batched=False)
        
    dataset = dataset.shuffle(seed=42).select(range(data_size))
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
    
def preprocess(dataset, task):
    if task in ['SA', 'SA_train']:
        dataset = dataset.map(lambda x: {'text': clean_text(x['text'])})
    elif task in ['NLI', 'NLI_train']:
        dataset = dataset.map(lambda x: {'preimse': clean_text(x['premise']), 'hypothesis': clean_text(x['hypothesis'])})
        
    return dataset

def get_index(l, default=False):
    return l.index('Ġ') if 'Ġ' in l else default

def create_erased_inputs(encoded, sentence, cls_explainer, tokenizer):
    if cls_explainer.__class__.__name__.startswith('Sequence'):
        word_attributions = cls_explainer(sentence)
    elif cls_explainer.__class__.__name__.startswith('Pairwise'):
        word_attributions = cls_explainer(sentence[0], sentence[1])
    else:
        raise ValueError(f'Invalid explainer: {cls_explainer.__class__.__name__}')

    erased = []
    mod_input_ids = encoded['input_ids'][0].clone()

    word_attributions = [(word, score) for word, score in word_attributions if word != '']
    
    # format input_ids to the same format as transformers_interpret
    while True:
        idx = get_index(tokenizer.convert_ids_to_tokens(mod_input_ids))
        if idx:
            mod_input_ids = torch.cat([mod_input_ids[:idx], mod_input_ids[idx+1:]])
        else:
            break
            
    assert len(word_attributions) == len(mod_input_ids), f'{word_attributions} != {tokenizer.convert_ids_to_tokens(mod_input_ids)}'

    for i, _ in sorted(enumerate(word_attributions), key=lambda x: x[1][1]):
        if i == 0 or i == encoded['input_ids'][0].shape[0]-1 or mod_input_ids[i] == tokenizer.sep_token_id:
            continue
        mod_input_ids[i] = tokenizer.mask_token_id
        
        erased.append(mod_input_ids.clone())
    
    # if there is only one token in sentence, return None
    if len(erased) == 1:
        return None
    
    return torch.stack(erased[:-1])

def enumerate_hypothesis(dataset, cls_explainer, model, tokenizer, task, device='cuda'):
    dataset = preprocess(dataset, task=task)
    
    none = 0
    aggr_predictions = []
    for data in tqdm(dataset, desc='Masking'):
        if cls_explainer.__class__.__name__.startswith('Sequence'):
            sentence = data['text']
            encoded = tokenizer(sentence, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
        elif cls_explainer.__class__.__name__.startswith('Pair'):
            premise, hypothesis = data['premise'], data['hypothesis']
            sentence = [premise, hypothesis]
            encoded = tokenizer(premise, hypothesis, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
        else:
            raise ValueError(f'Invalid explainer: {cls_explainer.__class__.__name__}')
        
        gold_label = data['label']
        
        # Put outside of inference mode to get the word attributions,
        # otherwise the gradients will not be computed. (it disconnects the graph)
        erased_inputs = create_erased_inputs(encoded, sentence, cls_explainer, tokenizer)
        if erased_inputs is None:
            none += 1
            continue
        
        with torch.inference_mode():
            origin_output = model(input_ids=encoded['input_ids'])
            origin_logits = origin_output.logits[0].detach().cpu().numpy()

            output = model(input_ids=erased_inputs)

        predictions = []

        for logit, masked_sentence, masked_sentence_seq in zip(output.logits, erased_inputs, tokenizer.batch_decode(erased_inputs)):
            logit = logit.detach().cpu().softmax(dim=0).numpy()
            
            predictions.append(MaskedPattern(
                masked_sentence=masked_sentence_seq,
                masked_len=len([token for token in masked_sentence if token == tokenizer.mask_token_id]),
                correct=origin_logits.argmax() == logit.argmax(),
                score_diff=origin_logits.max() - logit.max(),
                origin_pred=origin_logits.argmax().item(),
                masked_pred=logit.argmax(0).item(),
                gold_label=gold_label,
                ))
            
        predictions.sort(key=lambda x: (-int(x.correct), -x.masked_len, -x.score_diff))
        
        if predictions is not None:
                aggr_predictions.append(predictions[0])
        else:
            none += 1
    logger.info(f'Availabe pattern {len(aggr_predictions) - none}')
    return aggr_predictions

def main(args):
    logger.info(f'Loading model')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    # model = load_model(args.model_path, device=args.device)
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.task in ['SA', 'SA_train']:
        cls_explainer = SequenceClassificationExplainer(
            model,
            tokenizer)
    elif args.task in ['NLI', 'NLI_train']:
        cls_explainer = PairwiseSequenceClassificationExplainer(
            model,
            tokenizer)
    else:
        raise ValueError(f'Invalid task: {args.task}')
    
    pattern_data = load_data(args.task, args.data_size)
    logger.info('Iterating reductive maksing...')
    masked_pattern = enumerate_hypothesis(pattern_data, cls_explainer, model, tokenizer, args.task, device=args.device)
    logger.info(f'Saving to {args.output_dir}')
    with open(args.output_dir / 'masked_pattern.pkl', 'wb') as f:
        pickle.dump(masked_pattern, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--IF', action='store_true')
    parser.add_argument('--task', type=str, default='SA')
    parser.add_argument('--data-size', type=int, default=1000)
    
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent
    args.model_path = BASE_DIR / 'output' / args.task / 'epoch=4'
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.task in ['SA', 'SA_train']:
        args.model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        args.tokenizer = 'roberta-base' # 'cardiffnlp/twitter-roberta-base-sentiment' has ill defined tokenizer
    elif args.task in ['NLI', 'NLI_train']:
        args.model_name = 'roberta-large-mnli'
        args.tokenizer = args.model_name
    else:
        raise ValueError(f'Invalid task: {args.task}')

    args.data_dir = BASE_DIR / 'data' / args.task

    output_dir = BASE_DIR / 'explainer' / 'pattern' / args.task
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    args.output_dir = output_dir

    main(args)
