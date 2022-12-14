import argparse
import pathlib
import pickle
import json
import string
from logging import getLogger, basicConfig, INFO

from dataclasses import dataclass, field

import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tqdm import tqdm

from sklearn.metrics import f1_score

from models.model import BertClassifier
from .data_utils import Examples, MaskedPattern

basicConfig(format='[%(asctime)s] [%(levelname)s] <%(funcName)s> %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S',
                    level=INFO)
logger = getLogger(__name__)


def relabel(example):
    label = example['label']
    if label <= 1:
        example['label'] = 0
    elif label == 2:
        example['label'] = 1
    elif label >= 3:
        example['label'] = 2
    return example

def load_datas(pattern_dir, task):
    logger.info(f'Loading data...')
    if task == 'SA':
        dataset = load_dataset('yelp_review_full', split='train')
        dataset = dataset.shuffle(seed=42).select(range(25000))
        dataset = dataset.map(relabel, batched=False)
    elif task == 'NLI':
        dataset = load_dataset('anli', split='train_r1')
    masked_pattern = pickle.load(open(pattern_dir / 'masked_pattern.pkl', 'rb'))
    return dataset, masked_pattern

def remove_stopwords(text):
    stopwords = string.punctuation
    for sw in stopwords:
        text = text.strip(sw)
    return text

def get_examples(dataset, masked_patern, tokenizer, task):
    logger.info(f'Creating corpus...')
    logger.info(f'Number of available seed pattern: {len(masked_patern)}')
    
    aggr_examples = []
    stopwords = list(string.punctuation)
    special_tokens = tokenizer.all_special_tokens
    
    if task == 'SA':
        
        def tokenize(example):
            tokenized = tokenizer.tokenize(example['text'])
            tokenized = [token.replace('Ġ', '') for token in tokenized if not token == 'Ġ']
            example['tokenized'] = tokenized
            return example
        
        logger.info('Tokenizing dataset...')
        tokenized_texts = dataset.map(tokenize, batched=False)['tokenized']
        tokenized_texts = list(map(set, tokenized_texts))
        
        for pattern in tqdm(masked_patern, desc='Collecting examples'):
            tokenized = tokenizer.tokenize(pattern.masked_sentence)
            tokenized = [token.replace('Ġ', '') for token in tokenized if not token == 'Ġ']
            key_words = [word for word in tokenized if word not in special_tokens]
            
            if not key_words:
                continue
            
            example_indicies = []
            for i, text in enumerate(tokenized_texts):
                if set(key_words).issubset(text):
                    example_indicies.append(i)
                    
            aggr_examples.append(Examples(**vars(pattern), 
                                          keywords=key_words,
                                          examples_idx=example_indicies)) 
        
    elif task == 'NLI':
        
        def tokenize(example):
            tokenized_premise = tokenizer.tokenize(example['premise'])
            tokenized_hypothesis = tokenizer.tokenize(example['hypothesis'])
            tokenized_premise = [token.replace('Ġ', '') for token in tokenized_premise if not token == 'Ġ']
            tokenized_hypothesis = [token.replace('Ġ', '') for token in tokenized_hypothesis if not token == 'Ġ']
            example['tokenized_premise'] = tokenized_premise
            example['tokenized_hypothesis'] = tokenized_hypothesis
            return example
        
        logger.info('Tokenizing dataset...')
        tokenized_premise = dataset.map(tokenize, batched=False)['tokenized_premise']
        tokenized_hypothesis = dataset.map(tokenize, batched=False)['tokenized_hypothesis']
        tokenized_premise = list(map(set, tokenized_premise))
        tokenized_hypothesis = list(map(set, tokenized_hypothesis))
        
        for pattern in tqdm(masked_patern, desc='Collecting examples'):
            tokenized_pattern = tokenizer.tokenize(pattern.masked_sentence)
            tokenized_pattern = [token.replace('Ġ', '') for token in tokenized_pattern if not token == 'Ġ']
            sep_token = tokenizer.sep_token
            sep_idx = tokenized_pattern.index(sep_token)
            keywords_premise = [word for word in tokenized_pattern[:sep_idx]
                                if word not in special_tokens]
            keywords_hypothesis = [word for word in tokenized_pattern[sep_idx+1:]
                                   if word not in special_tokens]
            
            if not keywords_premise and not keywords_hypothesis:
                continue
            
            example_indicies = []
            for i, (prem, hypo) in enumerate(zip(tokenized_premise, tokenized_hypothesis)):
                if set(keywords_hypothesis).issubset(hypo) and set(keywords_premise).issubset(prem):
                    example_indicies.append(i)
                    
            aggr_examples.append(Examples(**vars(pattern),
                                          keywords=keywords_premise + ['<SEP>'] + keywords_hypothesis, 
                                          examples_idx=example_indicies)) 
    
    return aggr_examples

def calculate_generality(aggr_examples, model, tokenizer, dataset, task):
    logger.info('########### Calculating pattern generality ###########')
    logger.info(f'Number of available pattern: {len(aggr_examples)}')
    logger.info(f'Number of corpus: {len(dataset)}')
    logger.info('The estimated run time is quite unstable since it depends on the number of examples in each pattern.')
    
    model.eval()
    
    logger.info(f'Tokenizing dataset into input_ids...')
    if task == 'SA':
        dataset = dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True, return_tensors='pt'), batched=True, batch_size=None)
    elif task == 'NLI':
        dataset = dataset.map(lambda x: tokenizer(x['premise'], x['hypothesis'], padding=True, truncation=True, return_tensors='pt'), batched=True, batch_size=None)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels']) # Roberta doesn't need token_type_ids
    
    for example in tqdm(aggr_examples, desc='Calculating generality'):
        # Skip if there is no example
        if not example.examples_idx:
            continue
        origin_pred = example.origin_pred
        num_same = 0
        
        # Truncate examples to 2500 if the number of examples is too large
        if len(example.examples_idx) >= 2500:
            examples = dataset.select(example.examples_idx[:2500])
            example.isTruncated = True
        else:
            examples = dataset.select(example.examples_idx)
            
        dataloader = DataLoader(examples, batch_size=32)
        
        # Predict answer for each example
        example_preds = []
        scores = []
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f'Predicting example label({len(example.examples_idx)})', leave=False):
                labels = batch.pop('labels').numpy()
                batch = {k: v.to(model.device) for k, v in batch.items()}
                pred = model(**batch).logits.detach().cpu().numpy().argmax(-1)
                num_same += (pred == origin_pred).sum().item()
                scores.append(f1_score(labels, pred, average='macro'))
                example_preds.extend(pred.tolist())
                
        example.generality = num_same / len(examples)
        example.example_f1 = np.mean(scores)
        example.example_preds = example_preds
            
def model_performance(model, tokenizer, dataset):
    dataset = dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True, return_tensors='pt'), batched=True, batch_size=None)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels']) # Roberta doesn't need token_type_ids
    dataLoader = DataLoader(dataset, batch_size=32)
    with torch.inference_mode():
        scores = []
        for batch in tqdm(dataLoader, desc='Evaluating model performance'):
            labels = batch.pop('labels').numpy()
            batch = {k: v.to(model.device) for k, v in batch.items()}
            pred = model(**batch).logits.detach().cpu().numpy().argmax(-1)
            scores.append(f1_score(labels, pred, average='macro'))
    return np.mean(scores)  

def main(args):
    # Setup
    dataset, masked_patern = load_datas(args.pattern_dir, args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(args.device)
    
    # Create corpus that contains tokens in each pattern
    aggr_examples = get_examples(dataset, masked_patern, tokenizer, args.task)
    
    # Calculate generality
    calculate_generality(aggr_examples, model, tokenizer, dataset, args.task)
    
    #score = model_performance(model, tokenizer, dataset)
    #logger.info(f'Corpus F1: {score}')
    
    # Save
    logger.info(f'Saving to {args.output_dir}')
    with open(args.output_dir / 'examples.pkl', 'wb') as f:
        pickle.dump(aggr_examples, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='SA')
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent
    args.pattern_dir = BASE_DIR / 'explainer' / 'pattern' / args.task
    
    args.model_path = BASE_DIR / 'output' / args.task / 'epoch=4'
    
    if args.task == 'SA':
        args.model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        args.tokenizer = 'roberta-base'
    elif args.task == 'NLI':
        args.model_name = 'roberta-large-mnli'
        args.tokenizer = 'roberta-large-mnli'
    else:
        raise ValueError(f'Invalid task: {args.task}')

    output_dir = BASE_DIR / 'explainer' / 'examples' / args.task
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    args.output_dir = output_dir
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)
