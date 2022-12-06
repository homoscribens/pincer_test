import logging
import argparse
import pathlib
import pickle
import json
import string

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tqdm import tqdm

from models.model import BertClassifier
from .data_utils import Examples, MaskedPattern

logging.basicConfig(format='[%(asctime)s] [%(levelname)s] <%(funcName)s> %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
    

def load_datas(pattern_dir):
    logger.info(f'Loading data...')
    dataset = load_dataset('imdb', split='unsupervised')
    masked_pattern = pickle.load(open(pattern_dir / 'masked_pattern.pkl', 'rb'))
    return dataset, masked_pattern

def remove_stopwords(text):
    stopwords = string.punctuation
    for sw in stopwords:
        text = text.strip(sw)
    return text

def create_corpus(dataset, masked_patern, tokenizer):
    logger.info(f'Creating corpus...')
    logger.info(f'Number of available seed pattern: {len(masked_patern)}')
    
    def tokenize(example):
        example['tokenized'] = tokenizer.tokenize(example['text'])
        return example
    
    aggr_examples = []
    logger.info('Tokenizing dataset...')
    tokenixed_texts = dataset.map(tokenize, batched=False)['tokenized']
    tokenixed_texts = list(map(set, tokenixed_texts))
    stopwords = list(string.punctuation)
    special_tokens = tokenizer.all_special_tokens
    for pattern in tqdm(masked_patern, desc='Collecting examples'):
        key_words = [word for word in tokenizer.tokenize(pattern.masked_sentence)
                     if word not in stopwords+special_tokens]
        if not key_words:
            continue
        text_indicies = []
        for i, text in enumerate(tokenixed_texts):
            if all((kw in text and kw != '') for kw in key_words):
                text_indicies.append(i)
                
        aggr_examples.append(Examples(**vars(pattern), 
                                      keywords=key_words, 
                                      examples_idx=text_indicies)) 
    return aggr_examples

def calculate_generality(aggr_examples, model, tokenizer, dataset):
    logger.info('########### Calculating pattern generality ###########')
    logger.info(f'Number of available pattern: {len(aggr_examples)}')
    logger.info(f'Number of corpus: {len(dataset)}')
    logger.info('The estimated run time is quite unstable since it depends on the number of examples in each pattern.')
    
    model.eval()
    logger.info(f'Tokenizing dataset into input_ids...')
    dataset = dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True, return_tensors='pt'), batched=True, batch_size=None)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask']) # Roberta doesn't need token_type_ids
    for example in tqdm(aggr_examples, desc='Calculating generality'):
        if not example.examples_idx:
            continue
        origin_pred = example.origin_pred
        num_same = 0
        examples = dataset.select(example.examples_idx)
        dataloader = DataLoader(examples, batch_size=32)
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc='Predicting for examples', leave=False):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                pred = model(**batch).logits.detach().cpu().numpy().argmax(-1)
                num_same += (pred == origin_pred).sum().item()
            example.generality = num_same / len(example.examples_idx)

def main(args):
    # Setup
    dataset, masked_patern = load_datas(args.pattern_dir)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.to(args.device)
    
    # Create corpus that contains tokens in each pattern
    aggr_examples = create_corpus(dataset, masked_patern, tokenizer)
    
    # Calculate generality
    calculate_generality(aggr_examples, model, tokenizer, dataset)
    
    # Save
    logger.info(f'Saving to {args.output_dir}')
    with open(args.output_dir / 'examples.pkl', 'wb') as f:
        pickle.dump(aggr_examples, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='cardiffnlp/twitter-roberta-base-sentiment')
    parser.add_argument('--task', type=str, default='SA')
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent
    args.pattern_dir = BASE_DIR / 'explainer' / 'pattern' / args.task
    
    args.model_path = BASE_DIR / 'output' / args.task / 'epoch=4'

    output_dir = BASE_DIR / 'explainer' / 'examples' / args.task
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    args.output_dir = output_dir
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)
