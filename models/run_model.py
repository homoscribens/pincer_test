import pathlib
import argparse
import logging
import random

import numpy as np

from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tqdm import tqdm


logging.basicConfig(format='[%(asctime)s] [%(levelname)s] <%(funcName)s> %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def load_dataset(tokenizer, task):
    logger.info('Loading dataset...')
    if task =='SA':
        ds = datasets.load_dataset('imdb', split='train')
        ds = ds.train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
        test_dataset = datasets.load_dataset('imdb', split='test')

        def tokenize(examples):
            encoded = tokenizer(examples['text'], 
                                truncation=True,
                                padding="max_length", 
                                return_tensors='pt')
            return encoded

        ds = ds.map(tokenize, batched=True)
        train_dataset = ds['train']
        val_dataset = ds['test']
        test_dataset = test_dataset.map(tokenize, batched=True)
    elif task == 'NLI':
        ds = datasets.load_dataset('snli', split='train')
        ds = ds.train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
        test_dataset = datasets.load_dataset('snli', split='test')

        def tokenize(examples):
            encoded = tokenizer(examples['premise'], 
                                examples['hypothesis'],
                                truncation=True,
                                padding="max_length", 
                                return_tensors='pt')
            return encoded

        ds = ds.map(tokenize, batched=True)
        train_dataset = ds['train']
        val_dataset = ds['test']
        test_dataset = test_dataset.map(tokenize, batched=True)
    else:
        raise ValueError('Task not supported')

    train_dataset = train_dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
    val_dataset = val_dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

    logger.info('Creating Dataloaders...')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=52, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=52, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=52, pin_memory=True)
    return train_loader, val_loader, test_loader

def calculate_loss_and_accuracy(model, loader, device):
    model.eval()
    valid_loss = []
    score = []
    with torch.no_grad():
        for data in tqdm(loader, desc='valid'):
            batch = {k: v.to(device) for k, v in data.items()}

            outputs = model(**batch)

            loss = outputs.loss.detach().cpu().numpy()
            valid_loss.append(loss)
            
            pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            labels = labels.detach().cpu().numpy()
            score.append(f1_score(labels, pred))
            
    return np.mean(valid_loss), np.mean(score)

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device='cuda'):
    logger.info(f'TRAINING MODEL num_epoch: {num_epochs}')
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}')

        loss_train = []
        model.train()
        for data in tqdm(train_loader, desc='train'):
            batch = {k: v.to(device) for k, v in data.items()}

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            loss_train.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()

        loss_train = np.mean(loss_train)
        loss_valid, f1_valid = calculate_loss_and_accuracy(model, val_loader, device)
        
        logger.info(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid}, f1_valid: {f1_valid}') 

        logger.info('Saving model...')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint_epoch={epoch + 1}.pt')

def main(args):
    logger.info('Loading Tokenizer and Model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model = model.to(args.device)

    train_loader, val_loader, test_loader = load_dataset(tokenizer, args.task)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    train_model(model, train_loader, val_loader, optimizer, args.epochs, device=args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task", default='SA', type=str)
    # Training arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
    output_dir = BASE_DIR / 'output'
    if not output_dir.exists():
        output_dir.mkdir()
    args.output_dir = str(output_dir)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch_fix_seed(args.seed)

    main(args)
