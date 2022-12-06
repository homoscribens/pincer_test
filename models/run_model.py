import os
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

from .model import BertClassifier


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(format='[%(asctime)s] [%(levelname)s] <%(funcName)s> %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

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

def load_dataset(tokenizer, task, data_dir):
    logger.info('Loading dataset...')
    if data_dir.exists():
        logger.info(f'Loading cached dataset from {data_dir}')
        train_dataset = datasets.load_from_disk(data_dir / 'train')
        val_dataset = datasets.load_from_disk(data_dir / 'val')
        test_dataset = datasets.load_from_disk(data_dir / 'test')
    else:
        if task =='SA':
            dataset = datasets.load_dataset('glue', 'sst2', split='train')
            dataset = dataset.train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
            val_dataset = datasets.load_dataset('glue', 'sst2', split='validation')

            def tokenize(examples):
                encoded = tokenizer(examples['sentence'], 
                                    truncation=True,
                                    padding=True, 
                                    return_tensors='pt')
                return encoded

            dataset = dataset.map(tokenize, batched=True, batch_size=None)
            train_dataset = dataset['train']
            test_dataset = dataset['test']
            val_dataset = val_dataset.map(tokenize, batched=True, batch_size=None)
        elif task == 'NLI':
            ds = datasets.load_dataset('snli', split='train')
            ds = ds.train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
            test_dataset = datasets.load_dataset('snli', split='test')

            def tokenize(examples):
                encoded = tokenizer(examples['premise'], 
                                    examples['hypothesis'],
                                    truncation=True,
                                    padding=True, 
                                    return_tensors='pt')
                return encoded

            ds = ds.map(tokenize, batched=True, batch_size=None)
            train_dataset = ds['train']
            val_dataset = ds['test']
            test_dataset = test_dataset.map(tokenize, batched=True, batch_size=None)
        else:
            raise ValueError('Task not supported')

        train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
        test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])

        logger.info(f'Saving dataset to {data_dir}')
        data_dir.mkdir(parents=True)
        train_dataset.save_to_disk(data_dir / 'train')
        val_dataset.save_to_disk(data_dir / 'val')
        test_dataset.save_to_disk(data_dir / 'test')

    logger.info('Creating Dataloaders...')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=52, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=52, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=52, pin_memory=True)

    num_labels = len(train_dataset.features['label'].names)
    return train_loader, val_loader, test_loader, num_labels

def calculate_loss_and_accuracy(model, loader, device='cuda'):
    model.eval()
    valid_loss = []
    score = []
    with torch.no_grad():
        for data in tqdm(loader, desc='valid'):
            batch = {k: v.to(device) for k, v in data.items()}

            outputs = model(**batch)

            loss = outputs.loss.detach().cpu().numpy()
            valid_loss.append(loss)
            
            pred = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
            labels = batch.pop('labels').detach().cpu().numpy()
            score.append(f1_score(labels, pred))
            
    return np.mean(valid_loss), np.mean(score)

def train_model(model, train_loader, val_loader, optimizer, num_epochs, output_dir, task, device='cuda'):
    logger.info(f'TRAINING MODEL num_epoch: {num_epochs}, task: {task}')
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
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        epoch_output_dir = output_dir / task / f'epoch={epoch}'
        if not epoch_output_dir.exists():
            epoch_output_dir.mkdir(parents=True)

        # torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint_epoch={epoch + 1}.pt')

        output_model_file = str(epoch_output_dir / WEIGHTS_NAME)
        output_config_file = str(epoch_output_dir / CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

def main(args):
    logger.info('Loading Tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader, test_loader, num_labels = load_dataset(tokenizer, args.task, args.data_dir)

    logger.info('Loading Model...')
    model = BertClassifier.from_pretrained(args.model_name, num_labels=num_labels)
    model = model.to(args.device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    # train_model(model, train_loader, val_loader, optimizer, num_epochs=args.epochs, output_dir=args.output_dir, task=args.task, device=args.device)

    logger.info('Evaluating model...')
    trained_model_path = args.output_dir / args.task / f'epoch={4}'
    trained_model = BertClassifier.from_pretrained(trained_model_path, num_labels=num_labels)
    trained_model = trained_model.to(args.device)
    _, f1_test = calculate_loss_and_accuracy(trained_model, test_loader, device=args.device)
    logger.info(f'f1_test: {f1_test}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--task', default='SA', type=str)
    # Training arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
    output_dir = BASE_DIR / 'output'
    args.output_dir = output_dir

    data_dir = BASE_DIR / 'data' / args.task
    args.data_dir = data_dir

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch_fix_seed(args.seed)

    main(args)
