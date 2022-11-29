import pathlib
import argparse
import logging

from torch.utils.data import DataLoader

import datasets
from transformers import AutoTokenizer

from model import SentimentClassifier

import wandb
import pytorch_lightning as pl


logging.basicConfig(format='[%(asctime)s] [%(levelname)s] <%(funcName)s> %(message)s',
                    datefmt='%Y/%d/%m %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(model_name):
    logger.info('Loading Tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info('Loading dataset...')
    train_dataset = datasets.load_dataset("glue", "mrpc", split="train")
    val_dataset = datasets.load_dataset("glue", "mrpc", split="test")
    test_dataset = datasets.load_dataset("glue", "mrpc", split="test")

    train_dataset = train_dataset.map(lambda examples: tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        padding="max_length",
        return_tensors='pt'),
                                      batched=True)
    val_dataset = val_dataset.map(lambda examples: tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        padding="max_length",
        return_tensors='pt'),
                                  batched=True)
    test_dataset = test_dataset.map(lambda examples: tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        padding="max_length",
        return_tensors='pt'),
                                    batched=True)

    train_dataset = train_dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
    val_dataset = val_dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

    logger.info('Creating Dataloaders...')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader

def main(args):
    train_loader, val_loader, test_loader = load_dataset(args.model_name)

    model = SentimentClassifier(args.model_name, lr=args.lr)

    checkpoints = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.output_dir,
        filename='{epoch}',
        save_top_k=1,
        every_n_epochs=1,
        mode='min',
    )

    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min',
    )

    logger = pl.loggers.WandbLogger(
        name='NLI',
        project='PincerTest',
        log_model=True,)

    trainer=pl.Trainer(
        accelerator='gpu',
        devices=-1,
        logger=logger,
        precision=args.precision,
        deterministic=True,
        callbacks=[checkpoints, early_stop])

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--patience', type=int, default=3)
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).parent.parent.absolute()
    output_dir = BASE_DIR / 'output'
    if not output_dir.exists():
        output_dir.mkdir()
    args.output_dir = str(output_dir)

    main(args)
    wandb.finish()
