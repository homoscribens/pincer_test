import pickle
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from .data_utils import Examples, MaskedPattern

def generality_hist(fig_dir, examples, threshold=100):
    generality = [e.generality for e in examples if e.generality is not None and e.generality < threshold]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Generality', ylabel='Count', title='Generality distribution')
    ax.hist(generality, bins=100)
    fig.savefig(fig_dir / 'generality_hist.png')
    
def generality_vs_prediction(fig_dir, examples):
    generality = []
    preds = []
    for e in examples:
        if e.generality is not None:
            generality.append(e.generality)
            preds.append(e.origin_pred)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Generality', ylabel='Prediction', title='Generality vs Prediction')
    ax.scatter(generality, preds)
    fig.savefig(fig_dir / 'generality_vs_prediction.png')

def num_examples_hist(fig_dir, examples):
    num_examples = [len(e.examples_idx) for e in examples if e.examples_idx is not None]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Number of examples', ylabel='Count', title='Number of examples distribution')
    ax.hist(num_examples, bins=100, log=True)
    fig.savefig(fig_dir / 'num_examples_hist.png')

def prediction_dist(fig_dir, examples):
    preds = [e.origin_pred for e in examples if e.origin_pred is not None]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Prediction', ylabel='Count', title='Prediction distribution')
    ax.bar(np.arange(0, 3), np.bincount(preds))
    plt.savefig(fig_dir / 'prediction_dist.png')
    
def generality_vs_num_examples(fig_dir, examples):
    generality = []
    num_examples = []
    for e in examples:
        if e.generality is not None:
            generality.append(e.generality)
            num_examples.append(len(e.examples_idx))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, yscale='log', xlabel='Generality', ylabel='Number of examples', title='Generality vs Number of examples')
    ax.scatter(generality, num_examples)
    fig.savefig(fig_dir / 'generality_vs_num_examples.png')

def main(examples,fig_dir):
    generality_hist(fig_dir, examples)
    generality_vs_prediction(fig_dir, examples)
    num_examples_hist(fig_dir, examples)
    prediction_dist(fig_dir, examples)
    generality_vs_num_examples(fig_dir, examples)
    
if __name__ == '__main__':
    BASE = BASE_DIR = pathlib.Path(__file__).parent.parent
    path = BASE / 'explainer' / 'examples' / 'SA' / 'examples.pkl'
    fig_dir = BASE / 'explainer' / 'examples' / 'SA' / 'figs'
    
    if not fig_dir.exists():
        fig_dir.mkdir()
    
    with open(path, 'rb') as f:
        examples = pickle.load(f)
        
    main(examples, fig_dir)
