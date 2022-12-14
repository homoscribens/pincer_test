import pickle
import pathlib
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
    
def example_prediction_dist(fig_dir, examples):
    preds = [e.example_preds for e in examples if e.example_preds is not None]
    preds = [p for ps in preds for p in ps]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Prediction', ylabel='Count', title='Example prediction distribution')
    ax.bar(np.arange(0, 3), np.bincount(preds))
    plt.savefig(fig_dir / 'example_prediction_dist.png')
    
def generality_vs_num_examples(fig_dir, examples, task):
    gen_pos, gen_neu, gen_neg = [], [], []
    num_pos, num_neu, num_neg = [], [], []
    for e in examples:
        if e.generality is not None:
            if e.origin_pred == 2:
                gen_pos.append(e.generality)
                num_pos.append(len(e.examples_idx))
            elif e.origin_pred == 1:    
                gen_neu.append(e.generality)
                num_neu.append(len(e.examples_idx))
            else:
                gen_neg.append(e.generality)
                num_neg.append(len(e.examples_idx))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, yscale='log', xlabel='Generality', ylabel='Number of examples', title='Generality vs Number of examples')
    ax.scatter(gen_pos, num_pos)
    ax.scatter(gen_neu, num_neu)
    ax.scatter(gen_neg, num_neg)
    if task == 'SA':
        fig.legend(['Positive', 'Neutral', 'Negative'])
    elif task == 'NLI':
        fig.legend(['Entailment', 'Neutral', 'Contradiction'])
    fig.savefig(fig_dir / 'generality_vs_num_examples.png')
    
def generality_vs_example_f1(fig_dir, examples):
    generality = []
    f1 = []
    for e in examples:
        if e.generality is not None:
            if len(e.example_preds) >= 10:
                generality.append(e.generality)
                f1.append(e.example_f1)
    
    clf = LinearRegression()
    clf.fit(np.array(generality).reshape(-1, 1), np.array(f1))
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='Generality', ylabel='F1', title='Generality vs Example F1')
    ax.scatter(generality, f1)
    ax.plot(np.array(generality).reshape(-1, 1), clf.predict(np.array(generality).reshape(-1, 1)), color='red')
    fig.savefig(fig_dir / 'generality_vs_example_f1.png')
    
def create_dataframes(examples):
    df_list = []
    for e in examples:
        if e.generality is not None:
            preds_dist = np.bincount(e.example_preds)
            if len(preds_dist) == 2:
                preds_dist = np.append(preds_dist, 0)
            elif len(preds_dist) == 1:
                preds_dist = np.append(preds_dist, [0, 0])
            else:
                pass
            
            df_list.append({'keywords': [kw.replace('Ġ', '') for kw in e.keywords],
                            'pattern': e.masked_sentence,
                            'label': e.origin_pred,
                            'gold_label': e.gold_label,
                            'correct': True if e.origin_pred == e.gold_label else False,
                            'generality': e.generality, 
                            'num_examples': len(e.examples_idx),
                            'truncate': e.isTruncated,
                            'example_f1': e.example_f1,
                            'pred_0 %': preds_dist[0] / len(e.example_preds),
                            'pred_1 %': preds_dist[1] / len(e.example_preds),
                            'pred_2 %': preds_dist[2] / len(e.example_preds)})
            
    df = pd.DataFrame(df_list)
    return df

def glance_at_data(examples):
    for e in examples:
        if e.generality is not None:
            if e.generality >=0.7 and len(e.examples_idx) > 100:
                print([kw.replace('Ġ', '') for kw in e.keywords])
                print(e.generality) 
                print(len(e.examples_idx))
                print(e.origin_pred)
                print('-----------------')

def main(args, examples):
    # glance_at_data(examples)
    generality_hist(args.fig_dir, examples)
    generality_vs_prediction(args.fig_dir, examples)
    num_examples_hist(args.fig_dir, examples)
    prediction_dist(args.fig_dir, examples)
    example_prediction_dist(args.fig_dir, examples)
    generality_vs_num_examples(args.fig_dir, examples, args.task)
    generality_vs_example_f1(args.fig_dir, examples)
    
    df = create_dataframes(examples)
    df.to_csv(args.fig_dir / 'data.csv')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='SA')
    args = parser.parse_args()
    BASE = BASE_DIR = pathlib.Path(__file__).parent.parent
    path = BASE / 'explainer' / 'examples' / args.task / 'examples.pkl'
    fig_dir = BASE / 'explainer' / 'examples' / args.task / 'figs'
    
    if not fig_dir.exists():
        fig_dir.mkdir()
    args.fig_dir = fig_dir
    
    with open(path, 'rb') as f:
        examples = pickle.load(f)
        
    main(args, examples)
