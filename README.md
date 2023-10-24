# Discovering Highly Influential Shortcut Reasoning: An Automated Template-Free Approach

## Abstract
Shortcut reasoning is an irrational process of inference, which degrades the robustness of an NLP model.
While a number of previous work has tackled the identification of shortcut reasoning, there are still two major limitations: (i) a method for quantifying the severity of the discovered shortcut reasoning is not provided; (ii) certain types of shortcut reasoning may be missed.
To address these issues, we propose a novel method for identifying shortcut reasoning.
The proposed method quantifies the severity of the shortcut reasoning by leveraging out-of-distribution data and does not make any assumptions about the type of tokens triggering the shortcut reasoning.
Our experiments on Natural Language Inference and Sentiment Analysis demonstrate that our framework successfully discovers known and unknown shortcut reasoning in the previous work.

## Requirements
```linux
$ conda env create --file requirements.yaml
```

## Run
To execute Input Reduction, run the following command:
```linux
$ python -m explainer.input_reduction_ig_approx --task {task}
```
and to execute calculate Generality, run the following command:
```linux
$ python -m explainer.calculate_generality --task {task}
```
and to identify Shortcut Reasoning, run the following command:
```linux
$ python -m explainer.check --task {task}
```
where `{task}` is one of `nli` or `sa`.

## Citation
```
@inproceedings{
anonymous2023discovering,
title={Discovering Highly Influential Shortcut Reasoning: An Automated Template-Free Approach},
author={Anonymous},
booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
url={https://openreview.net/forum?id=czxX6jjpVJ}
}
```
