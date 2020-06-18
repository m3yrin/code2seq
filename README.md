# code2seq [WIP]

A PyTorch re-implementation code for "[code2seq: Generating Sequences from Structured Representations of Code](https://arxiv.org/abs/1808.01400)"

* Paper(Arxiv) : https://arxiv.org/abs/1808.01400  
* Official Github : https://github.com/tech-srl/code2seq

## Requirements
Please see requirements.txt

## Usage
* notebooks/preparation.ipynb 
for downloading dataset, making some directories etc.

* notebooks/code2seq.ipynb
for training and evaluating the model.

## Memo
* Beam search is not implemented.
* GCP AI Platform Notebooks is used to train model.
* AI Platform Notebooks requires google_compute_engine api so please install this before installing other packages if you use AI Platform Notebooks.
