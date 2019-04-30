# code2seq
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1gsiqfP1KKVlfekFfQAmXdQAkYhusCt4y/view?usp=sharing)  

A PyTorch re-implementation code for "[code2seq: Generating Sequences from Structured Representations of Code](https://arxiv.org/abs/1808.01400)"

* Paper(Arxiv) : https://arxiv.org/abs/1808.01400  
* Official Github : https://github.com/tech-srl/code2seq

## Requirements
* Python 3.6 +
* Pytorch 1.0 +
* Jupyter-notebook
* scikit-learn
* nltk
* hyperdash (optional)

## TODO
* Beam search is not implemented.

## Memo
* F1 Score is 10.9 on Java-small dataset after 5 epoch training.   
  still unstable. Performance is not well tested due to no enough computer resources... 
