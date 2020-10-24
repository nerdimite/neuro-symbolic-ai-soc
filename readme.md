# Neuro-Symbolic AI for Visual Question Answering
## Sort-of-CLEVR Dataset

Neuro-Symbolic AI allows us to combine Deep Learning’s superior pattern recognition abilities with the reasoning abilities of symbolic methods like program synthesis. This repository is an implementation of NSAI for Visual Question Answering on the Sort-of-CLEVR dataset using PyTorch. This implementation is inspired by the [Neuro-Symbolic VQA](https://arxiv.org/abs/1810.02338) paper by MIT-IBM Watson AI Lab.

The basic idea behind using NSAI for VQA is parsing the visual scene into a symbolic representation and using NLP to parse the query into an executable program which the program executor can use on the scene to find the answer. This implementation gets more than 99% on the Sort-of-CLEVR dataset.

## Requirements
- PyTorch version 1.2 and above
- OpenCV
- dlib
- Scikit Learn
- Pandas
- Numpy

## Usage
- The Step-by-Step usage is in the [NSAI on Sort-of-CLEVR.ipynb](NSAI on Sort-of-CLEVR.ipynb) notebook from training the individual modules to plugging everything together to test it.
- You can easily run this repository using Colab <a href="https://colab.research.google.com/github/nerdimite/neuro-symbolic-ai-soc/blob/master/NSAI%20on%20Sort-of-CLEVR.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>
- To understand more about the design and workflow, check out [NSAI Flow Diagram.pdf](NSAI Flow Diagram.pdf) which contains the workflows of every component i.e. Perception Module, Semantic Parser and Program Executor.

## References
- Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding https://arxiv.org/abs/1810.02338
- The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision https://arxiv.org/abs/1904.12584
- https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py
- https://www.learnopencv.com/training-a-custom-object-detector-with-dlib-making-gesture-controlled-applications/
