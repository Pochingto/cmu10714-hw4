# CMU 10-414/10-714 Deep Learning System homework 4

This repo contains all finished homework for the CMU course 10-714 Deep Learning System.

This repo is forked from the `hw4` git repo but contains all the codes I wrote from `hw0-3` as each homework builds on previous work and ultimately leads to a fully functional `PyTorch`-like deep learning library.

**Features and differences with other "neural network from scratch" projects**
- this library implemented the automatic differentiation mechanism
- implemented MLP and some advanced layer with the support of automatic differentiation including CNN, RNN, LSTM, and most other "from scratch" project only implement MLP
- also implemented a C++ and CUDA matrix multiplication runtime for CPU (roughly similar to `numpy`) and GPU

**Key concepts learned in the course**
- automatic differentiation mechanism
- some neural network theoretical insight
- matrix multiplication library implementation (stride and its relation to `reshape`, `broadcast` etc, caching)
- differentiation of popular layers: MLP, CNN, RNN, LSTM, Transformers

Highly recommended!

course website: https://dlsyscourse.org/

assignment URL: https://github.com/dlsyscourse/hw4.git

implementation by a better coder: https://github.com/PKUFlyingPig/CMU10-714.git

I found this course through his awesome cs self-learning guide: https://csdiy.wiki/en/
