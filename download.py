# Download the datasets you will be using for this assignment

import urllib.request
import tarfile
import os

# !mkdir -p './data/ptb'
# Download Penn Treebank dataset
folders = ['./data', './data/ptb']
for folder in folders:
    if not os.path.exists(folder):
        os.mkdir(folder)

ptb_data = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
for f in ['train.txt', 'test.txt', 'valid.txt']:
    if not os.path.exists(os.path.join('./data/ptb', f)):
        urllib.request.urlretrieve(ptb_data + f, os.path.join('./data/ptb', f))

# Download CIFAR-10 dataset
if not os.path.isdir("./data/cifar-10-batches-py"):
    urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "./data/cifar-10-python.tar.gz")
    cifar10_tar = tarfile.open('./data/cifar-10-python.tar.gz')
    cifar10_tar.extractall('./data')
    cifar10_tar.close()
    # !tar -xvzf './data/cifar-10-python.tar.gz' -C './data'