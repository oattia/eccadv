#!/bin/sh
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xvzf cifar-10-python.tar.gz
mv cifar-10-batches-py/* ./
rm -r cifar-10-batches-py