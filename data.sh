#!/usr/bin/env bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && \
mkdir -p data && cd data && \
echo downloading datasets && \

function fetch_gz {
    mkdir -p $2 && cd $2 && \
    curl -O $1 && \
    gunzip *.gz && \
    cd ..
}

fetch_gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz mnist && \
fetch_gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz mnist && \
fetch_gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz mnist && \
fetch_gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz mnist && \

echo done || echo oops
