#!/usr/bin/env bash
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && \
mkdir -p data && cd data && \
echo downloading datasets && \

function fetch {
    mkdir -p $2 && cd $2 && \
    curl -O $1 && \
    if [ -f *.gz ]; then gunzip *.gz; fi && \
    cd ..
}

fetch http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv titanic && \
fetch http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz mnist && \
fetch http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz mnist && \
fetch http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz mnist && \
fetch http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz mnist && \

echo done || echo oops
