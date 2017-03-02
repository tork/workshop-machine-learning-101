#!/usr/bin/env bash

docker-machine --version &> /dev/null
if [ $? -ne 0 ]; then
    echo "Failed to run docker-machine --version. Is Docker Toolbox properly installed?"
    exit
fi

MACHINE=ml101
docker-machine inspect $MACHINE &> /dev/null
if [ $? -ne 0 ]; then
    echo Creating docker machine $MACHINE
    docker-machine create --driver virtualbox $MACHINE
fi

eval $(docker-machine env $MACHINE)

SRC_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$SRC_PATH/env-native.sh
