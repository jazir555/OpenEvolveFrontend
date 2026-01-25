#!/bin/bash

pushd kimina-lean-server

mkdir -p logs
TIME=$(date "+%Y-%m-%d-%H-%M-%S")
SERVER_LOG_PATH="logs/verification_${TIME}.log"
python -m server > $SERVER_LOG_PATH 2>&1 &

popd
