#!/bin/env bash

cd packaging/docker/ || exit
export DOCKER_BUILDKIT=1

DEV=libertem/libertem-dev
TRIAGE=libertem/libertem:triage

./update_reqs.sh

docker pull python:3.9-slim
docker pull $TRIAGE || true
docker pull $DEV || true
docker build --cache-from=$TRIAGE --cache-from=$DEV -t $TRIAGE --build-arg BUILDKIT_INLINE_CACHE=1 ../../ -f Dockerfile
docker build --cache-from=$TRIAGE --cache-from=$DEV -t $DEV --build-arg BUILDKIT_INLINE_CACHE=1 --build-arg dev=1 ../../ -f Dockerfile
