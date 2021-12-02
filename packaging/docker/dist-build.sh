#!/bin/env bash

cd packaging/docker/ || exit
export DOCKER_BUILDKIT=1

DEV=libertem/libertem-dev
CONT=libertem/libertem:continuous

./update_reqs.sh

docker pull $CONT || true
docker pull $DEV || true
docker build --cache-from=$CONT --cache-from=$DEV -t $CONT --build-arg BUILDKIT_INLINE_CACHE=1 ../../ -f Dockerfile
docker build --cache-from=$CONT --cache-from=$DEV -t $DEV --build-arg BUILDKIT_INLINE_CACHE=1 --build-arg dev=1 ../../ -f Dockerfile
