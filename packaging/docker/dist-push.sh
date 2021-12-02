#!/bin/env bash
if [ -n "$DOCKER_ACCESS_TOKEN" ] \
    && [ "$DOCKER_ACCESS_TOKEN" != '$(DOCKER_ACCESS_TOKEN)' ] \
    && [ -n "$DOCKER_USER" ] && [ "$DOCKER_USER" != '$(DOCKER_USER)' ]
then
    cd packaging/docker/ || exit
    export DOCKER_BUILDKIT=1
    docker login -u "$DOCKER_USER" --password-stdin <<< "$DOCKER_ACCESS_TOKEN"
    docker push libertem/libertem-dev
    docker push libertem/libertem:continuous
else
    echo "not authenticated, skipping"
fi
