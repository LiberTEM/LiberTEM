#!/usr/bin/env bash

set -e
BASE_DIR=$(dirname "$(readlink -f "${0}")")
cd "$BASE_DIR"

if [ -n "$DOCKER_ACCESS_TOKEN" ] \
    && [ "$DOCKER_ACCESS_TOKEN" != '$(DOCKER_ACCESS_TOKEN)' ] \
    && [ -n "$DOCKER_USER" ] && [ "$DOCKER_USER" != '$(DOCKER_USER)' ]
then
    export DOCKER_BUILDKIT=1
    docker login -u "$DOCKER_USER" --password-stdin <<< "$DOCKER_ACCESS_TOKEN"
    docker push libertem/libertem-dev
    docker push libertem/libertem:triage
else
    echo "not authenticated, skipping"
fi
