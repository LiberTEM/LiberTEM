#!/usr/bin/env bash

set -e
BASE_DIR=$(dirname "$(readlink -f "${0}")")
cd "$BASE_DIR"

if [ -n "$DOCKER_ACCESS_TOKEN" ] \
    && [ "$DOCKER_ACCESS_TOKEN" != '$(DOCKER_ACCESS_TOKEN)' ] \
    && [ -n "$DOCKER_USER" ] && [ "$DOCKER_USER" != '$(DOCKER_USER)' ]
then
    TAGS=$( python3 ../../scripts/release docker-tags )
    CONT=libertem/libertem:continuous
    TRIAGE=libertem/libertem:triage
    if [ -n "$TAGS" ]
    then
        export DOCKER_BUILDKIT=1
        docker login -u "$DOCKER_USER" --password-stdin <<< "$DOCKER_ACCESS_TOKEN"
        ./update_reqs.sh
        docker pull $TRIAGE || true
        docker build --cache-from=$TRIAGE -t $CONT --build-arg BUILDKIT_INLINE_CACHE=1 ../../ -f Dockerfile
        for TAG in $TAGS
        do
            echo "tagging and pushing $TAG"
            docker tag $CONT "libertem/libertem:$TAG"
            docker push "libertem/libertem:$TAG"
        done
    else
        echo "No release tags, skipping"
    fi
else
    echo "not authenticated, skipping"
fi