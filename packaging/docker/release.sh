#!/bin/env bash
if [ -n "$DOCKER_ACCESS_TOKEN" ] && [ -n "$DOCKER_USER" ]
then
    cd packaging/docker/ || exit
    TAGS=$( python3 ../../scripts/release docker-tags )
    CONT=libertem/libertem:continuous
    if [ -n "$TAGS" ]
    then
        export DOCKER_BUILDKIT=1
        docker login -u "$DOCKER_USER" --password-stdin <<< "$DOCKER_ACCESS_TOKEN"
        ./update_reqs.sh
        docker pull $CONT || true
        docker build --cache-from=$CONT -t $CONT --build-arg BUILDKIT_INLINE_CACHE=1 ../../ -f Dockerfile
        docker push $CONT
        for TAG in $TAGS
        do
            echo "tagging and pushing $TAG"
            docker tag $CONT "libertem/libertem:$TAG"
            docker push "libertem/libertem:$TAG"
        done
    else
        echo "Only continuous release, skipping"    
    fi
else
    echo "not authenticated, skipping"
fi