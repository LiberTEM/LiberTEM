#!/bin/bash

set -e

PROJECT=libertem-test-$(whoami)

trap "docker compose -p $PROJECT down --volumes" EXIT

./update_reqs.sh
./dist-build.sh
docker compose -p $PROJECT down --volumes
docker compose -p $PROJECT run --rm tests
