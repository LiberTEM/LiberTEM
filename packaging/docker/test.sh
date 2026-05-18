#!/bin/bash

set -e

trap 'docker compose down' EXIT

./update_reqs.sh
./dist-build.sh
docker compose down
docker compose run --rm tests
