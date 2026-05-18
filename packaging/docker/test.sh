#!/bin/bash

set -e

trap 'docker compose down' EXIT

./update_reqs.sh
docker compose build
docker compose run tests
