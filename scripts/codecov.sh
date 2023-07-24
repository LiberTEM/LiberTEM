#!/bin/bash

set -e

# pin the version so we can compare sha
curl -Os https://uploader.codecov.io/v0.6.1/linux/codecov
echo '0c9b79119b0d8dbe7aaf460dc3bd7c3094ceda06e5ae32b0d11a8ff56e2cc5c5 codecov' | sha256sum -c

chmod +x codecov
./codecov $*
