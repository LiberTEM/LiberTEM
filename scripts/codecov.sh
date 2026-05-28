#!/bin/bash

set -e

# pin the version so we can compare sha
curl -Os https://uploader.codecov.io/v0.8.0/linux/codecov
echo 'b37359013b48fbc3b0790d59fc474a52a260fb96e28e1b2c2ae001dc9b9cc996 codecov' | sha256sum -c

chmod +x codecov
./codecov $*
