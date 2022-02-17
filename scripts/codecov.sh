#!/bin/bash

set -e

# pin the version so we can compare sha
curl -Os https://uploader.codecov.io/v0.1.17/linux/codecov
echo 'ca88335829e3a5b589674a200fdd1dae8f2ef27775647bc3aef6677266a6fda2 codecov' | sha256sum -c

chmod +x codecov
./codecov $*
