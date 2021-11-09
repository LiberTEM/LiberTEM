#!/bin/bash

BASE=$(dirname $(realpath -s "$0"))

FILE_LIST=$(cat "$BASE/../.mypy-checked")

mypy "$FILE_LIST"
