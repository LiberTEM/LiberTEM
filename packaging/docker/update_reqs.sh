#!/bin/bash
pip-compile --no-emit-index-url ../../pyproject.toml -Uo requirements.txt
