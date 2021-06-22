# syntax=docker/dockerfile:1.0.0-experimental
FROM python:3.9-slim AS builder

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip \
    git \
    libfftw3-dev

RUN python3.9 -m venv /venv/
# Update requirements.txt with ./update_reqs.sh
COPY packaging/docker/requirements.txt /requirements.txt
COPY test_requirements.txt /test_requirements.txt
RUN /venv/bin/pip install -U pip
RUN /venv/bin/pip install --no-cache wheel && /venv/bin/pip install --no-cache -r /requirements.txt -r /test_requirements.txt
# Include pyproject.toml to allow building for Python 3.9
# https://github.com/pyFFTW/pyFFTW/pull/226
# FIXME switch to release version as soon as included
RUN /venv/bin/pip install -e 'git+https://github.com/pyFFTW/pyFFTW.git#egg=pyfftw'
# Include https://github.com/ipython/ipyparallel/pull/415
# FIXME switch to release version as soon as included
RUN /venv/bin/pip install -e 'git+https://github.com/ipython/ipyparallel.git@3638a89d7eb6b4969ed0a6f44a4e8b234d574005#egg=ipyparallel'

COPY . /code/
RUN venv/bin/pip install --no-cache /code/

FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip \
    libfftw3-dev

COPY --from=builder /venv/ /venv/
COPY --from=builder /code/ /code/

ENV PYTHONUNBUFFERED 1

CMD ["/venv/bin/libertem-server"]
