# syntax=docker/dockerfile:1.0.0-experimental
FROM python:3.9-slim AS builder0

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip \
    git \
    libfftw3-dev

FROM builder0 as builder

RUN python3.9 -m venv /venv/
# Update requirements.txt with ./update_reqs.sh
COPY packaging/docker/requirements.txt /requirements.txt
RUN /venv/bin/pip install -U pip
RUN /venv/bin/pip install --no-cache wheel && /venv/bin/pip install --no-cache -r /requirements.txt

# Include pyproject.toml to allow building for Python 3.9
# https://github.com/pyFFTW/pyFFTW/pull/226
# FIXME switch to release version as soon as included
RUN /venv/bin/pip install -e 'git+https://github.com/pyFFTW/pyFFTW.git#egg=pyfftw'

COPY . /code/
RUN venv/bin/pip install --no-cache /code/

# to share the "apt-get install" call between stages, we inherit from
# "builder0" instead of the base image here:
FROM builder0

COPY --from=builder /venv/ /venv/
COPY --from=builder /code/ /code/

ENV PYTHONUNBUFFERED 1

EXPOSE 9000

CMD ["/venv/bin/libertem-server"]

# vim:ft=dockerfile
