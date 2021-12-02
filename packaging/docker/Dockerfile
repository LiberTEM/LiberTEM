# syntax=docker/dockerfile:1
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

# Only install test requirements if dev is set
# docker build (...) --build-arg dev=1 (...) for dev build
ARG dev
COPY test_requirements.txt /test_requirements.txt
RUN if [ -z "$dev" ] ; then echo Skipping test requirements ; else \
/venv/bin/pip install --no-cache -r /test_requirements.txt && /venv/bin/pip install ipyparallel ; fi


# to share the "apt-get install" call between stages, we inherit from
# "builder0" instead of the base image here:
FROM builder0

COPY --from=builder /venv/ /venv/

# Only here so we have only a small delta when the code changes,
# not the entire venv
COPY . /code/
RUN venv/bin/pip install --no-cache /code/

ENV PYTHONUNBUFFERED 1

# LiberTEM server
EXPOSE 9000
# Dask.distributed scheduler
EXPOSE 8786
# Dask.distributed dashboard
EXPOSE 8787


CMD ["/venv/bin/libertem-server", "--host", "0.0.0.0", "--insecure", "--port", "9000"]

# vim:ft=dockerfile
