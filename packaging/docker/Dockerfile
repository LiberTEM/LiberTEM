# syntax=docker/dockerfile:1.0.0-experimental
FROM python:3.9-slim-buster AS builder

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip \
    git

RUN python3.9 -m venv /venv/
# COPY packaging/docker/requirements.txt /requirements.txt
COPY test_requirements.txt /test_requirements.txt
RUN /venv/bin/pip install -U pip
RUN /venv/bin/pip install --no-cache wheel && /venv/bin/pip install --no-cache -r /test_requirements.txt
# Include https://github.com/ipython/ipyparallel/pull/415
# FIXME switch to release version as soon as included
RUN /venv/bin/pip install ipyparallel

COPY . /code/
RUN venv/bin/pip install --no-cache /code/

FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip

COPY --from=builder /venv/ /venv/
COPY --from=builder /code/ /code/

ENV PYTHONUNBUFFERED 1

CMD ["/venv/bin/libertem-server"]
