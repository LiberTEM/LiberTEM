FROM debian:stable-slim AS builder

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip \
    git

RUN python3.7 -m venv /venv/
COPY packaging/docker/requirements.txt /requirements.txt
COPY test_requirements.txt /test_requirements.txt
RUN /venv/bin/pip install -U pip
RUN /venv/bin/pip install --no-cache wheel && /venv/bin/pip install --no-cache -r /requirements.txt -r /test_requirements.txt
RUN /venv/bin/pip install -e 'git+https://github.com/sk1p/ipyparallel.git@fix-become-dask#egg=ipyparallel'

COPY . /code/
RUN venv/bin/pip install --no-cache /code/

FROM debian:stable-slim

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip

COPY --from=builder /venv/ /venv/
COPY --from=builder /code/ /code/

ENV PYTHONUNBUFFERED 1

CMD ["/venv/bin/libertem-server"]
