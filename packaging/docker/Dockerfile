FROM debian:stable-slim AS builder

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip

RUN python3.7 -m venv /venv/
COPY packaging/docker/requirements.txt /requirements.txt
RUN /venv/bin/pip install --no-cache wheel && /venv/bin/pip install --no-cache -r /requirements.txt

COPY . /code/
RUN venv/bin/pip install --no-cache /code/

FROM debian:stable-slim

RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip

COPY --from=builder /venv/ /venv/

ENV PYTHONUNBUFFERED 1

CMD ["/venv/bin/libertem-server"]
