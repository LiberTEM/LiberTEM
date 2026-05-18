# syntax=docker/dockerfile:1
FROM python:3.14-slim AS builder0

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    apt-get update && apt-get install -y \
    python3-venv \
    python3-pip \
    git

FROM builder0 as builder

RUN python3.14 -m venv /venv/
# Update requirements.txt with ./update_reqs.sh
COPY packaging/docker/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache /venv/bin/pip install uv pip
RUN --mount=type=cache,target=/root/.cache /venv/bin/uv pip install --python /venv/bin/python wheel && \
    /venv/bin/uv pip install --python /venv/bin/python -r /requirements.txt

# Only install test requirements if dev is set
# docker build (...) --build-arg dev=1 (...) for dev build
ARG dev
COPY test_requirements.txt /test_requirements.txt
RUN --mount=type=cache,target=/root/.cache if [ -z "$dev" ] ; then echo Skipping test requirements ; else \
    /venv/bin/uv pip install --python /venv/bin/python -r /test_requirements.txt && /venv/bin/uv pip install --python /venv/bin/python ipyparallel ; fi


# to share the "apt-get install" call between stages, we inherit from
# "builder0" instead of the base image here:
FROM builder0

COPY --from=builder /venv/ /venv/

# Only here so we have only a small delta when the code changes,
# not the entire venv
COPY . /code/
RUN --mount=type=cache,target=/root/.cache /venv/bin/uv pip install --python /venv/bin/python /code/

ENV PYTHONUNBUFFERED 1

# LiberTEM server
EXPOSE 9000
# Dask.distributed scheduler
EXPOSE 8786
# Dask.distributed dashboard
EXPOSE 8787


CMD ["/venv/bin/libertem-server", "--host", "0.0.0.0", "--insecure", "--port", "9000"]

# vim:ft=dockerfile
