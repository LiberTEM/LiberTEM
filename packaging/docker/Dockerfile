# syntax=docker/dockerfile:1.0.0-experimental
FROM libertem/libertem-dev:base as builder

COPY . /code/
RUN venv/bin/pip install --no-cache /code/

FROM libertem/libertem-dev:base

COPY --from=builder /venv/ /venv/
COPY --from=builder /code/ /code/

ENV PYTHONUNBUFFERED 1

# LiberTEM server
EXPOSE 9000
# Dask.distributed scheduler
EXPOSE 8786
# Dask.distributed dashboard
EXPOSE 8787


CMD ["/venv/bin/libertem-server", "--host", "0.0.0.0", "--insecure", "--port", "9000"]

# vim:ft=dockerfile
