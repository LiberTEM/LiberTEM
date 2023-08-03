# Container images

The Dockerfile here allows to build two different kinds of images:

- `dev=0` (default): production image
    - tagged in CI as `ghcr.io/libertem/libertem` with tags `continuous` for the master branch, `latest` for the latest stable version, and version tags for stable releases
    - also tagged as `ghcr.io/libertem/libertem-triage:pre-{sha}` as a temporary image while building
- `dev=1`: installs test requirements into the image
    - tagged in CI as `ghcr.io/libertem/libertem-dev` with tags `pre-{sha}` and `latest` - these images are also used as a cache for subsequent builds
