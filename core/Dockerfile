FROM python:3.13-slim

WORKDIR /app

# Dependencies first: this layer is cached unless requirements.txt changes.
# requirements-dev.txt (pytest et al) is deliberately NOT installed here.
COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

# COPY, not ADD: ADD additionally unpacks local archives and fetches URLs,
# neither of which is wanted here.
COPY . /app

# Deliberately still runs as root. Dropping to a non-root user needs that user
# to be in the group owning the mounted /var/run/docker.sock, or run_code_sandbox
# loses access to the daemon entirely - and that GID is a property of the host,
# so it belongs in the chart's securityContext (supplementalGroups), not in a
# blind useradd here.

ENTRYPOINT ["python3", "-W", "ignore", "-u", "main.py"]
