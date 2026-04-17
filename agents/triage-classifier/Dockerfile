FROM python:3.12-slim

WORKDIR /app

# Install the vendored AgentYard SDK. We copy from the repo root so
# every agent in this repo shares one SDK version — bump it by updating
# vendor-agentyard-sdk/ at the root.
COPY vendor-agentyard-sdk /tmp/sdk/
RUN pip install --no-cache-dir /tmp/sdk/ && rm -rf /tmp/sdk/

# Per-agent source. When you add a new agent, copy this Dockerfile,
# bump the paths, and rebuild.
COPY agents/triage-classifier/main.py .

ENV YARD_PORT=9101
CMD ["python", "main.py"]
