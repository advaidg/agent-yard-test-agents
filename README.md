# agent-yard-test-agents

A reference monorepo for AgentYard agents. Every agent lives in
`agents/<name>/` with its own `Dockerfile`, `main.py`, and `agent.yaml`.
CI builds each agent's image and pushes it to a single shared Docker
Hub repo — one repo, many agents, distinguished by tag prefix.

## Layout

```
agents/
  triage-classifier/
    Dockerfile
    main.py
    agent.yaml
  triage-sentiment/
  triage-responder/
vendor-agentyard-sdk/   # vendored AgentYard SDK used by every agent
scripts/
  notify-agentyard.sh   # POST an image-push event to your AgentYard
.github/workflows/
  build-and-push.yml    # matrix workflow, auto-detects changed agents
```

## Tag scheme

One Docker Hub repo (default `advaidg/agentyard2`) holds every agent.
Tags distinguish them:

```
docker.io/advaidg/agentyard2:triage-classifier-9f8a3c1    # immutable
docker.io/advaidg/agentyard2:triage-classifier-latest     # rolling
```

## CI — what the workflow does

On `push` to `main`:
1. Diffs `HEAD~1..HEAD` to find which `agents/<X>/` dirs changed
2. If `vendor-agentyard-sdk/` or `.github/` changed, rebuilds **all** agents
3. For each changed agent: `docker build`, `push` two tags, announce to AgentYard

On `workflow_dispatch`: pick one agent or `all` manually.

## GitHub repo setup (one time)

Required secrets (Settings → Secrets and variables → Actions → Secrets):

| Secret | Purpose |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub PAT with write access to the repo |
| `YARD_WEBHOOK_SECRET` | Shared secret matching your AgentYard registry env |

Optional **variable** (Settings → Variables):

| Variable | Purpose |
|---|---|
| `AGENTYARD_WEBHOOK_URL` | Full URL to your AgentYard webhook endpoint, e.g. `https://agentyard.example.com/api/webhooks/agent-image`. Leave unset while AgentYard is local-only — the workflow will print a copy-pasteable curl command instead of calling. |

## Adding a new agent

1. Copy one of the existing folders under `agents/`
2. Edit `main.py` (your agent logic), `agent.yaml` (name, namespace, version, port), and `Dockerfile` (bump the `COPY` path and `YARD_PORT`)
3. Push to main — CI picks it up automatically

## Manual webhook replay (while AgentYard isn't exposed publicly)

If you didn't set `AGENTYARD_WEBHOOK_URL`, the workflow prints a curl command in the job log. Copy it, fill in `<your-agentyard>` and `<YARD_WEBHOOK_SECRET>`, and run it locally. Or use the helper:

```bash
cp .env.example .env                  # fill in
source .env
./scripts/notify-agentyard.sh \
  --agent triage-classifier \
  --image docker.io/advaidg/agentyard2 \
  --tag   triage-classifier-9f8a3c1 \
  --version 1.0.0 \
  --commit 9f8a3c1
```

Response is JSON. `applied: true` means the agent was already registered and has been bumped. `pending: true` means the event is held until the agent boots for the first time — no action needed.

## Updating the vendored SDK

The SDK is a copy of `backend/sdk/` from the main AgentYard repo. To bump:

```bash
rm -rf vendor-agentyard-sdk/
cp -r /path/to/AgentYard/backend/sdk vendor-agentyard-sdk
rm -rf vendor-agentyard-sdk/agentyard.egg-info vendor-agentyard-sdk/tests
git add vendor-agentyard-sdk
git commit -m "Bump SDK to vX.Y"
```

Pushing that will rebuild every agent (the workflow detects shared changes).
