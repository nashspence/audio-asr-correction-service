# Devcontainer Notes

This workspace runs inside a VS Code devcontainer, but Docker commands talk to the outer host daemon through the mounted `/var/run/docker.sock`.

That means there are two different `localhost` values to keep in mind:

- Devcontainer shell `localhost`: the workspace container itself
- Docker-published ports: the outer Docker host

## Host-Published Ports

`.devcontainer/devcontainer.json` adds:

- `--add-host=host.docker.internal:host-gateway`

Use that hostname from the workspace shell when you want to reach ports published by `docker compose`:

```bash
curl http://host.docker.internal:8000/health/ready
curl http://host.docker.internal:9000/v1/health/ready
```

These may fail if you use `localhost` from the workspace shell, because `localhost` there is the devcontainer, not the Docker host.

## Common Access Patterns

From the workspace shell to published ports:

```bash
curl http://host.docker.internal:8000/health/ready
curl http://host.docker.internal:9000/v1/health/ready
```

From inside a running service container:

```bash
docker exec asr-api curl http://127.0.0.1:8080/health/ready
docker exec asr curl http://127.0.0.1:9000/v1/health/ready
```

Across the compose network by service name:

```bash
docker exec asr-api curl http://asr:9000/v1/health/ready
```

## GPU And Docker

The devcontainer is configured to expose GPUs:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Docker CLI commands run against the host daemon:

```bash
docker ps
docker compose version
```

## Port Forwarding In VS Code

The devcontainer forwards these ports for convenient UI access:

- `8000` for `asr-api`
- `9000` for NIM HTTP
- `50051` for NIM gRPC
- `8080`, `3000`, `5000` for additional app workflows

Port forwarding helps your editor/client reach services, but it does not change what `localhost` means inside the workspace shell.

## Useful Workflow

Start the stack:

```bash
docker compose up -d --build
docker compose ps
```

Check readiness:

```bash
curl http://host.docker.internal:8000/health/ready
curl http://host.docker.internal:9000/v1/health/ready
```

Run the included environment smoke test:

```bash
bash .devcontainer/smoke-test.sh
```

Stop the stack:

```bash
docker compose down
```

## Repo-Specific Notes

- The NIM cache lives under `./nim/.cache`.
- In this environment, the bind-mounted cache is writable by container root, so the `asr` compose service runs as root.
- First-time NIM startup can take several minutes while the model workspace and engine are generated.
