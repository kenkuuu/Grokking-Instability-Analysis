# Grokking‑Instability‑Analysis

## Quick start (Docker)

```bash
cd ~/Work/Grokking-Instability-Analysis

# 1. Build the image (first time only)
docker compose -f docker/docker-compose.yml build

# 2. Start the container
docker compose -f docker/docker-compose.yml up -d

# 3. Enter the running container
docker exec -it grokking_trainer bash
```
