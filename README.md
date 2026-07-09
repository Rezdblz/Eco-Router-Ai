# Eco-Router-AI

Eco-Router-AI is a Fireworks-backed task router that reads JSON tasks, classifies each task into one of eight capability categories, selects an allowed model, and writes answers plus analytics back to disk.

## What It Does

- Reads tasks from [input/tasks.json](input/tasks.json)
- Classifies each task before inference
- Selects from the models listed in `ALLOWED_MODELS`
- Writes results to [output/results.json](output/results.json)
- Writes analytics to [output/analytics.json](output/analytics.json) and [output/analytics_logs/](output/analytics_logs)

## Requirements

- Python 3.12+
- Fireworks AI API access
- Docker if you want to run the containerized workflow

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file for local runs:
   ```env
   FIREWORKS_API_KEY=your_api_key_here
   FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
   ALLOWED_MODELS=model-a,model-b
   ```

3. Run the pipeline:
   ```bash
   py -m app.main
   ```

## Docker

The repository includes [docker-compose.yml](docker-compose.yml) with the expected input/output mounts.

```bash
docker compose up --build
```

If you prefer plain Docker:

```bash
docker build -t eco-router-ai:latest .
docker run --rm \
  --env-file .env \
  -v $(Get-Location)/input:/input \
  -v $(Get-Location)/output:/output \
  eco-router-ai:latest
```

## Input Format

`input/tasks.json` must contain a JSON array of task objects. Each task needs a `task_id` and `prompt`.

```json
[
  {
    "task_id": "t1",
    "prompt": "Summarize the following text in one sentence."
  },
  {
    "task_id": "t2",
    "prompt": "What is 17% of 240?"
  }
]
```

## Output Files

- [output/results.json](output/results.json) contains the final answers:

```json
[
  {
    "task_id": "t1",
    "answer": "..."
  }
]
```

- [output/analytics.json](output/analytics.json) contains per-run summary data and task-level analytics.
- [output/analytics_logs/](output/analytics_logs) stores run snapshots named `run_<run_id>.json`.

## Configuration

The pipeline reads these environment variables at runtime:

| Variable | Required | Purpose | Default |
| --- | --- | --- | --- |
| `FIREWORKS_API_KEY` | Yes | Fireworks API key | None |
| `FIREWORKS_BASE_URL` | Yes | Fireworks base URL | None |
| `ALLOWED_MODELS` | Yes | Comma-separated list of allowed model IDs | None |
| `ROUTER_MODEL` | No | Model used by the router/classifier | First entry in `ALLOWED_MODELS` |
| `INPUT_PATH` | No | Input tasks file | `/input/tasks.json` |
| `OUTPUT_PATH` | No | Results file | `/output/results.json` |
| `ANALYTICS_PATH` | No | Analytics summary file | `/output/analytics.json` |
| `ANALYTICS_HISTORY_DIR` | No | Analytics history directory | `/output/analytics_logs` |
| `LOG_LEVEL` | No | Python logging level | `INFO` |
| `DEFAULT_TEMPERATURE` | No | Default generation temperature | `0.0` |
| `MAX_RETRIES` | No | Retry count for model calls | `3` |
| `REQUEST_TIMEOUT` | No | HTTP timeout in seconds | `60` |

## Repository Layout

```text
app/
├── main.py
├── clients/
│   ├── fireworks_client.py
│   └── response_parser.py
├── core/
│   └── config.py
├── io/
│   ├── reader.py
│   └── writer.py
├── models/
│   ├── result.py
│   └── task.py
├── router/
│   ├── classifier.py
│   ├── model_capabilities.json
│   ├── model_selector.py
│   ├── prompt_templates.py
│   └── token_allocator.py
└── services/
    ├── analytics.py
    ├── inference.py
    ├── pipeline.py
    └── processor.py
```

## Testing

Run the test suite with:

```bash
py -m pytest
```

## Notes

- Model capability metadata lives in [app/router/model_capabilities.json](app/router/model_capabilities.json).
- The complete task specification is in [context.md](context.md).

## License

See [LICENSE](LICENSE) for details.
