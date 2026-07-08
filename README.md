# Eco-Router-AI

An intelligent AI agent for the AMD Developer Hackathon: ACT II — Track 1 (General-Purpose AI Agent). This agent reads natural language tasks, classifies them into one of 8 capability categories, routes each task to the most token-efficient Fireworks AI model, and outputs results in JSON format.

## Overview

**Eco-Router-AI** optimizes inference across multiple capability domains by:
- 🎯 Classifying tasks into 8 predefined capability categories
- 🧠 Routing to the most token-efficient model that maintains accuracy
- ⚡ Maximizing cost and latency efficiency via Fireworks AI
- 📦 Running in Docker with strict resource constraints (< 10GB, 10-minute runtime)

## Capability Categories

The agent handles tasks across all eight categories:

| #   | Category                          | What it covers                                                  |
| --- | --------------------------------- | --------------------------------------------------------------- |
| 1   | **Factual Knowledge**             | Explaining concepts, definitions, how things work               |
| 2   | **Mathematical Reasoning**        | Multi-step arithmetic, percentages, word problems, projections  |
| 3   | **Sentiment Classification**      | Labeling sentiment and justifying the classification            |
| 4   | **Text Summarization**            | Condensing passages to specific format or length constraints    |
| 5   | **Named Entity Recognition**      | Extracting and labeling entities (person, org, location, date)  |
| 6   | **Code Debugging**                | Identifying bugs in code snippets and providing corrections     |
| 7   | **Logical / Deductive Reasoning** | Constraint-based puzzles where all conditions must be satisfied |
| 8   | **Code Generation**               | Writing correct, well-structured functions from a spec          |

## Requirements

- **Python** 3.12+
- **Docker** (for containerized deployment)
- **Fireworks AI API** access and key

## Getting Started

### Local Development Setup

1. **Install dependencies:**
   ```bash
   pip install httpx python-dotenv orjson pydantic pytest
   ```

2. **Create a `.env` file** (development only):
   ```
   FIREWORKS_API_KEY=your_api_key_here
   FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
   ALLOWED_MODELS=model1,model2,model3
   ```

3. **Run the agent locally:**
   ```bash
   python app/main.py
   ```

### Docker Build & Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t eco-router-ai:latest .
   ```

2. **Run the container:**
   ```bash
   docker run --rm \
     -e FIREWORKS_API_KEY=your_key \
     -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
     -e ALLOWED_MODELS=model1,model2 \
     -v $(pwd)/input:/input \
     -v $(pwd)/output:/output \
     eco-router-ai:latest
   ```

## Usage

### Input Format

Place your tasks in `/input/tasks.json`:
```json
[
  {
    "task_id": "t1",
    "prompt": "Summarize the following text in one sentence: ..."
  },
  {
    "task_id": "t2",
    "prompt": "..."
  }
]
```

### Output Format

Results are written to `/output/results.json`:
```json
[
  {
    "task_id": "t1",
    "answer": "..."
  },
  {
    "task_id": "t2",
    "answer": "..."
  }
]
```

## Architecture

### Project Structure

```
eco-router-ai/
├── app/
│   ├── main.py                 # Entry point
│   │
│   ├── core/                   # Configuration & constants
│   │   ├── config.py           # Settings from environment
│   │   ├── logger.py           # Logging setup
│   │   └── constants.py        # Task categories, model configs
│   │
│   ├── clients/                # External services
│   │   └── fireworks_client.py # Fireworks AI API wrapper
│   │
│   ├── router/                 # AI routing logic
│   │   ├── router.py           # Main routing orchestration
│   │   ├── classifier.py       # Task category classification
│   │   ├── difficulty.py       # Task difficulty estimation
│   │   ├── model_selector.py   # Model selection for efficiency
│   │   └── prompt_optimizer.py # Prompt optimization
│   │
│   ├── prompts/                # Prompt templates
│   │   ├── summarization.py    # Summarization prompts
│   │   ├── reasoning.py        # Reasoning & logic prompts
│   │   ├── coding.py           # Code generation/debugging
│   │   ├── sentiment.py        # Sentiment classification
│   │   ├── ner.py              # Named entity recognition
│   │   └── system.py           # System prompts
│   │
│   ├── services/               # Business logic
│   │   ├── task_service.py     # Task processing
│   │   ├── inference_service.py # Inference orchestration
│   │   └── evaluation_service.py # Result evaluation
│   │
│   ├── io/                     # I/O operations
│   │   ├── reader.py           # JSON input reader
│   │   └── writer.py           # JSON output writer
│   │
│   ├── models/                 # Data models
│   │   ├── task.py             # Task schema (Pydantic)
│   │   ├── result.py           # Result schema
│   │   └── response.py         # API response schema
│   │
│   └── utils/                  # Utility functions
│       ├── json_utils.py       # JSON helpers
│       ├── text_utils.py       # Text processing
│       └── timer.py            # Performance monitoring
│
├── input/
│   └── tasks.json              # Input tasks
├── output/
│   └── results.json            # Output results
│
├── Dockerfile                  # Container image
├── docker-compose.yml          # Docker Compose config
├── context.md                  # Detailed specification
├── CLAUDE.md                   # Development guide
└── README.md                   # This file
```

## Environment Variables

The following environment variables must be provided at runtime. **Do not hardcode these in the image.**

| Variable             | Description                               | Example                                 |
| -------------------- | ----------------------------------------- | --------------------------------------- |
| `FIREWORKS_API_KEY`  | API key provided by the harness           | `sk-...`                                |
| `FIREWORKS_BASE_URL` | Base URL for all Fireworks API calls      | `https://api.fireworks.ai/inference/v1` |
| `ALLOWED_MODELS`     | Comma-separated list of allowed model IDs | `model1,model2,model3`                  |

## Router Capability Metadata

Router model capabilities are stored in [app/router/model_capabilities.json](app/router/model_capabilities.json).
Update that file if you want to refine which model is described as better for a given task type.

## Tech Stack

| Layer             | Technology                       |
| ----------------- | -------------------------------- |
| **Language**      | Python 3.12                      |
| **API**           | Fireworks AI (OpenAI-compatible) |
| **HTTP**          | httpx (async)                    |
| **JSON**          | orjson (high-performance)        |
| **Validation**    | Pydantic v2                      |
| **Async Runtime** | asyncio                          |
| **Configuration** | python-dotenv (dev)              |
| **Testing**       | pytest                           |
| **Container**     | Docker                           |

## Constraints & Requirements

- ✅ All inference **must** route through Fireworks AI via `FIREWORKS_BASE_URL`
- ✅ Only models in `ALLOWED_MODELS` may be called
- ✅ Output must be valid JSON (malformed output scores zero)
- ✅ Max runtime: **10 minutes**
- ✅ Container startup: **≤ 60 seconds**
- ✅ Image architecture: `linux/amd64`
- ✅ Compressed image size: **< 10GB**
- ✅ Accuracy is gated first, token efficiency ranked second among accurate solutions

## Development

For detailed development guidelines and internal architecture decisions, see [CLAUDE.md](CLAUDE.md).

For the complete specification, including scoring criteria and capability definitions, see [context.md](context.md).

## Testing

Run the test suite:
```bash
pytest app/ -v
```

## Performance Monitoring

The agent logs inference latency, token usage, and model selection decisions. Monitor these metrics to optimize routing and model efficiency.

## License

See [LICENSE](LICENSE) file for details.