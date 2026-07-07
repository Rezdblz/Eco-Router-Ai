# Eco-Router-Ai
dependencies

pip install httpx python-dotenv orjson pydantic pytest

stack
Language
├── Python 3.12
API
├── Fireworks AI (OpenAI-compatible)
HTTP
├── httpx
JSON
├── orjson
Validation
├── pydantic
Async
├── asyncio
Configuration
├── python-dotenv (development only)
Testing
├── pytest
Container
└── Docker

ECO-ROUTER-AI
├── app/
│   ├── main.py                 # Entry point
│   │
│   ├── core/                   # Configuration & constants
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── constants.py
│   │
│   ├── clients/                # External services
│   │   └── fireworks_client.py
│   │
│   ├── router/                 # AI Router logic
│   │   ├── router.py
│   │   ├── classifier.py
│   │   ├── difficulty.py
│   │   ├── model_selector.py
│   │   └── prompt_optimizer.py
│   │
│   ├── prompts/                # Prompt templates
│   │   ├── summarization.py
│   │   ├── reasoning.py
│   │   ├── coding.py
│   │   ├── sentiment.py
│   │   └── system.py
│   │
│   ├── services/               # Business logic
│   │   ├── task_service.py
│   │   ├── inference_service.py
│   │   └── evaluation_service.py
│   │
│   ├── io/
│   │   ├── reader.py
│   │   └── writer.py
│   │
│   ├── models/                 # Data models
│   │   ├── task.py
│   │   ├── result.py
│   │   └── response.py
│   │
│   └── utils/
│       ├── json_utils.py
│       ├── text_utils.py
│       └── timer.py