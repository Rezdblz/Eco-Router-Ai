import os
from pydantic import BaseModel, field_validator

try:
    from dotenv import load_dotenv
    if os.getenv("ENVIRONMENT", "production") == "development":
        load_dotenv()
except ImportError:
    pass


class Settings(BaseModel):
    fireworks_api_key: str
    fireworks_base_url: str
    allowed_models: list[str]

    input_path: str = "/input/tasks.json"
    output_path: str = "/output/results.json"

    log_level: str = "INFO"
    default_temperature: float = 0.0
    max_retries: int = 3
    request_timeout: int = 60

    @field_validator("allowed_models", mode="before")
    @classmethod
    def parse_models(cls, v: object) -> list[str]:
        if isinstance(v, str):
            models = [m.strip() for m in v.split(",") if m.strip()]
            if not models:
                raise ValueError("ALLOWED_MODELS is set but contains no valid model IDs")
            return models
        return v

    @field_validator("fireworks_base_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


def _require_env(var: str) -> str:
    val = os.getenv(var, "").strip()
    if not val:
        raise RuntimeError(
            f"Missing or empty required environment variable: {var}. "
            "This must be injected by the harness at runtime."
        )
    return val


def _parse_numeric(var: str, cast: type, default: str) -> object:
    raw = os.getenv(var, default)
    try:
        return cast(raw)
    except (ValueError, TypeError):
        raise RuntimeError(
            f"Environment variable {var}={raw!r} is not a valid {cast.__name__}."
        )


def load_settings() -> Settings:
    api_key = _require_env("FIREWORKS_API_KEY")
    base_url = _require_env("FIREWORKS_BASE_URL")
    allowed_models = _require_env("ALLOWED_MODELS")

    return Settings(
        fireworks_api_key=api_key,
        fireworks_base_url=base_url,
        allowed_models=allowed_models,
        input_path=os.getenv("INPUT_PATH", "/input/tasks.json"),
        output_path=os.getenv("OUTPUT_PATH", "/output/results.json"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        default_temperature=_parse_numeric("DEFAULT_TEMPERATURE", float, "0.0"),
        max_retries=_parse_numeric("MAX_RETRIES", int, "3"),
        request_timeout=_parse_numeric("REQUEST_TIMEOUT", int, "60"),
    )
