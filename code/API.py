import os
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v3"
API_KEY_ENV = "DS_API_KEY"

DEFAULT_REASONING_EFFORT = "high"
DEFAULT_EXTRA_BODY: Dict[str, Any] = {"thinking": {"type": "enabled"}}


def _load_file_values() -> Dict[str, str]:
    values: Dict[str, str] = {}
    for env_path in (Path.cwd() / "sample.env", Path.cwd() / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _config_value(name: str, default: str = "") -> str:
    file_values = _load_file_values()
    return os.getenv(name, file_values.get(name, default)).strip()


def get_api_key() -> str:
    api_key = _config_value(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is empty. Please set it in .env, sample.env, or your environment.")
    return api_key


def get_client() -> Any:
    from openai import OpenAI

    return OpenAI(api_key=get_api_key(), base_url=_config_value("DS_ENDPOINT", DEFAULT_BASE_URL) or DEFAULT_BASE_URL)


def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    return get_client().chat.completions.create(
        model=model or _config_value("DS_DEPLOYMENT", DEFAULT_MODEL) or DEFAULT_MODEL,
        messages=messages,
        stream=stream,
        reasoning_effort=kwargs.pop("reasoning_effort", DEFAULT_REASONING_EFFORT),
        extra_body=kwargs.pop("extra_body", DEFAULT_EXTRA_BODY),
        **kwargs,
    )


if __name__ == "__main__":
    response = chat_completion(
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
    )
    print(response.choices[0].message.content)
