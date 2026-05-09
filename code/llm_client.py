import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from API import DEFAULT_BASE_URL, DEFAULT_EXTRA_BODY, DEFAULT_MODEL, DEFAULT_REASONING_EFFORT


class LLMConfigError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


def _load_env_file(env_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not env_path.exists():
        return data

    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        data[key] = value
    return data


def _resolve_config() -> Dict[str, str]:
    cwd = Path.cwd()
    sample_env = cwd / "sample.env"
    dot_env = cwd / ".env"

    file_values = {}
    if sample_env.exists():
        file_values.update(_load_env_file(sample_env))
    if dot_env.exists():
        file_values.update(_load_env_file(dot_env))

    endpoint = os.getenv("DS_ENDPOINT", file_values.get("DS_ENDPOINT", DEFAULT_BASE_URL)).strip()
    api_key = os.getenv("DS_API_KEY", file_values.get("DS_API_KEY", "")).strip()
    deployment = os.getenv("DS_DEPLOYMENT", file_values.get("DS_DEPLOYMENT", DEFAULT_MODEL)).strip()

    if not endpoint:
        raise LLMConfigError("DS_ENDPOINT is empty. Please set it in sample.env or environment variables.")
    if not api_key:
        raise LLMConfigError("DS_API_KEY is empty. Please set it in sample.env or environment variables.")
    if not deployment:
        raise LLMConfigError("DS_DEPLOYMENT is empty. Please set it in sample.env or environment variables.")

    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "deployment": deployment,
    }


def _normalize_chat_completions_url(endpoint: str) -> str:
    ep = endpoint.strip().rstrip("/")
    lower = ep.lower()
    if lower.endswith("/chat/completions"):
        return ep
    if lower.endswith("/v1"):
        return ep + "/chat/completions"
    if "/openai/deployments/" in lower and "api-version=" in lower:
        return ep
    if "/openai/deployments/" in lower and not lower.endswith("/chat/completions"):
        return ep + "/chat/completions"
    if lower.endswith("/openai"):
        return ep + "/v1/chat/completions"
    if lower.endswith("/anthropic"):
        return ep[: -len("/anthropic")] + "/chat/completions"
    return ep + "/v1/chat/completions"


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise LLMRequestError("Model returned empty content; expected JSON.")

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start < 0:
        raise LLMRequestError("Model response is not valid JSON object.")

    depth = 0
    in_string = False
    escape = False
    json_text = ""
    for pos in range(start, len(candidate)):
        ch = candidate[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_text = candidate[start : pos + 1]
                break

    if not json_text:
        raise LLMRequestError("Model response contains an incomplete JSON object.")

    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise LLMRequestError(f"Failed to parse JSON object from model response: {exc}") from exc

    if not isinstance(obj, dict):
        raise LLMRequestError("Parsed JSON is not an object.")
    return obj


class DeepSeekClient:
    def __init__(self, timeout: int = 120) -> None:
        cfg = _resolve_config()
        self.endpoint = cfg["endpoint"]
        self.api_key = cfg["api_key"]
        self.deployment = cfg["deployment"]
        self.timeout = timeout
        self.chat_url = _normalize_chat_completions_url(self.endpoint)
        # Default: bypass env proxies (many local/dev envs set 127.0.0.1 dead proxy).
        # Set DS_DISABLE_PROXY=0 to re-enable system proxy usage.
        self.disable_proxy = os.getenv("DS_DISABLE_PROXY", "1").strip() != "0"
        self._opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if self.disable_proxy
            else urllib.request.build_opener()
        )

    @property
    def model_source(self) -> str:
        return self.deployment

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.chat_url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise LLMRequestError(f"LLM HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMRequestError(f"LLM URL error: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMRequestError(f"LLM response is not valid JSON: {raw[:500]}") from exc

        if "error" in data:
            raise LLMRequestError(f"LLM returned error: {data['error']}")

        return data

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 600,
        retries: int = 5,
    ) -> str:
        payload = {
            "model": self.deployment,
            "temperature": temperature,
            "max_tokens": max(max_tokens, 3000),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if os.getenv("DS_ENABLE_THINKING", "0").strip() == "1":
            payload["reasoning_effort"] = DEFAULT_REASONING_EFFORT
            payload.update(DEFAULT_EXTRA_BODY)

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                data = self._post_json(payload)
                choices = data.get("choices", [])
                if not choices:
                    raise LLMRequestError("LLM returned no choices.")
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(str(item["text"]))
                    content = "".join(parts)
                return str(content).strip()
            except Exception as exc:
                last_err = exc
                if attempt < retries:
                    time.sleep(1.0 + attempt)
                    continue
                break

        raise LLMRequestError(f"LLM request failed after retries: {last_err}")

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 800,
        retries: int = 5,
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                text = self.chat_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                )
                return _extract_json_object(text)
            except LLMRequestError as exc:
                last_err = exc
                if attempt < retries:
                    time.sleep(1.0 + attempt)
                    continue
                break

        raise LLMRequestError(f"LLM JSON request failed after retries: {last_err}")
