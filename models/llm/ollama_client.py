"""Small dependency-free Ollama client and binary log-probability parser."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot satisfy the protocol."""


class ScoreUnavailable(OllamaError):
    """Raised when a response cannot supply comparable yes/no probabilities."""


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=(
                None
                if payload is None
                else json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
        except TimeoutError as error:
            raise OllamaError(
                f"Ollama request exceeded {self.timeout_seconds} seconds"
            ) from error
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {error.code}: {body[:1000]}") from error
        except urllib.error.URLError as error:
            raise OllamaError(f"Cannot reach Ollama: {error.reason}") from error
        except json.JSONDecodeError as error:
            raise OllamaError("Ollama returned a non-JSON response") from error
        if not isinstance(value, dict):
            raise OllamaError("Ollama response must be a JSON object")
        return value

    def version(self) -> str:
        return str(self.request("/api/version").get("version", ""))

    def tags(self) -> dict[str, dict[str, Any]]:
        result = self.request("/api/tags")
        return {
            str(item.get("name") or item.get("model")): item
            for item in result.get("models", [])
        }

    def show(self, model: str) -> dict[str, Any]:
        return self.request("/api/show", {"model": model})

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("/api/chat", payload)


@dataclass(frozen=True)
class BinaryScore:
    predicted_label: int
    ambiguity_probability: float
    log_odds_yes_no: float
    yes_logprob: float
    no_logprob: float
    output_text: str


def _normalized_label_token(value: object) -> str:
    return str(value).strip().casefold()


def extract_binary_score(response: dict[str, Any]) -> BinaryScore:
    """Read the generated label and normalize its top-logprob yes/no alternatives."""

    if str(response.get("done_reason", "")) in {"length", "max_tokens"}:
        raise ScoreUnavailable("Ollama exhausted the output-token limit")
    message = response.get("message")
    if not isinstance(message, dict):
        raise ScoreUnavailable("Ollama response has no assistant message")
    output = str(message.get("content", "")).strip().casefold()
    if output not in {"yes", "no"}:
        raise ScoreUnavailable(f"Expected exactly yes or no, received {output!r}")

    entries = response.get("logprobs")
    if not isinstance(entries, list) or not entries:
        raise ScoreUnavailable("Ollama returned no token log-probabilities")
    chosen: dict[str, float] | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        alternatives = [entry, *(entry.get("top_logprobs") or [])]
        by_label: dict[str, float] = {}
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                continue
            label = _normalized_label_token(alternative.get("token", ""))
            if label not in {"yes", "no"}:
                continue
            try:
                logprob = float(alternative["logprob"])
            except (KeyError, TypeError, ValueError):
                continue
            by_label[label] = max(logprob, by_label.get(label, -math.inf))
        if _normalized_label_token(entry.get("token", "")) == output:
            chosen = by_label
            break
    if chosen is None or set(chosen) != {"yes", "no"}:
        raise ScoreUnavailable(
            "The generated-label token did not expose both yes and no among "
            "top_logprobs; PR-AUC scoring would be invalid"
        )
    log_odds = chosen["yes"] - chosen["no"]
    probability = (
        1.0 / (1.0 + math.exp(-log_odds))
        if log_odds >= 0
        else math.exp(log_odds) / (1.0 + math.exp(log_odds))
    )
    return BinaryScore(
        predicted_label=1 if output == "yes" else 0,
        ambiguity_probability=probability,
        log_odds_yes_no=log_odds,
        yes_logprob=chosen["yes"],
        no_logprob=chosen["no"],
        output_text=output,
    )
