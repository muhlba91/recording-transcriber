"""Text backend for OpenAI-compatible /chat/completions APIs."""

import logging

import httpx

from ..models.model_response import ModelResponse
from ..models.usage import Usage
from ..utils.const import (
    CHAT_ROUTE,
    CHAT_SCHEMA_NAME,
    MODELS_ROUTE,
    REQUEST_TIMEOUT_MS,
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    STATUS_UNKNOWN,
)

_LOGGER = logging.getLogger(__name__)

# the finish reasons of a chat completion: a normal stop, and a cut-off at the limit
_FINISH_STATUS = {"stop": STATUS_COMPLETED, "length": STATUS_INCOMPLETE}


class ChatApiBackend:
    """Complete text prompts with the JSON-schema structured output of an
    OpenAI-compatible chat API (OpenAI, Mistral or a custom endpoint).

    Nothing is stored at the provider, so there is nothing to delete."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        data_policy: str,
        client: httpx.Client | None = None,
    ):
        """Initialize the backend.

        provider: the provider name, for logs and run.json
        base_url: the API root; /chat/completions is appended
        api_key: the provider key
        data_policy: the provider's logging and retention terms, for run.json
        client: a preconfigured client, used by tests (default: None)"""
        self.provider = provider
        root = base_url.removesuffix("/")
        self.url = root + CHAT_ROUTE
        self.models_url = root + MODELS_ROUTE
        self.data_policy = data_policy
        self.__api_key = api_key
        self.__client = client or httpx.Client(timeout=REQUEST_TIMEOUT_MS / 1000)

    def describe(self) -> dict:
        """Return provider, endpoint and logging setting for run.json."""
        return {
            "provider": self.provider,
            "endpoint": self.url,
            "logging": self.data_policy,
        }

    def check(self, model: str) -> bool:
        """Return whether the key works and the provider lists the model.

        Only a rejected key or a listing without the model fail: some keys and
        gateways cannot list models but still complete text.

        model: the model id"""
        try:
            response = self.__client.get(
                self.models_url,
                headers=self._headers(),
            )
        except httpx.HTTPError as error:
            _LOGGER.error(
                "Could not access %s model %s: %s.", self.provider, model, error
            )
            return False
        if response.status_code == httpx.codes.UNAUTHORIZED:
            _LOGGER.error(
                "%s rejected the key: %s.", self.provider, response.text[:300]
            )
            return False
        try:
            response.raise_for_status()
            listed = _listed(response.json()["data"])
        except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as error:
            _LOGGER.warning(
                "Could not list the %s models (%s); trying %s anyway.",
                self.provider,
                error,
                model,
            )
            return True
        if model not in listed:
            _LOGGER.error("%s does not list the model %s.", self.provider, model)
            return False
        return True

    def complete(self, prompt: str, schema: dict, model: str) -> ModelResponse | None:
        """Return the JSON answer to a text prompt, or None if the call failed.

        prompt: the instructions
        schema: the JSON schema of the answer
        model: the model id"""
        # pragma: no mutate start (wire format)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": CHAT_SCHEMA_NAME, "schema": schema},
            },
        }
        # pragma: no mutate end
        try:
            response = self.__client.post(
                self.url,
                headers=self._headers(),
                json=body,
            )
        except httpx.HTTPError as error:
            _LOGGER.error(
                "Could not call %s model %s: %s.", self.provider, model, error
            )
            return None
        if response.status_code != httpx.codes.OK:
            _LOGGER.error(
                "%s model %s returned %s: %s.",
                self.provider,
                model,
                response.status_code,
                response.text[:300],
            )
            return None
        try:
            return _parse(response.json())
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            _LOGGER.error("Invalid answer from %s: %s.", self.provider, error)
            return None

    def _headers(self) -> dict:
        """Return the request headers with the key."""
        # header names are case-insensitive
        return {"Authorization": f"Bearer {self.__api_key}"}

    def close(self) -> None:
        """Close the HTTP client; nothing is stored at the provider."""
        self.__client.close()


def _listed(entries: list[dict]) -> set[str]:
    """Return the model ids and aliases of a model list, without a "models/" prefix.

    entries: the "data" entries of the answer"""
    names = set()
    for entry in entries:
        names.update([entry["id"], *(entry.get("aliases") or [])])
    return {name.removeprefix("models/") for name in names}


def _parse(data: dict) -> ModelResponse:
    """Return the first choice and the usage of a chat completion; raise on an
    unexpected shape.

    data: the JSON answer"""
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    # reasoning tokens are part of completion_tokens; Usage counts them separately
    reasoning = (usage.get("completion_tokens_details") or {}).get(
        "reasoning_tokens"
    ) or 0
    return ModelResponse(
        text=choice["message"]["content"],
        status=_FINISH_STATUS.get(choice.get("finish_reason"), STATUS_UNKNOWN),
        usage=Usage(
            input_text_tokens=usage.get("prompt_tokens") or 0,
            output_tokens=(usage.get("completion_tokens") or 0) - reasoning,
            thought_tokens=reasoning,
        ),
    )
