import json
import warnings

# Assume these helper functions exist or are implemented elsewhere.
# For example, you might have:
# - JSON_SCHEMA: a JSON string to guide the LLM responses.
# - PII_LABELS: a list of allowed PII labels.
# - get_system_prompt: function that returns a default system prompt given the pii_labels.
# - redact: function that redacts PII entities in a given text.
# - validate_entity: function that checks if a returned entity is valid.
from my_llm_pii_utils import JSON_SCHEMA, PII_LABELS, get_system_prompt, redact, validate_entity

# This DocumentModifier is assumed to be the abstract base class.
from my_document_modifier import DocumentModifier  # Replace with your actual import path


class CustomLLMPiiInference:
    """
    A synchronous inference wrapper for redacting PII
    using a custom LLM client.
    """

    def __init__(self, client, model: str, system_prompt: str,
                 temperature: float = 0.7, top_k: int = 40,
                 top_p: float = 0.95, max_tokens: int = 1024):
        """
        Args:
            client: An LLM client instance that implements
                client.chat.completions.create(system_prompt, user_prompt, model, temperature, top_k, top_p, max_tokens)
            model (str): The model identifier.
            system_prompt (str): The system prompt used to instruct the LLM.
            temperature (float): Sampling temperature. Default: 0.7.
            top_k (int): Top-k sampling setting.
            top_p (float): Top-p sampling setting.
            max_tokens (int): Maximum tokens to generate.
        """
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

    def infer(self, text: str) -> list[dict[str, str]]:
        """
        Invokes the LLM to extract PII entities from the input text.

        Args:
            text (str): The text from which to extract PII.

        Returns:
            List[dict[str, str]]: A list of entities detected as PII.
        """
        text = text.strip()
        user_prompt = text

        # Call the client's API following the provided API pattern
        response = self.client.chat.completions.create(
            self.system_prompt,
            user_prompt,
            self.model,
            self.temperature,
            self.top_k,
            self.top_p,
            self.max_tokens,
        )

        # Assume the response is a dict structured similarly to:
        # {
        #    "choices": [
        #        {"message": {"content": "<JSON string with entities>"}}
        #    ]
        # }
        assistant_message = response["choices"][0]["message"]["content"]

        # Parse results
        try:
            entities = json.loads(assistant_message)
            if not entities:
                return []  # Valid JSON but no entities found
            else:
                # Validate each entity; only include those valid for the given text.
                return [e for e in entities if validate_entity(e, text)]
        except json.decoder.JSONDecodeError:
            # In case the LLM response isn't valid JSON, return an empty list.
            return []


class CustomLLMPiiModifier(DocumentModifier):
    """
    A synchronous modifier for removing PII from text.
    Mimics the structural architecture of LLMPiiModifier, but using a
    custom LLM client passed in as an argument.
    """

    def __init__(self, client, model: str = "custom-llm-model",
                 system_prompt: str | None = None,
                 pii_labels: list[str] | None = None,
                 temperature: float = 0.7, top_k: int = 40,
                 top_p: float = 0.95, max_tokens: int = 1024,
                 language: str = "en"):
        """
        Args:
            client: The custom LLM client to use.
            model (str): The LLM model identifier.
            system_prompt (Optional[str]): Custom system prompt. If omitted, a default
                prompt is generated based on pii_labels.
            pii_labels (Optional[List[str]]): List of PII labels to consider.
                Defaults to all supported labels.
            temperature, top_k, top_p, max_tokens: Additional parameters passed to the LLM.
            language (str): Language of the text. For non-English texts, a custom system_prompt
                is recommended.
        """
        super().__init__()

        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

        if pii_labels is None:
            pii_labels = PII_LABELS

        if system_prompt is None:
            self.system_prompt = get_system_prompt(pii_labels)
        else:
            self.system_prompt = system_prompt

        if language != "en" and system_prompt is None:
            warnings.warn("Default system prompt is for English only. "
                          "Please provide a custom prompt for language '{}'.".format(language),
                          stacklevel=2)
        elif language == "en" and system_prompt is not None:
            warnings.warn("Using a custom system prompt for English text; "
                          "ensure it includes the necessary JSON schema.",
                          stacklevel=2)

    def modify_document(self, text: str) -> str:
        """
        Redacts the detected PII in the text.

        Args:
            text (str): The input text.

        Returns:
            str: The redacted text.
        """
        inference = CustomLLMPiiInference(
            client=self.client,
            model=self.model,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        pii_entities = inference.infer(text)
        return redact(text, pii_entities)


import asyncio
import json
import warnings
import pandas as pd

from my_llm_pii_utils import JSON_SCHEMA, PII_LABELS, get_system_prompt, redact, validate_entity
from my_document_modifier import DocumentModifier  # Abstract base class
from my_decorators import batched  # Assume you have a decorator for batch processing


class AsyncCustomLLMPiiInference:
    """
    An asynchronous inference wrapper for redacting PII using a custom LLM client.
    """

    def __init__(self, client, model: str, system_prompt: str,
                 temperature: float = 0.7, top_k: int = 40,
                 top_p: float = 0.95, max_tokens: int = 1024):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

    async def infer(self, text: str) -> list[dict[str, str]]:
        text = text.strip()
        user_prompt = text

        # Asynchronous call to the LLM client.
        response = await self.client.chat.completions.create(
            self.system_prompt,
            user_prompt,
            self.model,
            self.temperature,
            self.top_k,
            self.top_p,
            self.max_tokens,
        )

        assistant_message = response["choices"][0]["message"]["content"]

        try:
            entities = json.loads(assistant_message)
            if not entities:
                return []
            else:
                return [e for e in entities if validate_entity(e, text)]
        except json.decoder.JSONDecodeError:
            return []


class AsyncCustomLLMPiiModifier(DocumentModifier):
    """
    An asynchronous modifier that applies PII redaction to text
    using a custom LLM client. This implementation mirrors the
    AsyncLLMPiiModifier structural architecture.
    """

    def __init__(self, client,
                 model: str = "custom-llm-model",
                 system_prompt: str | None = None,
                 pii_labels: list[str] | None = None,
                 temperature: float = 0.7,
                 top_k: int = 40,
                 top_p: float = 0.95,
                 max_tokens: int = 1024,
                 language: str = "en",
                 max_concurrent_requests: int | None = None):
        super().__init__()
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_concurrent_requests = max_concurrent_requests

        if pii_labels is None:
            pii_labels = PII_LABELS

        if system_prompt is None:
            self.system_prompt = get_system_prompt(pii_labels)
        else:
            self.system_prompt = system_prompt

        if language != "en" and system_prompt is None:
            warnings.warn(
                "Default system prompt is in English only. Please provide a custom prompt "
                "when using language '{}'.".format(language),
                stacklevel=2,
            )
        elif language == "en" and system_prompt is not None:
            warnings.warn(
                "Using a custom system prompt for English text. Ensure it includes the required JSON schema.",
                stacklevel=2,
            )

    @batched
    def modify_document(self, text: pd.Series) -> pd.Series:
        """
        Processes a batch of text data and applies asynchronous PII redaction.

        Args:
            text (pd.Series): Series of text strings.

        Returns:
            pd.Series: Series with redacted text.
        """
        # Optionally, create or load the asynchronous inference object.
        inference = AsyncCustomLLMPiiInference(
            client=self.client,
            model=self.model,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        pii_entities_lists = asyncio.run(self._infer_batch(text, inference))
        return self._apply_redaction(text, pii_entities_lists)

    async def _infer_batch(self, text: pd.Series, inference: AsyncCustomLLMPiiInference) -> list[list[dict[str, str]]]:
        tasks = [inference.infer(item) for item in text]
        # Use a semaphore or similar approach if max_concurrent_requests is set.
        if self.max_concurrent_requests:
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)

            async def sem_task(task):
                async with semaphore:
                    return await task

            tasks = [sem_task(task) for task in tasks]

        return await asyncio.gather(*tasks)

    def _apply_redaction(self, texts: pd.Series, pii_entities_lists: list[list[dict[str, str]]]) -> pd.Series:
        redacted = [
            redact(text, entities) for text, entities in zip(texts, pii_entities_lists)
        ]
        return pd.Series(redacted, index=texts.index)
