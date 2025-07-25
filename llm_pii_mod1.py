# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import warnings
from typing import Any, Optional

import pandas as pd

from nemo_curator.modifiers import DocumentModifier
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import load_object_on_worker
from nemo_curator.utils.llm_pii_utils import (
    JSON_SCHEMA,
    PII_LABELS,
    get_system_prompt,
    redact,
    validate_entity,
)

__all__ = ["CustomLLMPiiModifier"]


class CustomLLMInference:
    """A class for redacting PII via LLM inference using a provided client"""

    def __init__(
        self,
        llm_client: Any,
        model: str,
        system_prompt: str,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: int = 4096,
    ):
        self.client = llm_client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

    def infer(self, text: str) -> list[dict[str, str]]:
        """Invoke LLM to get PII entities"""

        text = text.strip()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

        # Use the client's pattern to create the completion
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body={"nvext": {"guided_json": JSON_SCHEMA}} if hasattr(self.client, "nvext") else None,
            stream=False,
        )

        # Extract the content from the response
        if hasattr(response, "choices") and len(response.choices) > 0:
            if hasattr(response.choices[0], "message"):
                assistant_message = response.choices[0].message.content
            else:
                # Handle different response formats
                assistant_message = response.choices[0].get("message", {}).get("content", "")
        else:
            # Handle unexpected response format
            return []

        # Parse results
        try:
            entities = json.loads(assistant_message)
            if not entities:
                # LLM returned valid JSON but no entities discovered
                return []
            else:
                # Check that each entity returned is valid
                return [e for e in entities if validate_entity(e, text)]
        except json.decoder.JSONDecodeError:
            return []


class CustomLLMPiiModifier(DocumentModifier):
    """
    This class is a custom implementation for LLM-based PII de-identification using a provided LLM client.
    
    Example usage:
    ```python
    from openai import OpenAI
    
    # Initialize the client
    client = OpenAI(api_key="your-api-key")
    
    # Create the modifier
    modifier = CustomLLMPiiModifier(
        llm_client=client,
        model="gpt-4",
        pii_labels=PII_LABELS,
        language="en",
        batch_size=10,
    )
    
    # Use with the Modify operation
    modify = Modify(modifier)
    modified_dataset = modify(dataset)
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        llm_client: Any,
        model: str,
        system_prompt: Optional[str] = None,
        pii_labels: Optional[list[str]] = None,
        language: str = "en",
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: int = 4096,
        batch_size: int = 10,
    ):
        """
        Initialize the CustomLLMPiiModifier

        Args:
            llm_client (Any): The LLM client to use for inference
            model (str): The model to use for the LLM
            system_prompt (Optional[str]): The system prompt to feed into the LLM.
                If None, a default system prompt is used.
            pii_labels (Optional[List[str]]): The PII labels to identify and remove from the text.
                See documentation for full list of PII labels.
                Default is None, which means all PII labels will be used.
            language (str): The language to use for the LLM.
                Default is "en" for English. If non-English, it is recommended
                to provide a custom system prompt.
            temperature (float): Temperature for the LLM. Default is 0.0 for deterministic outputs.
            top_k (Optional[int]): Top-k sampling parameter. Default is None.
            top_p (Optional[float]): Top-p sampling parameter. Default is None.
            max_tokens (int): Maximum number of tokens to generate. Default is 4096.
            batch_size (int): Number of documents to process in a batch. Default is 10.
        """
        super().__init__()

        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.batch_size = batch_size

        if system_prompt is not None and pii_labels is not None:
            warnings.warn(
                "Custom system_prompt and custom pii_labels were both provided, "
                "but the PII labels should already be included in the system prompt. "
                "The pii_labels will be ignored.",
                stacklevel=2,
            )

        if pii_labels is None:
            pii_labels = PII_LABELS
        if system_prompt is None:
            self.system_prompt = get_system_prompt(pii_labels)
        else:
            self.system_prompt = system_prompt

        if language != "en" and system_prompt is None:
            warnings.warn(
                "The default system prompt is only available for English text. "
                "For other languages, please provide a custom system prompt. "
                "Please refer to the default system prompt as a guide: "
                "\n"
                f"{get_system_prompt(pii_labels)}"
                "\n"
                "In particular, please ensure that the JSON schema is included in the system prompt exactly as shown: "
                "\n"
                f"{JSON_SCHEMA!s}",
                stacklevel=2,
            )
        if language == "en" and system_prompt is not None:
            warnings.warn(
                "Using the default system prompt is strongly recommended for English text. "
                "If you are customizing the system prompt, please refer to the default system prompt as a guide: "
                f"{get_system_prompt(pii_labels)}"
                "\n"
                "In particular, please ensure that the JSON schema is included in the system prompt exactly as shown: "
                "\n"
                f"{JSON_SCHEMA!s}",
                stacklevel=2,
            )

    @batched(batch_arg="batch_size")
    def modify_document(self, text: pd.Series) -> pd.Series:
        """
        Process a batch of documents to redact PII entities
        
        Args:
            text (pd.Series): Series of text documents to process
            
        Returns:
            pd.Series: Series of processed text documents with PII redacted
        """
        self._inferer_key = f"inferer_{id(self)}"
        inferer = load_object_on_worker(self._inferer_key, self.load_inferer, {})
        
        # Process each document in the batch
        pii_entities_lists = [inferer.infer(doc) for doc in text]
        
        # Apply redaction
        return self.batch_redact(text, pii_entities_lists)

    def load_inferer(self) -> CustomLLMInference:
        """Helper function to load the LLM inference object"""
        inferer: CustomLLMInference = CustomLLMInference(
            llm_client=self.llm_client,
            model=self.model,
            system_prompt=self.system_prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        return inferer

    def batch_redact(self, text: pd.Series, pii_entities_lists: list[list[dict[str, str]]]) -> pd.Series:
        """
        Apply redaction to a batch of documents
        
        Args:
            text (pd.Series): Series of original text documents
            pii_entities_lists (list[list[dict[str, str]]]): List of PII entities for each document
            
        Returns:
            pd.Series: Series of redacted text documents
        """
        redacted_texts = [
            redact(text_str, pii_entities) for text_str, pii_entities in zip(text, pii_entities_lists, strict=False)
        ]
        return pd.Series(redacted_texts, index=text.index)



