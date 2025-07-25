import pandas as pd
import logging

from nemo_curator.modifiers import DocumentModifier
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import load_object_on_worker

# You can set your own default values if needed.
DEFAULT_BATCH_SIZE = 2000
DEFAULT_SUPPORTED_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"]

class PiiCountModifier(DocumentModifier):
    """
    A modifier that uses the PII detection engine and returns a dictionary
    containing the total number of PII detections and a breakdown count by PII entity.
    
    Example output for a document might look like:
    {
        "pii_count": 3,
        "pii_by_entity": {
            "PERSON": 1,
            "EMAIL_ADDRESS": 1,
            "PHONE_NUMBER": 1
        }
    }
    """

    def __init__(
        self,
        language: str = "en",
        supported_entities: list[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "gpu",
        **kwargs,
    ):
        super().__init__()
        self.language = language
        # Use default supported entities if none provided
        self.supported_entities = supported_entities or DEFAULT_SUPPORTED_ENTITIES
        self.batch_size = batch_size
        self.device = device
        self.kwargs = kwargs

    @batched
    def modify_document(self, text: pd.Series, partition_info: dict | None = None) -> pd.Series:
        logger = logging.getLogger(__name__)
        # Use the helper to load the deidentifier object on the worker.
        deidentifier = load_object_on_worker("deidentifier", self.load_deidentifier, {})

        result = []
        # Process each document in the Series.
        for document in text.tolist():
            try:
                # Use the analyzer to detect PII entities.
                analyzer_results = deidentifier.analyze_text(
                    text=document,
                    entities=self.supported_entities,
                    language=self.language,
                )
                # analyzer_results is a list of lists (one per supported entity split by the analyzer).
                total_count = sum(len(entity_list) for entity_list in analyzer_results)

                # Break down the counts by entity type.
                counts_by_entity = {}
                for entity_list in analyzer_results:
                    for entity in entity_list:
                        # Assuming each analyzer result has an attribute "entity_type"
                        entity_type = entity.entity_type
                        counts_by_entity[entity_type] = counts_by_entity.get(entity_type, 0) + 1

                result.append({"pii_count": total_count, "pii_by_entity": counts_by_entity})
            except Exception:
                logger.exception(f"Error processing document in partition {partition_info.get('number') if partition_info else 'unknown'}")
                result.append({"error": True})
        return pd.Series(result, index=text.index)

    def load_deidentifier(self):
        """
        Helper function to load the PII deidentifier. Similar to the one used in PiiModifier.
        """
        import spacy
        from nemo_curator.pii.algorithm import PiiDeidentifier
        from nemo_curator.pii.constants import DEFAULT_MAX_DOC_SIZE

        if self.device == "gpu":
            spacy.require_gpu()
        deidentifier = PiiDeidentifier(
            language=self.language,
            supported_entities=self.supported_entities,
            # We are not performing redaction here; the action is not used.
            anonymize_action="redact",
            **self.kwargs,
        )
        # Ensure the underlying spacy model can handle lengthy documents.
        deidentifier.analyzer.nlp_engine.nlp[deidentifier.language].max_length = DEFAULT_MAX_DOC_SIZE
        return deidentifier
