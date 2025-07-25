import pandas as pd
from nemo_curator.modifiers.doc_modifier import DocumentModifier
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import load_object_on_worker

class PIICounter(DocumentModifier):
    """
    A class to count PII entities detected in JSONL or Parquet documents.

    This class inherits from DocumentModifier and overrides the modify_document method
    to count the PII entities in the supplied text.

    Example:
        dataframe = pd.DataFrame({'text': ['Sarah and Ryan went out to play', 'Jensen is the CEO of NVIDIA']})
        dataset = DocumentDataset.from_pandas(dataframe, npartitions=1)

        pii_counter = PIICounter(language='en', supported_entities=['PERSON', 'EMAIL_ADDRESS'])
        modify = Modify(pii_counter)
        result = modify(dataset)
        print(result.df)
    """

    def __init__(self, language: str = "en", supported_entities: list[str] | None = None, **kwargs):
        super().__init__()
        self.language = language
        self.supported_entities = supported_entities or []
        self.kwargs = kwargs

    @batched
    def modify_document(self, text: pd.Series, partition_info: dict | None = None) -> pd.Series:
        import logging

        logging.basicConfig(
            format="%(asctime)s %(levelname)s:%(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        deidentifier = load_object_on_worker("deidentifier", self.load_deidentifier, {})
        try:
            results = deidentifier.analyze_text_batch(
                texts=text.tolist(),
                entities=self.supported_entities,
                language=self.language,
                batch_size=32,  # Adjust batch size as needed
            )
        except Exception:
            logger.exception(f"Error occurred during PII detection in partition {partition_info}")
            return pd.Series([{}] * len(text), index=text.index)

        # Count entities for each document
        counts = []
        for document_results in results:
            entity_counts = {}
            for result in document_results:
                entity_type = result.entity_type
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            counts.append(entity_counts)

        return pd.Series(counts, index=text.index)

    def load_deidentifier(self) -> "PiiDeidentifier":  # noqa: F821
        """
        Helper function to load the de-identifier.
        """
        from nemo_curator.pii.algorithm import PiiDeidentifier

        return PiiDeidentifier(
            language=self.language,
            supported_entities=self.supported_entities,
            anonymize_action="analyze",  # Only analyze, no anonymization
            **self.kwargs,
        )
