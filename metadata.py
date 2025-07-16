# nemo_curator/metadata/metadata_generator.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Literal


class MetadataGenerator(ABC):
    def __init__(self):
        super().__init__()
        self._name = self.__class__.__name__

    @abstractmethod
    def generate_metadata(self, text: str) -> Dict[str, Any]:
        """
        Generate metadata from text document.
        
        Args:
            text (str): The text content to analyze
            
        Returns:
            Dict[str, Any]: Dictionary with metadata keys and values
        """
        pass

    @property
    def backend(self) -> Literal["pandas", "cudf", "any"]:
        """
        The dataframe backend the generator operates on.
        Can be 'pandas', 'cudf', or 'any'. Defaults to 'pandas'.
        
        Returns:
            str: A string representing the dataframe backend the generator needs as input
        """
        return "pandas"
        
# nemo_curator/metadata/basic_stats.py
import re
from typing import Dict, Any

from nemo_curator.metadata.metadata_generator import MetadataGenerator
from nemo_curator.utils.text_utils import get_sentences, get_paragraphs, get_words


class BasicStatsGenerator(MetadataGenerator):
    """Generate basic statistical metadata about text documents."""
    
    def generate_metadata(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "avg_word_length": 0,
                "avg_sentence_length": 0,
            }
        
        # Get basic components
        words, _ = get_words(text)
        sentences = get_sentences(text)
        paragraphs = get_paragraphs(text)
        
        # Calculate stats
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        
        # Calculate averages
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
        }
        
# nemo_curator/metadata/readability.py
import math
from typing import Dict, Any

from nemo_curator.metadata.metadata_generator import MetadataGenerator
from nemo_curator.utils.text_utils import get_sentences, get_words


class ReadabilityGenerator(MetadataGenerator):
    """Generate readability metrics for text documents."""
    
    def generate_metadata(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
                "smog_index": 0,
                "automated_readability_index": 0,
            }
        
        # Get basic components
        words, _ = get_words(text)
        sentences = get_sentences(text)
        
        # Count syllables (simplified approach)
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            vowels = "aeiouy"
            count = 0
            prev_is_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
            if word.endswith('e'):
                count -= 1
            return max(1, count)
        
        word_count = len(words)
        sentence_count = len(sentences)
        syllable_count = sum(count_syllables(word) for word in words)
        char_count = sum(len(word) for word in words)
        
        # Calculate readability scores
        if word_count == 0 or sentence_count == 0:
            return {
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
                "smog_index": 0,
                "automated_readability_index": 0,
            }
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        
        # SMOG Index
        smog_index = 1.043 * math.sqrt(30 * (syllable_count / sentence_count)) + 3.1291
        
        # Automated Readability Index
        automated_readability_index = 4.71 * (char_count / word_count) + 0.5 * (word_count / sentence_count) - 21.43
        
        return {
            "flesch_reading_ease": flesch_reading_ease,
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "smog_index": smog_index,
            "automated_readability_index": automated_readability_index,
        }
        
# nemo_curator/metadata/language_detector.py
from typing import Dict, Any

from nemo_curator.metadata.metadata_generator import MetadataGenerator


class LanguageDetector(MetadataGenerator):
    """Detect language of text documents using fasttext."""
    
    def __init__(self, model_path=None):
        super().__init__()
        self.model = None
        self.model_path = model_path
        
    def _load_model(self):
        try:
            import fasttext
            if self.model_path:
                self.model = fasttext.load_model(self.model_path)
            else:
                # Download model if not provided
                import os
                import urllib.request
                model_path = os.path.join(os.path.expanduser("~"), ".fasttext", "lid.176.bin")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve(
                        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                        model_path
                    )
                self.model = fasttext.load_model(model_path)
        except ImportError:
            raise ImportError("fasttext is required for language detection. Install with: pip install fasttext")
    
    def generate_metadata(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {"language": "unknown", "language_confidence": 0.0}
        
        if not self.model:
            self._load_model()
        
        # Get prediction
        predictions = self.model.predict(text.replace("\n", " "))
        language = predictions[0][0].replace("__label__", "")
        confidence = float(predictions[1][0])
        
        return {
            "language": language,
            "language_confidence": confidence,
        }
        
# nemo_curator/metadata/sentiment_analyzer.py
from typing import Dict, Any

from nemo_curator.metadata.metadata_generator import MetadataGenerator


class SentimentAnalyzer(MetadataGenerator):
    """Analyze sentiment of text documents."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = None
    
    def _load_analyzer(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError("vaderSentiment is required for sentiment analysis. Install with: pip install vaderSentiment")
    
    def generate_metadata(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {
                "sentiment_negative": 0.0,
                "sentiment_neutral": 0.0,
                "sentiment_positive": 0.0,
                "sentiment_compound": 0.0,
            }
        
        if not self.analyzer:
            self._load_analyzer()
        
        # Get sentiment scores
        scores = self.analyzer.polarity_scores(text)
        
        return {
            "sentiment_negative": scores['neg'],
            "sentiment_neutral": scores['neu'],
            "sentiment_positive": scores['pos'],
            "sentiment_compound": scores['compound'],
        }


# nemo_curator/metadata/topic_extractor.py
from typing import Dict, Any, List

from nemo_curator.metadata.metadata_generator import MetadataGenerator
from nemo_curator.utils.text_utils import get_words


class TopicExtractor(MetadataGenerator):
    """Extract key topics from text documents."""
    
    def __init__(self, num_topics=5):
        super().__init__()
        self.num_topics = num_topics
        self.stopwords = None
        
    def _load_stopwords(self):
        try:
            import nltk
            try:
                self.stopwords = set(nltk.corpus.stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords')
                self.stopwords = set(nltk.corpus.stopwords.words('english'))
        except ImportError:
            raise ImportError("nltk is required for topic extraction. Install with: pip install nltk")
    
    def generate_metadata(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {"topics": []}
        
        if not self.stopwords:
            self._load_stopwords()
        
        # Get words and remove stopwords
        words, _ = get_words(text)
        filtered_words = [word.lower() for word in words if word.lower() not in self.stopwords and len(word) > 3]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:self.num_topics]
        topics = [word for word, count in top_words]
        
        return {
            "topics": topics,
        }
        
# nemo_curator/modules/generate_metadata.py
from nemo_curator.datasets import DocumentDataset
from nemo_curator.metadata.metadata_generator import MetadataGenerator
from nemo_curator.modules.base import BaseModule
from nemo_curator.utils.module_utils import is_batched


class GenerateMetadata(BaseModule):
    """
    Module to generate metadata for text documents in a dataset.
    """
    
    def __init__(self, generator: MetadataGenerator, text_field: str = "text", metadata_field: str = None):
        """
        Initialize the GenerateMetadata module.
        
        Args:
            generator (MetadataGenerator): The metadata generator to use
            text_field (str): The field containing the text to analyze
            metadata_field (str, optional): The field to store the metadata in.
                If None, metadata will be stored in separate columns.
        """
        super().__init__(input_backend=generator.backend)
        self.generator = generator
        self.text_field = text_field
        self.metadata_field = metadata_field

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Generate metadata for documents in the dataset.
        
        Args:
            dataset (DocumentDataset): The dataset to generate metadata for
            
        Returns:
            DocumentDataset: The dataset with added metadata
        """
        if is_batched(self.generator.generate_metadata):
            # Handle batched processing
            metadata_dicts = dataset.df[self.text_field].map_partitions(
                self.generator.generate_metadata, meta=(None, dict)
            )
        else:
            # Handle single document processing
            metadata_dicts = dataset.df[self.text_field].apply(
                self.generator.generate_metadata, meta=(None, dict)
            )
        
        # Add metadata to dataset
        if self.metadata_field:
            # Store as a single column containing dictionaries
            dataset.df[self.metadata_field] = metadata_dicts
        else:
            # Expand metadata into separate columns
            for record in metadata_dicts:
                for key, value in record.items():
                    dataset.df[f"{self.generator._name}_{key}"] = value
                    
        return dataset
        
# nemo_curator/metadata/__init__.py
from .metadata_generator import MetadataGenerator
from .basic_stats import BasicStatsGenerator
from .readability import ReadabilityGenerator
from .language_detector import LanguageDetector
from .sentiment_analyzer import SentimentAnalyzer
from .topic_extractor import TopicExtractor

__all__ = [
    "MetadataGenerator",
    "BasicStatsGenerator",
    "ReadabilityGenerator",
    "LanguageDetector",
    "SentimentAnalyzer",
    "TopicExtractor",
]

from nemo_curator.datasets import DocumentDataset
from nemo_curator.metadata import (
    BasicStatsGenerator, 
    ReadabilityGenerator,
    LanguageDetector,
    SentimentAnalyzer,
    TopicExtractor
)
from nemo_curator.modules import GenerateMetadata, Sequential

# Create metadata generators
basic_stats = GenerateMetadata(BasicStatsGenerator())
readability = GenerateMetadata(ReadabilityGenerator())
language = GenerateMetadata(LanguageDetector())
sentiment = GenerateMetadata(SentimentAnalyzer())
topics = GenerateMetadata(TopicExtractor(num_topics=5))

# Create a sequential pipeline to apply all generators
metadata_pipeline = Sequential([
    basic_stats,
    readability,
    language,
    sentiment,
    topics
])

# Apply to a dataset
dataset = DocumentDataset(...)  # Your input dataset
enriched_dataset = metadata_pipeline(dataset)
