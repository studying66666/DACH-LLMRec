"""DACH-LLMRec recommendation prototype."""

from .embeddings import HashEmbeddingProvider
from .recommender import DACHLLMRecommender

__all__ = ["DACHLLMRecommender", "HashEmbeddingProvider"]
