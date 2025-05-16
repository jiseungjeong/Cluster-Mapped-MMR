import numpy as np
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
import os
import pickle
import logging

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Class for processing sentence embeddings"""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        cache_dir: str = "./data/embeddings",
    ):
        """
        Initialize embedding processor

        Args:
            model_name: SentenceTransformer model name
            cache_dir: Directory for storing embedding cache
        """
        logger.info(f"Initializing embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, dataset_name: str, content_type: str) -> str:
        """Return embedding cache file path"""
        return os.path.join(
            self.cache_dir,
            f"{dataset_name}_{content_type}_{self.model_name.replace('/', '_')}.pkl",
        )

    def embed_questions(
        self, data: List[Dict], dataset_name: str, force_recompute: bool = False
    ) -> np.ndarray:
        """
        Compute embeddings for questions or load from cache

        Args:
            data: List of question data
            dataset_name: Dataset name
            force_recompute: Whether to recompute even if cache exists

        Returns:
            Embedding array (n_samples, embedding_dim)
        """
        cache_path = self._get_cache_path(dataset_name, "questions")

        # Check cache
        if os.path.exists(cache_path) and not force_recompute:
            logger.info(f"Loading embeddings from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Extract question texts
        questions = [item["question"] for item in data]
        logger.info(f"Computing embeddings for {len(questions)} questions...")

        # Compute embeddings
        embeddings = self.model.encode(questions, show_progress_bar=True)

        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)

        logger.info(f"Embeddings saved: {cache_path}")
        return embeddings

    def compute_similarity(
        self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query embedding and corpus embeddings

        Args:
            query_embedding: Query embedding vector
            corpus_embeddings: Corpus embedding array

        Returns:
            Cosine similarity array
        """
        # Normalize vectors
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(
            corpus_embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarity
        return np.dot(corpus_embeddings, query_embedding)
