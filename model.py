import re
import string
import logging
import unicodedata
from typing import Tuple
from sentence_transformers import CrossEncoder

#basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
Semantic similarity model based on a supervised CrossEncoder.

Key characteristics of this approach (beyond basic embeddings):
- Uses a deep, supervised bi-encoder that jointly attends to both texts.
- Symmetric scoring: we evaluate (text1, text2) and (text2, text1) and average.
- Lightweight preprocessing for robustness to casing, punctuation, and odd unicode.
"""

# Using a cross-encoder pre-trained model from Sentence-Transformers that
# directly predicts a similarity score in [0, 1] for STS.
MODEL_NAME = 'cross-encoder/stsb-distilroberta-base'


class SimilarityModel:
    def __init__(self, model_name: str):
        try:
            logger.info(f"Loading Cross-Encoder model: {model_name}...")
            self.model = CrossEncoder(model_name)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Fatal Error: Could not load model '{model_name}'. {e}")
            self.model = None
            
    # Basic yet robust preprocessing to normalize text without losing semantics
    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        # Unicode normalize to reduce diacritics or odd whitespace
        text = unicodedata.normalize('NFKC', text)
        # Lowercase for case-insensitive matching
        text = text.lower()
        # Remove most punctuation but keep intra-word apostrophes and hyphens
        text = re.sub(r"[\u2018\u2019]", "'", text)  # curly quotes to straight
        text = re.sub(r"[\u2013\u2014]", "-", text)  # ndash/mdash to hyphen
        text = ''.join(
            ch for ch in text
            if (ch not in string.punctuation) or (ch in {"'", '-'})
        )
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _predict_pair(self, a: str, b: str) -> float:
        """Predict similarity score for an ordered pair (a, b)."""
        model_input = [[a, b]]
        score_array = self.model.predict(model_input, show_progress_bar=False)
        return float(score_array[0])

    def get_similarity_score(self, text1: str, text2: str) -> float:
        if self.model is None:
            logger.error("Model is not loaded. Returning score 0.0")
            return 0.0

        try:
            clean_text1 = self._preprocess_text(text1)
            clean_text2 = self._preprocess_text(text2)

            # Symmetric prediction for extra robustness
            score_ab = self._predict_pair(clean_text1, clean_text2)
            score_ba = self._predict_pair(clean_text2, clean_text1)
            score = (score_ab + score_ba) / 2.0

            # Clip the score to [0, 1]
            score = max(0.0, min(score, 1.0))
            # Optional rounding for API stability
            return float(round(score, 4))

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

# --- Global Instance ---
logger.info("Initializing global model instance...")
model_instance = SimilarityModel(model_name=MODEL_NAME)