import re
import string
import logging
import numpy as np
from datasketch import MinHash, MinHashLSH
import mmh3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinHashSimilarityModel:
    def __init__(self, num_perm=256):  # Increased for better accuracy
        """
        MinHash for Semantic Similarity
        Increased num_perm to 256 for better accuracy
        """
        try:
            logger.info(f"Initializing MinHash Semantic Fingerprinting Model with {num_perm} permutations...")
            
            self.num_perm = num_perm
            self.initialized = True
            
            # Common stop words to remove for better similarity detection
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
            }
            
            logger.info(f"MinHash model ready with {num_perm} permutations")
            
        except Exception as e:
            logger.error(f"Fatal Error: Could not initialize MinHash model. {e}")
            self.initialized = False

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text with stop word removal"""
        if not isinstance(text, str):
            text = str(text)
            
        text = text.lower()
        # Remove punctuation but keep words intact
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stop words
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        return ' '.join(filtered_words)

    def _text_to_shingles(self, text: str, shingle_size=2) -> set:
        """
        Convert text to shingles with multiple strategies
        Using multiple shingle sizes for better semantic capture
        """
        words = text.split()
        
        if len(words) == 0:
            return set()
        
        shingles = set()
        
        # Strategy 1: Word n-grams (captures word sequences)
        for size in [1, 2, 3]:  # Use unigrams, bigrams, trigrams
            if len(words) >= size:
                for i in range(len(words) - size + 1):
                    shingle = ' '.join(words[i:i + size])
                    if len(shingle) > 2:  # Only meaningful shingles
                        shingles.add(shingle)
        
        # Strategy 2: Character n-grams (captures subword similarities)
        full_text = text.replace(' ', '_')  # Use underscore to preserve word boundaries
        for i in range(len(full_text) - 3 + 1):
            char_shingle = full_text[i:i + 3]
            if '_' not in char_shingle or len(char_shingle.replace('_', '')) > 1:
                shingles.add(f"char_{char_shingle}")
        
        return shingles

    def _create_minhash(self, shingles: set) -> MinHash:
        """Create MinHash signature from shingles with better hashing"""
        minhash = MinHash(num_perm=self.num_perm)
        
        for shingle in shingles:
            try:
                # Use multiple hash seeds for better distribution
                for seed in [0, 42, 123, 456]:
                    hash_val = mmh3.hash(shingle, seed=seed)
                    minhash.update(hash_val.to_bytes(8, byteorder='big', signed=True))
            except Exception as e:
                continue
        
        return minhash

    def get_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using MinHash with enhanced processing
        """
        if not self.initialized:
            return 0.0

        try:
            clean_text1 = self._preprocess_text(text1)
            clean_text2 = self._preprocess_text(text2)
            
            # Handle empty texts
            if not clean_text1.strip() or not clean_text2.strip():
                return 0.0
            
            # Get shingles with multiple strategies
            shingles1 = self._text_to_shingles(clean_text1)
            shingles2 = self._text_to_shingles(clean_text2)
            
            # Calculate exact Jaccard similarity as baseline
            intersection = len(shingles1.intersection(shingles2))
            union = len(shingles1.union(shingles2))
            exact_jaccard = intersection / union if union > 0 else 0.0
            
            # Create MinHash signatures
            minhash1 = self._create_minhash(shingles1)
            minhash2 = self._create_minhash(shingles2)
            minhash_jaccard = minhash1.jaccard(minhash2)
            
            # Use weighted combination of both methods
            # MinHash is better for large sets, exact is better for small sets
            if union < 20:  # Small text, trust exact Jaccard more
                combined_score = 0.7 * exact_jaccard + 0.3 * minhash_jaccard
            else:  # Larger text, trust MinHash more
                combined_score = 0.3 * exact_jaccard + 0.7 * minhash_jaccard
            
            # Apply non-linear scaling to make scores more intuitive
            if combined_score > 0.8:
                adjusted_score = 0.8 + (combined_score - 0.8) * 1.2
            elif combined_score > 0.3:
                adjusted_score = combined_score * 1.3  # Boost medium scores
            else:
                adjusted_score = combined_score * 0.9  # Slightly compress low scores
            
            # Ensure the score is between 0 and 1
            final_score = max(0.0, min(adjusted_score, 1.0))
            
            logger.info(f"Similarity: exact={exact_jaccard:.3f}, minhash={minhash_jaccard:.3f}, final={final_score:.3f}")
            return round(final_score, 4)
            
        except Exception as e:
            logger.error(f"Error in similarity calculation: {e}")
            return 0.0

# --- Global Instance ---
logger.info("Initializing MinHash global model instance...")
model_instance = MinHashSimilarityModel(num_perm=256)  # Increased for accuracy