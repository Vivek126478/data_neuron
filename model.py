import re
import string
import logging
from sentence_transformers import CrossEncoder

#basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#Using a cross encoder pre trained model from sentence transformers
#It is supervised model perfectly aligned to give outputs between 0 and 1
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
            
# Basic preprocessing to clean the text
    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
            
        text = text.lower() 
        text = text.translate(str.maketrans('', '', string.punctuation)) 
        text = re.sub(r'\s+', ' ', text).strip()  
        return text

    def get_similarity_score(self, text1: str, text2: str) -> float:
       
        if self.model is None:
            logger.error("Model is not loaded. Returning score 0.0")
            return 0.0

        try:
            clean_text1 = self._preprocess_text(text1)
            clean_text2 = self._preprocess_text(text2)
            
            model_input = [[clean_text1, clean_text2]]
            
            score_array = self.model.predict(model_input, show_progress_bar=False)
            
            score = float(score_array[0])
            
            # Clip the score to be strictly between 0.0 and 1.0
            return max(0.0, min(score, 1.0))
        
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

# --- Global Instance ---
logger.info("Initializing global model instance...")
model_instance = SimilarityModel(model_name=MODEL_NAME)