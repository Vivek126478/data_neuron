import re
import string
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Using DistilBERT for lightweight transformer semantics
MODEL_NAME = 'distilbert-base-uncased'

class SimilarityModel:
    def __init__(self, model_name: str = MODEL_NAME):
        try:
            logger.info(f"Loading DistilBERT model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Set model to evaluation mode and use CPU for inference
            self.model.eval()
            logger.info("DistilBERT model loaded successfully in evaluation mode.")
        except Exception as e:
            logger.error(f"Fatal Error: Could not load model '{model_name}'. {e}")
            self.model = None
            self.tokenizer = None
            
    # Basic preprocessing to clean the text
    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
            
        text = text.lower() 
        text = text.translate(str.maketrans('', '', string.punctuation)) 
        text = re.sub(r'\s+', ' ', text).strip()  
        return text

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        """
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity score between two texts using DistilBERT embeddings
        Returns float between 0.0 and 1.0
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model is not loaded. Returning score 0.0")
            return 0.0

        try:
            clean_text1 = self._preprocess_text(text1)
            clean_text2 = self._preprocess_text(text2)
            
            # Tokenize sentences with limited length for memory efficiency
            encoded_input = self.tokenizer(
                [clean_text1, clean_text2], 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=128  # Limit sequence length for memory efficiency
            )
            
            # Compute token embeddings with no gradient calculation
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform mean pooling to get sentence embeddings
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings to unit length
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            # Compute cosine similarity between the two sentence embeddings
            cosine_sim = torch.mm(sentence_embeddings, sentence_embeddings.transpose(0, 1))
            score = cosine_sim[0][1].item()
            
            # Convert from [-1, 1] to [0, 1] range and clip
            normalized_score = (score + 1) / 2
            final_score = max(0.0, min(normalized_score, 1.0))
            
            logger.info(f"Similarity calculated: {final_score}")
            return final_score
        
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

# --- Global Instance ---
logger.info("Initializing global model instance...")
model_instance = SimilarityModel(model_name=MODEL_NAME)