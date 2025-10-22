from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys

# --- Setup Logging ---
logger = logging.getLogger(__name__)

# we are importing model_instance from model.py (Part A)
try:
    from model import model_instance
except ImportError:
    logger.error("Fatal Error: model.py not found or model_instance failed to load.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Fatal Error on model import: {e}")
    sys.exit(1)


# --- Initialize Flask App (Part B) ---
app = Flask(__name__)
# Enable CORS for all routes (useful for browser-based clients)
CORS(app)

# --- API Endpoints ---

@app.route('/')
def home():
    """To check whether the API is running."""

    return "Semantic Similarity API is running. Use /calculate-similarity endpoint."

@app.route('/health', methods=['GET'])
def health():
    """Lightweight health check endpoint for uptime probes."""
    return jsonify({"status": "ok"}), 200

@app.route('/calculate-similarity', methods=['POST'])
def calculate_similarity():
    """
    accepts json body with 'text1' and 'text2' keys
    returns json with 'similarity score' key
    """
    logger.info("Received request for /calculate-similarity")
    
    # 1. Get the JSON data from the request body
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("No JSON payload received.")
    except Exception as e:
        logger.warning(f"Bad Request: Could not parse JSON. Error: {e}")
        return jsonify({"error": "Invalid request. Body must be valid JSON."}), 400

    # 2. Validate the input keys
    if 'text1' not in data or 'text2' not in data:
        logger.warning("Bad Request: Missing 'text1' or 'text2' key.")
        return jsonify({
            "error": "Invalid input. Request body must be a JSON object "
                     "with 'text1' and 'text2' keys."
        }), 400  # Bad Request
    
    try:
        text1 = data['text1']
        text2 = data['text2']

        # 3. Calculate the similarity score using our model (Part A)
        score = model_instance.get_similarity_score(text1, text2)
        
        # 4. Format the response exactly as required
        response_body = {
            "similarity score": score
        }
        
        logger.info(f"Successfully calculated similarity. Score: {score}")
        return jsonify(response_body), 200

    except Exception as e:
        logger.error(f"Internal Server Error during calculation: {str(e)}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# --- Main execution block ---
if __name__ == '__main__':
    logger.info("Starting Flask server for local testing...")
    app.run(host='0.0.0.0', port=5000, debug=False)