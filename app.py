from flask import Flask, request, jsonify
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

# --- API Endpoints ---

@app.route('/')
def home():
    """To check whether the API is running."""
    return jsonify({
        "message": "Semantic Similarity API is running", 
        "algorithm": "MinHash Semantic Fingerprinting",
        "endpoint": "POST /calculate-similarity"
    })

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
        }), 400
    
    try:
        text1 = data['text1']
        text2 = data['text2']

        # Validate input types
        if not isinstance(text1, str) or not isinstance(text2, str):
            return jsonify({
                "error": "Both 'text1' and 'text2' must be strings."
            }), 400

        # 3. Calculate the similarity score using our MinHash model (Part A)
        score = model_instance.get_similarity_score(text1, text2)
        
        # 4. Format the response exactly as required
        response_body = {
            "similarity score": score
        }
        
        logger.info(f"Successfully calculated MinHash similarity. Score: {score}")
        return jsonify(response_body), 200

    except Exception as e:
        logger.error(f"Internal Server Error during calculation: {str(e)}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_initialized": model_instance.initialized,
        "algorithm": "MinHash Semantic Fingerprinting"
    })

# --- Main execution block ---
if __name__ == '__main__':
    logger.info("Starting Flask server with MinHash algorithm...")
    app.run(host='0.0.0.0', port=5000, debug=False)