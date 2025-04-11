import os
import json
import torch
import logging
from evo2 import Evo2

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Global model variables - cache for multiple models
model_cache = {}
default_model_name = os.environ.get('DEFAULT_MODEL_NAME', 'evo2_7b')

def model_fn(model_dir):
    """
    Load the default model for inference
    """
    global model_cache
    
    # Get default model name from environment variable
    logger.info(f"Loading default model: {default_model_name}")
    
    # Load default model
    model_cache[default_model_name] = Evo2(default_model_name)
    logger.info(f"Default model {default_model_name} loaded successfully")
    
    return model_cache

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_cache):
    """
    Generate predictions based on input data
    """
    logger.info("Generating predictions")
    
    # Extract parameters from input
    input_seqs = input_data.get('input_seqs', [])
    n_tokens = input_data.get('n_tokens', 500)
    temperature = input_data.get('temperature', 1.0)
    model_name = input_data.get('model_name', default_model_name)
    
    # Check if we need to load a new model
    if model_name not in model_cache:
        logger.info(f"Loading new model: {model_name}")
        try:
            model_cache[model_name] = Evo2(model_name)
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise ValueError(f"Could not load model {model_name}: {str(e)}")
    
    # Get the requested model
    model = model_cache[model_name]
    logger.info(f"Using model: {model_name}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Generate sequences
    with torch.no_grad():
        generations = model.generate(
            input_seqs,
            n_tokens=n_tokens,
            temperature=temperature,
        )
    
    return generations

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction output
    """
    logger.info("Preparing output")
    
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {response_content_type}")
