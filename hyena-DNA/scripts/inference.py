import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[HyenaDNA Inference] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    """

    print("-------------------COFIGS--------------------------------")
    print(os.getenv('TS_MAX_REQUEST_SIZE'))
    print(os.getenv('TS_MAX_RESPONSE_SIZE'))
    print(os.getenv('TS_DEFAULT_RESPONSE_TIMEOUT'))
    
    print("Loading the model-----------------------------------------------")
    MODEL_ID = os.getenv('MODEL_ID', 'LongSafari/hyenadna-small-32k-seqlen-hf')
    print(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
    checkpoint = torch.load(model_dir + '/checkpoint.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Model moved to GPU.")
    else:
        print("Using CPU for inference.")
    
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    """
    Process the input data. Assumes the input data is in a JSON format.
    """
    logger.info("input_fn Request Body.....................")
    logger.info(request_body)
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Generate predictions from the model and input data.
    """
    logger.info("predict_fn Input Data.....................")
    logger.info(input_data)
    
    inputs = torch.tensor(input_data, dtype=torch.long)
    device = "cuda"

    inputs = inputs.to(device)
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(input_ids=inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        last_hidden_states = hidden_states[-1]
        embeddings = last_hidden_states.to(dtype=torch.float32).cpu().numpy()

    return embeddings
        

def output_fn(prediction, response_content_type):
    """
    Format the prediction output.
    """
    logger.info("output_fn Prediction.....................")
    logger.info(prediction)
    
    if response_content_type == 'application/json':
        return json.dumps({'embeddings': prediction.tolist()})
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
