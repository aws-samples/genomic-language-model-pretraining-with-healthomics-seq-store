import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

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
    print("---------------input_fn--------------------------------")
    print(request_body)
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        # Handle other content-types here or raise an exception
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Generate predictions from the model and input data.
    """

    try:
        print("---------------predict_fn--------------------------------")
        print(input_data)
        inputs = torch.tensor(input_data, dtype=torch.long)  # Ensure the data type matches your model's requirements

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            print("Data moved to GPU.")
        else:
            print("Data use CPU.")
        
        # Generate predictions
        with torch.no_grad():
            outputs = model(input_ids=inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            last_hidden_states = hidden_states[-1]
            embeddings = last_hidden_states.to(dtype=torch.float32).cpu().numpy()

            print("---------------predictions--------------------------------")
            print(embeddings)
            
        return embeddings
        
    except Exception as e:
        # Handle exceptions or log error details here
        print(f"An error occurred during prediction: {e}")
        # Depending on your application, you may want to re-raise the exception,
        # return a default value, or return an error message
        return None  # or any appropriate error indication for your application

def output_fn(prediction, response_content_type):
    """
    Format the prediction output.
    """
    print("---------------output_fn--------------------------------")
    prediction = prediction.tolist()
    print(len(prediction))
    return json.dumps(prediction)
    # try:
    #     if response_content_type == 'application/json':
    #         if isinstance(prediction, np.ndarray):
    #             prediction = prediction.tolist()
    #         return json.dumps(prediction), response_content_type
    #     else:
    #         # Handle other content-types here or raise an exception
    #         raise ValueError(f"Unsupported response content type: {response_content_type}")

    # except Exception as e:
    #     # Handle exceptions or log error details here
    #     print(f"An error occurred during prediction: {e}")
    #     # Depending on your application, you may want to re-raise the exception,
    #     # return a default value, or return an error message
    #     return None  # or any appropriate error indication for your application

