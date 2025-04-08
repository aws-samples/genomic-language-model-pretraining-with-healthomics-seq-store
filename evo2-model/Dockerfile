FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    make \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel build

# Install sagemaker-inference for serving
RUN pip install sagemaker-inference

# Copy repository files from local directory
COPY . .
RUN git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git

WORKDIR /app/evo2
# Install application
RUN pip install .

# Copy the inference script
COPY inference.py /opt/program/inference.py

# Set up environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:\${PATH}"
ENV DEFAULT_MODEL_NAME="evo2_7b"

# SageMaker uses /opt/ml for input/output/model
WORKDIR /opt/program

# Set the entrypoint for SageMaker container
ENTRYPOINT ["python", "-m", "sagemaker_inference.serve"]