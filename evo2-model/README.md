# Evo 2

Evo 2 is the successor to Evo that has a longer context window (1 million bases or tokens) and is trained on more data (9.3 trillion bases).

Here we show how to deploy Evo 2 to a Sagemaker `Predictor`.

## Setup

First, use the [Dockerfile](Dockerfile) to create a customized Docker image that is based on a AWS [Deep Learning Container](https://github.com/aws/deep-learning-containers).

The resulting image is large so make sure you have an instance with at least 256Gb of disc. You may need to [install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), and you will need to authenticate the CLI with an IAM Role. Then you can do something like this to build the image and push it to an ECR repository (you'll need to create this):

```
export ECR_REGISTRY=763104351884.dkr.ecr.us-east-1.amazonaws.com # update this for your registry
# Note the 'sudo' is necessary:
aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin $ECR_REGISTRY
sudo docker build -t evo2 .
export ECR_REPO=${ECR_REGISTRY}/evo2:latest # replace this with your repository's URL
sudo docker tag evo2:latest $ECR_REPO
sudo docker push $ECR_REPO
```

## Creating and running the predictor

Now that we have a customized deep learning image, use the [Jupyter notebook](evo2.ipynb) to create and run a `Predictor`.