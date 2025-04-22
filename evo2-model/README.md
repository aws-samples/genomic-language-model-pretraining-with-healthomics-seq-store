# Evo 2

Many thanks to [Adam Stanley](https://github.com/astanley-work) (dmstn@amazon.com) and [Ashley Moon](https://github.com/DoomishFox) (chutkasc@amazon.com) for their contributions.

---

[Evo 2](https://github.com/ArcInstitute/evo2/blob/main/README.md) is the successor to [Evo](https://arcinstitute.org/news/blog/evo) that has a longer context window (1 million bases or tokens) and is trained on more data (9.3 trillion bases) across multiple kingdoms.

Here we show how to deploy Evo 2 to an EC2 instance.

## Installation

Create a new EC2 instance using the `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Ubuntu 22.04) 20250406` AMI (currently this has ID `ami-0af5ce3bbdd2c86d3`) and a `p5.48xlarge` or `ml.p5e.48xlarge` instance type. Make sure you have plenty of disk space; to use the 40B weights model we suggest 256Gb of disk space.

Once you have logged into the instance, start by updating system packages

```
sudo add-apt-repository -y ppa:deadsnakes/ppa # makes stable version of Python available on Ubuntu
sudo apt update
```

Next, install Python 3.12:

```
sudo apt install -y python3.12
sudo apt install -y python3.12-venv
sudo apt install -y python3.12-dev
sudo apt-get install -y build-essential
```

Then create a Python virtual environment:

```
python3.12 -m venv venv
source venv/bin/activate
```

In subsequent logins you'll need to rerun the activate script. Then, install some Python dependencies:

```
pip install --upgrade pip
pip install ninja cmake pybind11 numpy psutil setuptools wheel
```

Also, install the correct version of `transformer-engine` (note that we need a specific version of PyTorch):

```
pip uninstall -y transformer_engine
pip install torch==2.6
pip install transformer-engine[pytorch]==1.13
```

Finally, install Evo 2 itself:

```
git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git
cd evo2
pip install .
```
This will take a few minutes. 

## Testing

Once it is installed you can run the bundled test script. We're using the 7 billion weights model, note that the 
40 billion weights requires considerably more disc space, around 200Gb.

```
python3.12 ./test/test_evo2.py --model_name evo2_7b
```

And you should see output like this:

```
(venv) ubuntu@ip-172-31-1-242:~/evo2$ python3.12 ./test/test_evo2.py --model_name evo2_7b
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 87.0/87.0 [00:00<00:00, 1.02MB/s]
.gitattributes: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.52k/1.52k [00:00<00:00, 24.9MB/s]
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.02k/1.02k [00:00<00:00, 11.1MB/s]
evo2_7b.pt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13.8G/13.8G [01:09<00:00, 199MB/s]
Fetching 4 files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [01:09<00:00, 17.34s/it]
Found complete file in repo: evo2_7b.pt
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:01<00:00, 20.56it/s]
Extra keys in state_dict: {'blocks.24.mixer.dense._extra_state', 'blocks.17.mixer.attn._extra_state', 'blocks.24.mixer.attn._extra_state', 'blocks.2.mixer.mixer.filter.t', 'blocks.3.mixer.attn._extra_state', 'blocks.13.mixer.mixer.filter.t', 'blocks.31.mixer.dense._extra_state', 'blocks.10.mixer.dense._extra_state', 'blocks.17.mixer.dense._extra_state', 'blocks.9.mixer.mixer.filter.t', 'blocks.3.mixer.dense._extra_state', 'blocks.20.mixer.mixer.filter.t', 'blocks.30.mixer.mixer.filter.t', 'blocks.6.mixer.mixer.filter.t', 'unembed.weight', 'blocks.23.mixer.mixer.filter.t', 'blocks.27.mixer.mixer.filter.t', 'blocks.16.mixer.mixer.filter.t', 'blocks.10.mixer.attn._extra_state', 'blocks.31.mixer.attn._extra_state'}

Sequence Results:
Sequence 1: Loss = 0.182, Accuracy = 93.53%
Sequence 2: Loss = 0.354, Accuracy = 86.32%
Sequence 3: Loss = 0.500, Accuracy = 80.18%
Sequence 4: Loss = 0.355, Accuracy = 85.36%

Mean Loss: 0.348
Mean Accuracy: 86.346%

Test Passed! Loss matches expected 0.348
```
