## Genomic language model pretraining with the HealthOmics sequence store

In this repo we show how you can use AWS infrastructure---AWS HealthOmics and AWS Sagemaker---to
to easy and cost-effectively pre-train a genomic language model, [HyenaDNA](https://arxiv.org/pdf/2306.15794.pdf).

## Installation

1. Clone this repo in a SageMaker notebook.
2. If you don't already have your genomic data (FASTA files) in a sequence store, then use the `load-genome-to-sequence-store.ipynb` notebook to do so.
3. Use the `hyenaDNA-training.ipynb` notebook to initiate a training job.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

