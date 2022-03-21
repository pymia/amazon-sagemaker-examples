- [ ] add the files in search result in https://github.com/aws/amazon-sagemaker-examples/find/main to ## SageMaker Features section
- [ ] add industries
---
# Hugging Face on SageMaker

[Hugging Face](https://huggingface.co/) is an open-source that build, train and deploy state of the art models powered by the reference open source in machine learning. With Hugging Face AWS Deep Learning Containers(DLC), you can use Hugging Face for both training and inference. 

DLC support Amazon SageMaker Features like distributed training, training compiler, debugger profiler, managed spot training, real-time inference, asynchronous inference, serverless inference, and batch transform.

In this index file, you can visit Hugging Face on SageMaker Resources in one place.

## Use Case

- Product Review Sentiment Analysis
    - [Fine-tune Deploy Bert With Amazon Sagemaker For Hugging Face](https://github.com/aws-samples/finetune-deploy-bert-with-amazon-sagemaker-for-hugging-face), use the pre-trained DistilBERT model with the Amazon Reviews Polarity dataset.

- News Classification
    - [Classify News with Hugging Face on Amazon SageMaker](https://github.com/aws-samples/classify-news-amazon-sagemaker-hugging-face), fine-tuning and deploying state of the art(SOTA) models with public DLC images on Amazon SageMaker.

- Protein Classification
  - [Amazon Sagemaker Protein Classification](https://github.com/aws-samples/amazon-sagemaker-protein-classification), implementation of Protein Classification based on subcellular localization using ProtBert(Rostlab/prot_bert_bfd_localization) model from Hugging Face library, based on BERT model trained on large corpus of protein sequences.

- Sentence Similarity
    - [Amazon Sagemaker Sentence Similarity Hugging Face](https://github.com/aws-samples/amazon-sagemaker-sentence-similarity-hugging-face), find Similar Industrial Accidents using Sentence Transformers in PyTorch using Amazon SageMaker

- Paraphrased Text Identification
    - [Identify Paraphrased Text With Huggingface On Amazon Sagemaker](https://github.com/aws-samples/identify-paraphrased-text-with-huggingface-on-amazon-sagemaker), by identifying sentence paraphrases, a text summarization system could remove redundant information. Another application is to identify plagiarized documents. 

- Question And Answer(Q&A)
  - [Sagemaker HuggingFace NLP](https://github.com/aws-samples/sagemaker-huggingface-nlp), download the distilbert-base-uncased-distilled-squad model and SQuAD dataset, fine-tune and deploy to Amazon SageMaker Endpoint.

- Wav2Vec2
    - [Fine Tune And Deploy Wav2vec2 HuggingFace](https://github.com/aws-samples/amazon-sagemaker-fine-tune-and-deploy-wav2vec2-huggingface) shares how to work with TIMIT audio dataset with HuggingFace DLC.


## Tutorial

- [Hugging Face Workshop](https://github.com/aws-samples/hugging-face-workshop), a 90-minute hands on workshop about Hugging Face on SageMaker.

- [Amazon Sagemaker Workshop For HuggingFace](https://github.com/aws-samples/amazon-sagemaker-workshop-for-huggingface), a workshop for running HuggingFace Models on Amazon SageMaker.

## SageMaker Features
### Distributed Training 
### Training Compiler 
### Debugger Profiler 
### Managed Spot Training 
### Real Time Inference
### Multi-Model Deployment
- [amazon-sagemaker-deploy-nlp-huggingface](https://github.com/aws-samples/amazon-sagemaker-deploy-nlp-huggingface) contains examples of deploying Hugging Face models with PyTorch TorchServe on Amazon SageMaker. It includes both single model and multi-model deployments
### Serverless Inference 
### Asynchronous Inference
- Asynchronous Inference endpoints queue incoming requests. They’re ideal for workloads where the request sizes are large (up to 1 GB) and inference processing times are in the order of minutes (up to 15 minutes). Asynchronous inference enables you to save on costs by auto scaling the instance count to zero when there are no requests to process.
- [Amazon Sagemaker Asynchronous Inference Huggingface](https://github.com/aws-samples/amazon-sagemaker-asynchronous-inference-huggingface), code resource of the post: [Improve high-value research with Hugging Face and Amazon SageMaker asynchronous inference endpoints](https://aws.amazon.com/blogs/machine-learning/improve-high-value-research-with-hugging-face-and-amazon-sagemaker-asynchronous-inference-endpoints/)
### Batch Transform

### Benchmark
- [Amazon SageMaker HuggingFace Benchmark](https://github.com/aws-samples/amazon-sagemaker-huggingface-benchmark), benchmarking the trade-offs between cost, train time, and performance for fine-tuning HuggingFace models with distributed training on Amazon SageMaker.

## With Amazon AI services

- [Amazon Textract Transformer Pipeline](https://github.com/aws-samples/amazon-textract-transformer-pipeline), Post-process Amazon Textract results with Hugging Face transformer models for document understanding

## With other AWS services
- [zero-administration-inference-with-aws-lambda-for-hugging-face](https://github.com/aws-samples/zero-administration-inference-with-aws-lambda-for-hugging-face) consists of an AWS Cloud Development Kit (AWS CDK) script that automatically provisions container image-based Lambda functions that perform ML inference using pre-trained Hugging Face models. This solution also includes Amazon Elastic File System (EFS) storage that is attached to the Lambda functions to cache the pre-trained models and reduce inference latency.

- Neuron SDK
    - [Instructions to build the the HuggingFace model inference container with Neuron SDK](https://github.com/aws-samples/huggingface-model-inference), 

## Open Source library

- [SageMaker Hugging Face Inference Toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit) is an open-source library for serving 🤗 Transformers models on Amazon SageMaker. This library provides default pre-processing, predict and postprocessing for certain 🤗 Transformers models and tasks. It utilizes the SageMaker Inference Toolkit for starting up the model server, which is responsible for handling inference requests.

- [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers) are a set of Docker images for training and serving models in TensorFlow, TensorFlow 2, PyTorch, and MXNet.



## Contact


