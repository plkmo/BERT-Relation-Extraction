# BERT for Relation Extraction

## Overview
A PyTorch implementation of the models for the paper ["Matching the Blanks: Distributional Similarity for Relation Learning"](https://arxiv.org/pdf/1906.03158.pdf) published in ACL 2019.

## Requirements
Requirements: Python (3.6+), PyTorch (1.2.0), Spacy (2.1.8)
Pre-trained BERT model courtesy of HuggingFace.co (https://huggingface.co)

## Training by matching the blanks (MTB)
Run train.py with arguments below. Pre-training data can be any .txt continuous text file.
```bash
main.py [-h] 
	[--pretrain_data TRAIN_PATH] 
	[--batch_size BATCH_SIZE]
	[--gradient_acc_steps GRADIENT_ACC_STEPS]
	[--max_norm MAX_NORM]
	[--fp16 FP_16]  
	[--num_epochs NUM_EPOCHS]
	[--lr LR]
	[--model_no MODEL_NO]
```

## To add
- inference on benchmarks with & without MTB pre-training 
- fine-tuning MTB on supervised relation extraction tasks

