# BERT for Relation Extraction

## Overview
A PyTorch implementation of the models for the paper ["Matching the Blanks: Distributional Similarity for Relation Learning"](https://arxiv.org/pdf/1906.03158.pdf) published in ACL 2019.

## Requirements
Requirements: Python (3.6+), PyTorch (1.2.0), Spacy (2.1.8)
Pre-trained BERT model courtesy of HuggingFace.co (https://huggingface.co)

## Training by matching the blanks (MTB)
Run main_pretraining.py with arguments below. Pre-training data can be any .txt continuous text file.
```bash
main_pretraining.py [-h] 
	[--pretrain_data TRAIN_PATH] 
	[--batch_size BATCH_SIZE]
	[--gradient_acc_steps GRADIENT_ACC_STEPS]
	[--max_norm MAX_NORM]
	[--fp16 FP_16]  
	[--num_epochs NUM_EPOCHS]
	[--lr LR]
	[--model_no MODEL_NO]
```

## Fine-tuning on SemEval2010 Task 8
Run main_task.py with arguments below. Requires SemEval2010 Task 8 dataset, available [here.](https://github.com/sahitya0000/Relation-Classification/blob/master/corpus/SemEval2010_task8_all_data.zip)

```bash
main_task.py [-h] 
	[--train_data TRAIN_DATA]
	[--test_data TEST_DATA]
	[--use_pretrained_blanks USE_PRETRAINED_BLANKS]
	[--num_classes NUM_CLASSES] 
	[--batch_size BATCH_SIZE]
	[--gradient_acc_steps GRADIENT_ACC_STEPS]
	[--max_norm MAX_NORM]
	[--fp16 FP_16]  
	[--num_epochs NUM_EPOCHS]
	[--lr LR]
	[--model_no MODEL_NO]
```

## To add
- inference & results on benchmarks (SemEval2010 Task 8) with & without MTB pre-training 
- ~~fine-tuning MTB on supervised relation extraction tasks~~

