# BERT(S) for Relation Extraction

## Overview
A PyTorch implementation of the models for the paper ["Matching the Blanks: Distributional Similarity for Relation Learning"](https://arxiv.org/pdf/1906.03158.pdf) published in ACL 2019.  
Note: This is not an official repo for the paper.
Additional models for relation extraction, implemented here based on the paper's methodology:
- ALBERT (https://arxiv.org/abs/1909.11942)

## Requirements
Requirements: Python (3.6+), PyTorch (1.2.0), Spacy (2.1.8)  
Pre-trained BERT(S) model courtesy of HuggingFace.co (https://huggingface.co)

## Training by matching the blanks (MTB)
Run main_pretraining.py with arguments below. Pre-training data can be any .txt continuous text file.  
We use Spacy NLP to grab pairwise entities (within a window size of 40 tokens length) from the text to form relation statements for pre-training. Entities recognition are based on NER and dependency tree parsing of objects/subjects.  
The pre-training data (cnn.txt) that I've used can be downloaded [here.](https://drive.google.com/file/d/1aMiIZXLpO7JF-z_Zte3uH7OCo4Uk_0do/view?usp=sharing)

Note: Pre-training can take a long time, depending on available GPU. It is possible to directly fine-tune on the relation-extraction task and still get reasonable results, following the section below.  
ALBERT model trained on MTB with cnn.txt can be downloaded [here,](https://drive.google.com/drive/folders/1cTBi6gqPTgG1gYXuGH5LtZ3ZhD_Upy4M?usp=sharing) with MTB training results shown below.
```bash
main_pretraining.py [-h] 
	[--pretrain_data TRAIN_PATH] 
	[--batch_size BATCH_SIZE]
	[--freeze FREEZE]  
	[--gradient_acc_steps GRADIENT_ACC_STEPS]
	[--max_norm MAX_NORM]
	[--fp16 FP_16]  
	[--num_epochs NUM_EPOCHS]
	[--lr LR]
	[--model_no MODEL_NO (0: BERT ; 1: ALBERT)]
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
	[--model_no MODEL_NO (0: BERT ; 1: ALBERT)]
	[--train TRAIN]
	[--infer INFER]
```

### Inference (--infer=1)
To infer a sentence, you can annotate entity1 & entity2 of interest within the sentence with their respective entities tags [E1], [E2]. 
Example:
```bash
Type input sentence ('quit' or 'exit' to terminate):
The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor.

Sentence:  The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor.
Predicted:  Cause-Effect(e1,e2) 
```

```python
from src.tasks.infer import infer_from_trained

inferer = infer_from_trained(args, detect_entities=False)
test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
inferer.infer_sentence(test, detect_entities=False)
```
```bash
Sentence:  The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor.
Predicted:  Cause-Effect(e1,e2) 
```

The script can also automatically detect potential entities in an input sentence, in which case all possible relation combinations are inferred:
```python
inferer = infer_from_trained(args, detect_entities=True)
test2 = "After eating the chicken, he developed a sore throat the next morning."
inferer.infer_sentence(test2, detect_entities=True)
```
```bash
Sentence:  [E2]After eating the chicken[/E2] , [E1]he[/E1] developed a sore throat the next morning .
Predicted:  Other 

Sentence:  After eating the chicken , [E1]he[/E1] developed [E2]a sore throat[/E2] the next morning .
Predicted:  Other 

Sentence:  [E1]After eating the chicken[/E1] , [E2]he[/E2] developed a sore throat the next morning .
Predicted:  Other 

Sentence:  [E1]After eating the chicken[/E1] , he developed [E2]a sore throat[/E2] the next morning .
Predicted:  Other 

Sentence:  After eating the chicken , [E2]he[/E2] developed [E1]a sore throat[/E1] the next morning .
Predicted:  Other 

Sentence:  [E2]After eating the chicken[/E2] , he developed [E1]a sore throat[/E1] the next morning .
Predicted:  Cause-Effect(e2,e1) 
```

## Benchmark Results
### MTB pre-training
2) Base architecture: ALBERT base uncased (12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters)
MTB training results:
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/loss_vs_epoch_1.png) 
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/accuracy_vs_epoch_1.png) 

### SemEval2010 Task 8
1) Base architecture: BERT base uncased (12-layer, 768-hidden, 12-heads, 110M parameters)
With MTB pre-training: F1 results when trained on 100 % training data:
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/blanks_task_test_f1_vs_epoch_0.png) 

Without MTB pre-training: F1 results when trained on 100 % training data:
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/task_test_f1_vs_epoch_0.png) 

With 100 % training data, both models perform similarly, as reproduced in the paper. Yet to test cases where data is limited.

2) Base architecture: ALBERT base uncased (12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters)  
With MTB pre-training: F1 results when trained on 100 % training data:
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/blanks_task_test_f1_vs_epoch_1.png) 

Without MTB pre-training: F1 results when trained on 100 % training data:
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/task_test_f1_vs_epoch_1.png) 

For ALBERT, it looks like pretraining with MTB causes the model to overfit. Using ALBERT directly on the SemEval2010 Task 8 gives much better f1.  
It seems ALBERT's modifications: parameter-sharing across the layers & factorization of the embedding parametrization is not suitable with MTB pretraining.  

## To add
- ~~inference & results on benchmarks (SemEval2010 Task 8) with & without MTB pre-training~~
- ~~fine-tuning MTB on supervised relation extraction tasks~~
- felrel task

