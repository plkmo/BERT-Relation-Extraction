# BERT(S) for Relation Extraction

## Overview
A PyTorch implementation of the models for the paper ["Matching the Blanks: Distributional Similarity for Relation Learning"](https://arxiv.org/pdf/1906.03158.pdf) published in ACL 2019.  
Note: This is not an official repo for the paper.  
Additional models for relation extraction, implemented here based on the paper's methodology:  
- ALBERT (https://arxiv.org/abs/1909.11942)   
- BioBERT (https://arxiv.org/abs/1901.08746)

For more conceptual details on the implementation, please see https://towardsdatascience.com/bert-s-for-relation-extraction-in-nlp-2c7c3ab487c4

## Requirements
Requirements: Python (3.6+), PyTorch (1.2.0+), Spacy (2.1.8+)  

Pre-trained BERT models (ALBERT, BERT) courtesy of HuggingFace.co (https://huggingface.co)   
Pre-trained BioBERT model courtesy of https://github.com/dmis-lab/biobert   

To use BioBERT(biobert_v1.1_pubmed), download & unzip the [contents](https://drive.google.com/file/d/1zKTBqqrCGlclb3zgBGGpq_70Fx-qFpiU/view?usp=sharing) to ./additional_models folder.   

## Training by matching the blanks (BERT<sub>EM</sub> + MTB)
Run main_pretraining.py with arguments below. Pre-training data can be any .txt continuous text file.  
We use Spacy NLP to grab pairwise entities (within a window size of 40 tokens length) from the text to form relation statements for pre-training. Entities recognition are based on NER and dependency tree parsing of objects/subjects.  

The pre-training data taken from CNN dataset (cnn.txt) that I've used can be downloaded [here.](https://drive.google.com/file/d/1aMiIZXLpO7JF-z_Zte3uH7OCo4Uk_0do/view?usp=sharing)   
However, do note that the paper uses wiki dumps data for MTB pre-training which is much larger than the CNN dataset.   

Note: Pre-training can take a long time, depending on available GPU. It is possible to directly fine-tune on the relation-extraction task and still get reasonable results, following the section below.  

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
	[--model_no MODEL_NO (0: BERT ; 1: ALBERT ; 2: BioBERT)]  
	[--model_size MODEL_SIZE (BERT: 'bert-base-uncased', 'bert-large-uncased';   
				ALBERT: 'albert-base-v2', 'albert-large-v2';   
				BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed))]
```

## Fine-tuning on SemEval2010 Task 8 (BERT<sub>EM</sub>/BERT<sub>EM</sub> + MTB)
Run main_task.py with arguments below. Requires SemEval2010 Task 8 dataset, available [here.](https://github.com/sahitya0000/Relation-Classification/blob/master/corpus/SemEval2010_task8_all_data.zip) Download & unzip to ./data/ folder.

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
	[--model_no MODEL_NO (0: BERT ; 1: ALBERT ; 2: BioBERT)]  
	[--model_size MODEL_SIZE (BERT: 'bert-base-uncased', 'bert-large-uncased';   
				ALBERT: 'albert-base-v2', 'albert-large-v2';   
				BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed))]    
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

## FewRel Task
Download the FewRel 1.0 dataset [here.](https://drive.google.com/drive/folders/1ljobnuzxStFQJSlN4ZHMcMhZtEYaRAHy?usp=sharing) and unzip to ./data/ folder.  
Run main_task.py with argument 'task' set as 'fewrel'.
```bash
python main_task.py --task fewrel
```
Results:  
(5-way 1-shot)  
BERT<sub>EM</sub> without MTB, not trained on any FewRel data  
| Model size | Accuracy (41646 samples) |
|------------|--------------------------|
| bert-base-uncased  | 62.229 %         |
| bert-large-uncased | 72.766 %         |


## Benchmark Results

### SemEval2010 Task 8
1) Base architecture: BERT base uncased (12-layer, 768-hidden, 12-heads, 110M parameters)

Without MTB pre-training: F1 results when trained on 100 % training data:
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/task_test_f1_vs_epoch_0.png) 


2) Base architecture: ALBERT base uncased (12 repeating layers, 128 embedding, 768-hidden, 12-heads, 11M parameters)  

Without MTB pre-training: F1 results when trained on 100 % training data:
![](https://github.com/plkmo/BERT-Relation-Extraction/blob/master/results/CNN/task_test_f1_vs_epoch_1.png) 

## To add
- inference & results on benchmarks (SemEval2010 Task 8) with MTB pre-training
- felrel task

