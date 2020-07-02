#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:16:26 2019

@author: weetee
"""
from src.preprocessing_funcs import load_dataloaders
from src.trainer import train_and_fit
import logging
from argparse import ArgumentParser

'''
This trains the BERT model on matching the blanks 
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrain_data", type=str, default="./data/cnn.txt", \
                        help="pre-training data .txt file path")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--freeze", type=int, default=0, help='''1: Freeze most layers until classifier layers\
                                                                \n0: Don\'t freeze \
                                                                (Probably best not to freeze if GPU memory is sufficient)''')
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=18, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT\n
                                                                            2 - BioBERT''')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
                                                                                                'bert-large-uncased',\
                                                                                    For ALBERT: 'albert-base-v2',\
                                                                                                'albert-large-v2',\
                                                                                    For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)")
    
    args = parser.parse_args()
    
    output = train_and_fit(args)
    
    '''
    # For testing additional models
    from src.model.BERT.modeling_bert import BertModel, BertConfig
    from src.model.BERT.tokenization_bert import BertTokenizer as Tokenizer
    config = BertConfig.from_pretrained('./additional_models/biobert_v1.1_pubmed/bert_config.json')
    model = BertModel.from_pretrained(pretrained_model_name_or_path='./additional_models/biobert_v1.1_pubmed.bin', 
                                      config=config,
                                      force_download=False, \
                                      model_size='bert-base-uncased',
                                      task='classification',\
                                      n_classes_=12)
    tokenizer = Tokenizer(vocab_file='./additional_models/biobert_v1.1_pubmed/vocab.txt',
                          do_lower_case=False)
    '''