#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee
"""

from src.tasks.preprocessing_funcs import load_dataloaders
from src.tasks.trainer import train_and_fit
import logging
from argparse import ArgumentParser

'''
This fine-tunes the BERT model on SemEval task 
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT', \
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', \
                        help="test data .txt file path")
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=40, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    
    args = parser.parse_args()
    
    train_loader, test_loader, train_length, test_length = load_dataloaders(args)
    for train in train_loader:
        break
    for test in test_loader:
        break
    classification_logits, net, tokenizer = train_and_fit(args)