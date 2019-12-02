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
    parser.add_argument("--pretrain_data", type=str, default="./data/TheSingaporeStory_MemoirsOfLeeKuanYew.txt", \
                        help="pre-training data .txt file path")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=40, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    
    args = parser.parse_args()
    
    output = train_and_fit(args)