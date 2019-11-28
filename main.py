#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:16:26 2019

@author: weetee
"""
from src.preprocessing_funcs import load_dataloaders
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrain_data", type=str, default="./data/TheSingaporeStory_MemoirsOfLeeKuanYew.txt", \
                        help="pre-training data .txt file path")
    parser.add_argument("--batch_size", type=int, default=3, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--num_epochs", type=int, default=40, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    
    args = parser.parse_args()
    
    D, train_loader, train_set = load_dataloaders(args)
    
    for data in train_loader:
        break
    
    d0 = train_set.D[0]
    d0_ = train_set.__getitem__(0)
    print(train_set.tokenizer.decode(d0_[0].numpy()))
    print()
    print(train_set.tokenizer.decode(data[0][0].numpy()))
    print(train_set.tokenizer.decode(data[1][0].numpy()))
    print(data[2][0].numpy())
    print(train_set.tokenizer.decode(data[3][0].numpy()))
    print(train_set.tokenizer.decode(data[4][0].numpy()))