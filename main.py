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
    args = parser.parse_args()
    
    D = load_dataloaders(args)