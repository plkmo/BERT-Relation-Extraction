#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:44:17 2019

@author: weetee
"""

import pickle
import os
import pandas as pd
import torch
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class infer_from_trained(object):
    def __init__(self, args=None):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()
        
        logger.info("Loading tokenizer and model...")
        from .model.modeling_bert import BertModel
        from .train_funcs import load_state
        
        self.net = BertModel.from_pretrained('bert-base-uncased', force_download=False)
        self.tokenizer = load_pickle("BERT_tokenizer.pkl")
        self.net.resize_token_embeddings(len(self.tokenizer))
        if self.cuda:
            self.net.cuda()
        start_epoch, best_pred, amp_checkpoint = load_state(self.net, None, None, self.args, load_best=False)
        logger.info("Done!")
        
    def infer_sentence(self, sentence):
        self.net.eval()
        tokenized = self.tokenizerconvert_tokens_to_ids(sentence)
        return