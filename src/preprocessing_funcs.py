#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee
"""
import re
import spacy
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .model.tokenization_bert import BertTokenizer
from .misc import save_as_pickle
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def create_pretraining_corpus(raw_text, window_size=40):
    '''
    Input: Chunk of raw text
    Output: modified corpus of triplets (relation statement, entity1, entity2)
    '''
    logger.info("Loading Spacy NLP...")
    nlp = spacy.load("en_core_web_lg")
    
    logger.info("Processing sentences...")
    sents_doc = nlp(raw_text)
    ents = sents_doc.ents
    
    logger.info("Processing relation statements...")
    length_doc = len(sents_doc)
    D = []
    for i in tqdm(range(len(ents))):
        e1 = ents[i]
        for j in range(1, len(ents) - i):
            e2 = ents[i + j]; print(e1.text,",", e2.text)
            
            if (e1.end - e2.start) <= window_size: # check if next nearest entity within window_size
                # Find start of sentence
                punc_token = False
                start = e1.start - 1
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0
                
                # Find end of sentence
                punc_token = False
                start = e2.end
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                    right_r = start
                else:
                    right_r = length_doc
                x = [token.text for token in sents_doc[left_r:right_r]]
                r = (x, (e1.start, e1.end), (e2.start, e2.end))
                D.append((r, e1.text, e2.text))
        
    return D

class pretrain_dataset(Dataset):
    def __init__(self, D, alpha=0.7,):
        self.alpha = alpha
        self.D = D
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
        
    def put_blanks(self, D):
        logger.info("Introducing blanks with probability %.2f" % (1 - self.alpha))
        blank_idxs_e1 = np.random.choice([i for i in range(len(D))], \
                                                        size=round((1 - self.alpha)*len(D)),\
                                                        replace=False)
        blank_idxs_e2 = np.random.choice([i for i in range(len(D))], \
                                                        size=round((1 - self.alpha)*len(D)),\
                                                        replace=False)
        logger.info("Entity 1:")
        for idx in blank_idxs_e1:
            r, e1, e2 = self.D[idx]
            self.D[idx] = (r, "[BLANK]", e2)
        
        logger.info("Entity 2:")
        for idx in blank_idxs_e2:
            r, e1, e2 = self.D[idx]
            self.D[idx] = (r, e1, "[BLANK]")
        logger.info("Done!")
        
    def tokenize(self, D):
        return
    
    def __len__(self):
        return len(self.D)
    
    def __getitem__(self, idx):
        return

def load_dataloaders(args, max_length=50000):
    logger.info("Loading pre-training data...")
    with open(args.pretrain_data, "r", encoding="utf8") as f:
        text = f.read()
    
    logger.info("Length of text (characters): %d" % len(text))
    logger.info("Splitting into max length chunks of size %d" % max_length)
    text_chunks = [text[i*max_length:(i*max_length + max_length)] for i in range(math.ceil(len(text)//max_length))]
    D = []
    for text_chunk in tqdm(text_chunks):
        D.extend(create_pretraining_corpus(text_chunk, window_size=40))
    return D
