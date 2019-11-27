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


def process_sent(sent):
    if sent not in [" ", "\n", ""]:
        sent = sent.strip("\n")            
        sent = re.sub('<[A-Z]+/*>', '', sent) # remove special tokens eg. <FIL/>, <S>
        sent = re.sub(r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#]", " ", sent)
        sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
        sent = re.sub("^ +", "", sent) # remove space in front
        sent = re.sub(r"([\.\?,!]){2,}", r"\1", sent) # remove multiple puncs
        sent = re.sub(r" +([\.\?,!])", r"\1", sent) # remove extra spaces in front of punc
        #sent = re.sub(r"([A-Z]{2,})", lambda x: x.group(1).capitalize(), sent) # Replace all CAPS with capitalize
        return sent
    return

def process_textlines(text):
    text = [process_sent(sent) for sent in text]
    text = " ".join([t for t in text if t is not None])
    return text    

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
    entities_of_interest = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", \
                            "WORK_OF_ART", "LAW", "LANGUAGE"]
    length_doc = len(sents_doc)
    D = []
    for i in tqdm(range(len(ents))):
        e1 = ents[i]
        if e1.label_ not in entities_of_interest:
            continue
        if re.search("[\d+]", e1.text):
            continue
        
        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if e2.label_ not in entities_of_interest:
                continue
            if re.search("[\d+]", e2.text):
                continue
            
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
                print(e1.text,",", e2.text)
    return D

class pretrain_dataset(Dataset):
    def __init__(self, D):
        self.alpha = 0.7
        self.mask_probability = 0.15
        self.D = D
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        
    def put_blanks(self, D):
        blank_e1 = np.random.uniform()
        blank_e2 = np.random.uniform()
        if blank_e1 >= self.alpha:
            r, e1, e2 = D
            D = (r, "[BLANK]", e2)
        
        if blank_e2 >= self.alpha:
            r, e1, e2 = D
            D = (r, e1, "[BLANK]")
        return D
        
    def tokenize(self, D):
        (x, s1, s2), e1, e2 = D
        
        ### Include random masks for MLM training
        forbidden_idxs = [i for i in range(s1[0], s1[1])] + [i for i in range(s2[0], s2[1])]
        pool_idxs = [i for i in range(len(x)) if i not in forbidden_idxs]
        masked_idxs = np.random.choice(pool_idxs,\
                                        size=round(self.mask_probability*len(pool_idxs)),\
                                        replace=False)
        x = [token if (idx not in masked_idxs) else self.tokenizer.mask_token \
             for idx, token in enumerate(x)]
        masked_for_pred = [token for idx, token in enumerate(x) if (idx in masked_idxs)]
        
        ### replace x spans with '[BLANK]' if e is '[BLANK]'
        if (e1 == '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]' ,'[BLANK]', '[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]
        
        elif (e1 == '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]' ,'[BLANK]', '[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]
        
        elif (e1 != '[BLANK]') and (e2 == '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]'] + x[s1[0]:s1[1]] + ['[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]', '[BLANK]', '[/E2]'] + x[s2[1]:] + [self.sep_token]
        
        elif (e1 != '[BLANK]') and (e2 != '[BLANK]'):
            x = [self.cls_token] + x[:s1[0]] + ['[E1]'] + x[s1[0]:s1[1]] + ['[/E1]'] + \
                x[s1[1]:s2[0]] + ['[E2]'] + x[s2[0]:s2[1]] + ['[/E2]'] + x[s2[1]:] + [self.sep_token]
        
        e1_e2_start = ([i for i, e in enumerate(x) if e == '[E1]'][0],\
                        [i for i, e in enumerate(x) if e == '[E2]'][0])
         
        x = self.tokenizer.convert_tokens_to_ids(x)
        masked_for_pred = self.tokenizer.convert_tokens_to_ids(masked_for_pred)
        return x, masked_for_pred, e1_e2_start
    
    def __len__(self):
        return len(self.D)
    
    def __getitem__(self, idx):
        x, masked_for_pred, e1_e2_start = self.tokenize(self.put_blanks(self.D[idx]))
        return x, masked_for_pred, e1_e2_start

def load_dataloaders(args, max_length=50000):
    logger.info("Loading pre-training data...")
    with open(args.pretrain_data, "r", encoding="utf8") as f:
        text = f.readlines()
    text = text[:10000] # restrict size for testing
    
    text = process_textlines(text)
    
    logger.info("Length of text (characters): %d" % len(text))
    num_chunks = math.ceil(len(text)/max_length)
    logger.info("Splitting into %d max length chunks of size %d" % (num_chunks, max_length))
    text_chunks = [text[i*max_length:(i*max_length + max_length)] for i in range(num_chunks)]
    D = []
    for text_chunk in tqdm(text_chunks):
        D.extend(create_pretraining_corpus(text_chunk, window_size=40))
    return D
