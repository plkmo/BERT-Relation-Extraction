#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee
"""
import os
import re
import spacy
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .model.tokenization_bert import BertTokenizer
from .misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
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
    text = re.sub(' {2,}', ' ', text) # remove extra spaces > 1
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
        e1start = e1.start; e1end = e1.end
        if e1.label_ not in entities_of_interest:
            continue
        if re.search("[\d+]", e1.text): # entities should not contain numbers
            continue
        
        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            e2start = e2.start; e2end = e2.end
            if e2.label_ not in entities_of_interest:
                continue
            if re.search("[\d+]", e2.text): # entities should not contain numbers
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                continue
            
            if (1 <= (e2start - e1end) <= window_size): # check if next nearest entity within window_size
                # Find start of sentence
                punc_token = False
                start = e1start - 1
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
                start = e2end
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc
                x = [token.text for token in sents_doc[left_r:right_r]]
                
                ### empty strings check ###
                for token in x:
                    assert len(token) > 0
                assert len(e1.text) > 0
                assert len(e2.text) > 0
                assert e1start != e1end
                assert e2start != e2end
                assert (e2start - e1end) > 0
                
                r = (x, (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r))
                D.append((r, e1.text, e2.text))
                #print(e1.text,",", e2.text)
    return D

class pretrain_dataset(Dataset):
    def __init__(self, D, batch_size=None):
        self.internal_batching = True
        self.batch_size = batch_size # batch_size cannot be None if internal_batching == True
        self.alpha = 0.7
        self.mask_probability = 0.15
        
        self.df = pd.DataFrame(D, columns=['r','e1','e2'])
        self.e1s = list(self.df['e1'].unique())
        self.e2s = list(self.df['e2'].unique())
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.E1_token_id = self.tokenizer.encode("[E1]")[1:-1][0]
        self.E1s_token_id = self.tokenizer.encode("[/E1]")[1:-1][0]
        self.E2_token_id = self.tokenizer.encode("[E2]")[1:-1][0]
        self.E2s_token_id = self.tokenizer.encode("[/E2]")[1:-1][0]
        self.PS = Pad_Sequence(seq_pad_value=self.tokenizer.pad_token_id,\
                               label_pad_value=self.tokenizer.pad_token_id,\
                               label2_pad_value=-1,\
                               label3_pad_value=-1,\
                               label4_pad_value=-1)
        
        save_as_pickle("BERT_tokenizer.pkl", self.tokenizer)
        logger.info("Saved BERT tokenizer at ./data/BERT_tokenizer.pkl")
        
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
        x = [w.lower() for w in x if x != '[BLANK]'] # we are using uncased model
        
        ### Include random masks for MLM training
        forbidden_idxs = [i for i in range(s1[0], s1[1])] + [i for i in range(s2[0], s2[1])]
        pool_idxs = [i for i in range(len(x)) if i not in forbidden_idxs]
        masked_idxs = np.random.choice(pool_idxs,\
                                        size=round(self.mask_probability*len(pool_idxs)),\
                                        replace=False)
        masked_for_pred = [token for idx, token in enumerate(x) if (idx in masked_idxs)]
        masked_for_pred = [w.lower() for w in masked_for_pred] # we are using uncased model
        x = [token if (idx not in masked_idxs) else self.tokenizer.mask_token \
             for idx, token in enumerate(x)]

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
        e1 = [e for idx, e in enumerate(x) if idx in [i for i in\
              range(x.index(self.E1_token_id) + 1, x.index(self.E1s_token_id))]]
        e2 = [e for idx, e in enumerate(x) if idx in [i for i in\
              range(x.index(self.E2_token_id) + 1, x.index(self.E2s_token_id))]]
        return x, masked_for_pred, e1_e2_start, e1, e2
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        ### implements standard batching
        if not self.internal_batching:
            r, e1, e2 = self.df.iloc[idx]
            x, masked_for_pred, e1_e2_start, e1, e2 = self.tokenize(self.put_blanks((r, e1, e2)))
            x = torch.tensor(x)
            masked_for_pred = torch.tensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            e1, e2 = torch.tensor(e1), torch.tensor(e2)
            return x, masked_for_pred, e1_e2_start, e1, e2
        
        ### implements noise contrastive estimation
        else:
            r, e1, e2 = self.df.iloc[idx] # positive sample
            
            ### get negative samples
            '''
            choose from option: 
            1) sampling uniformly from all negatives
            2) sampling uniformly from negatives that share e1 or e2
            '''
            if np.random.uniform() > 0.5:   
                pool = self.df[((self.df['e1'] != e1) | (self.df['e2'] != e2))].index
                neg_idxs = np.random.choice(pool, \
                                            size=min((self.batch_size - 1), len(pool)), replace=False)
                Q = 1/len(pool)
            
            else:
                if np.random.uniform() > 0.5: # share e1 but not e2
                    pool = self.df[((self.df['e1'] == e1) & (self.df['e2'] != e2))].index
                    neg_idxs = np.random.choice(pool, \
                                                size=min((self.batch_size - 1), len(pool)), replace=False)

                else: # share e2 but not e1
                    pool = self.df[((self.df['e1'] != e1) & (self.df['e2'] == e2))].index
                    neg_idxs = np.random.choice(pool, \
                                                size=min((self.batch_size - 1), len(pool)), replace=False)
                    
                if len(neg_idxs) == 0: # if empty, sample from all negatives
                    pool = self.df[((self.df['e1'] != e1) | (self.df['e2'] != e2))].index
                    neg_idxs = np.random.choice(pool, \
                                            size=min((self.batch_size - 1), len(pool)), replace=False)
                Q = 1/len(pool)
            
            ## process positive sample
            x, masked_for_pred, e1_e2_start, e1, e2 = self.tokenize(self.put_blanks((r, e1, e2)))
            x = torch.tensor(x)
            masked_for_pred = torch.tensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            e1, e2 = torch.tensor(e1), torch.tensor(e2)
            batch = [(x, masked_for_pred, e1_e2_start, torch.tensor([0]).long(), torch.tensor([1]))]
            
            ## process negative samples
            negs_df = self.df.loc[neg_idxs]
            for idx, row in negs_df.iterrows():
                r, e1, e2 = row[0], row[1], row[2]
                x, masked_for_pred, e1_e2_start, e1, e2 = self.tokenize(self.put_blanks((r, e1, e2)))
                x = torch.tensor(x)
                masked_for_pred = torch.tensor(masked_for_pred)
                e1_e2_start = torch.tensor(e1_e2_start)
                e1, e2 = torch.tensor(e1), torch.tensor(e2)
                batch.append((x, masked_for_pred, e1_e2_start, torch.tensor([Q]), torch.tensor([0])))
            batch = self.PS(batch)
            return batch
    
class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=1, label2_pad_value=-1,\
                 label3_pad_value=-1, label4_pad_value=-1):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        self.label3_pad_value = label3_pad_value
        self.label4_pad_value = label4_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])
        
        labels3 = list(map(lambda x: x[3], sorted_batch))
        labels3_padded = pad_sequence(labels3, batch_first=True, padding_value=self.label3_pad_value)
        y3_lengths = torch.LongTensor([len(x) for x in labels3])
        
        labels4 = list(map(lambda x: x[4], sorted_batch))
        labels4_padded = pad_sequence(labels4, batch_first=True, padding_value=self.label4_pad_value)
        y4_lengths = torch.LongTensor([len(x) for x in labels4])
        return seqs_padded, labels_padded, labels2_padded, labels3_padded, labels4_padded,\
                x_lengths, y_lengths, y2_lengths, y3_lengths, y4_lengths

def load_dataloaders(args, max_length=50000):
    
    if not os.path.isfile("./data/D.pkl"):
        logger.info("Loading pre-training data...")
        with open(args.pretrain_data, "r", encoding="utf8") as f:
            text = f.readlines()
        
        #text = text[:1500] # restrict size for testing
        text = process_textlines(text)
        
        logger.info("Length of text (characters): %d" % len(text))
        num_chunks = math.ceil(len(text)/max_length)
        logger.info("Splitting into %d max length chunks of size %d" % (num_chunks, max_length))
        text_chunks = (text[i*max_length:(i*max_length + max_length)] for i in range(num_chunks))
        D = []
        for text_chunk in tqdm(text_chunks, total=num_chunks):
            D.extend(create_pretraining_corpus(text_chunk, window_size=40))
        save_as_pickle("D.pkl", D)
        logger.info("Saved pre-training corpus to %s" % "./data/D.pkl")
    else:
        logger.info("Loaded pre-training data from saved file")
        D = load_pickle("D.pkl")
        
    train_set = pretrain_dataset(D, batch_size=args.batch_size)
    train_length = len(train_set)
    '''
    # if using fixed batching
    PS = Pad_Sequence(seq_pad_value=train_set.tokenizer.pad_token_id,\
                      label_pad_value=train_set.tokenizer.pad_token_id,\
                      label2_pad_value=-1,\
                      label3_pad_value=-1,\
                      label4_pad_value=-1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
                              num_workers=0, collate_fn=PS, pin_memory=False)
    '''
    return train_set
