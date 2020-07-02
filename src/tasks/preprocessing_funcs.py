#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee
"""
import os
import re
import random
import copy
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ..misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def process_text(text, mode='train'):
    sents, relations, comments, blanks = [], [], [], []
    for i in range(int(len(text)/4)):
        sent = text[4*i]
        relation = text[4*i + 1]
        comment = text[4*i + 2]
        blank = text[4*i + 3]
        
        # check entries
        if mode == 'train':
            assert int(re.match("^\d+", sent)[0]) == (i + 1)
        else:
            assert (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1)
        assert re.match("^Comment", comment)
        assert len(blank) == 1
        
        sent = re.findall("\"(.+)\"", sent)[0]
        sent = re.sub('<e1>', '[E1]', sent)
        sent = re.sub('</e1>', '[/E1]', sent)
        sent = re.sub('<e2>', '[E2]', sent)
        sent = re.sub('</e2>', '[/E2]', sent)
        sents.append(sent); relations.append(relation), comments.append(comment); blanks.append(blank)
    return sents, relations, comments, blanks

def preprocess_semeval2010_8(args):
    '''
    Data preprocessing for SemEval2010 task 8 dataset
    '''
    data_path = args.train_data #'./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text, 'train')
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    data_path = args.test_data #'./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text, 'test')
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    rm = Relations_Mapper(df_train['relations'])
    save_as_pickle('relations.pkl', rm)
    df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    save_as_pickle('df_train.pkl', df_train)
    save_as_pickle('df_test.pkl', df_test)
    logger.info("Finished and saved!")
    
    return df_train, df_test, rm

class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}
        
        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1
        
        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1,\
                 ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        
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
        
        return seqs_padded, labels_padded, labels2_padded, \
                x_lengths, y_lengths, y2_lengths

def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
                        [i for i, e in enumerate(x) if e == e2_id][0])
    except Exception as e:
        e1_e2_start = None
        print(e)
    return e1_e2_start

class semeval_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Tokenizing data...")
        self.df['input'] = self.df.progress_apply(lambda x: tokenizer.encode(x['sents']),\
                                                             axis=1)
        
        self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['input'],\
                                                       e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
        print("\nInvalid rows/total: %d/%d" % (df['e1_e2_start'].isnull().sum(), len(df)))
        self.df.dropna(axis=0, inplace=True)
    
    def __len__(self,):
        return len(self.df)
        
    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['input']),\
                torch.LongTensor(self.df.iloc[idx]['e1_e2_start']),\
                torch.LongTensor([self.df.iloc[idx]['relations_id']])

def preprocess_fewrel(args, do_lower_case=True):
    '''
    train: train_wiki.json
    test: val_wiki.json
    For 5 way 1 shot
    '''
    def process_data(data_dict):
        sents = []
        labels = []
        for relation, dataset in data_dict.items():
            for data in dataset:
                # first, get & verify the positions of entities
                h_pos, t_pos = data['h'][-1], data['t'][-1]
                
                if not len(h_pos) == len(t_pos) == 1: # remove one-to-many relation mappings
                    continue
                
                h_pos, t_pos = h_pos[0], t_pos[0]
                
                if len(h_pos) > 1:
                    running_list = [i for i in range(min(h_pos), max(h_pos) + 1)]
                    assert h_pos == running_list
                    h_pos = [h_pos[0], h_pos[-1] + 1]
                else:
                    h_pos.append(h_pos[0] + 1)
                
                if len(t_pos) > 1:
                    running_list = [i for i in range(min(t_pos), max(t_pos) + 1)]
                    assert t_pos == running_list
                    t_pos = [t_pos[0], t_pos[-1] + 1]
                else:
                    t_pos.append(t_pos[0] + 1)
                
                if (t_pos[0] <= h_pos[-1] <= t_pos[-1]) or (h_pos[0] <= t_pos[-1] <= h_pos[-1]): # remove entities not separated by at least one token 
                    continue
                
                if do_lower_case:
                    data['tokens'] = [token.lower() for token in data['tokens']]
                
                # add entity markers
                if h_pos[-1] < t_pos[0]:
                    tokens = data['tokens'][:h_pos[0]] + ['[E1]'] + data['tokens'][h_pos[0]:h_pos[1]] \
                            + ['[/E1]'] + data['tokens'][h_pos[1]:t_pos[0]] + ['[E2]'] + \
                            data['tokens'][t_pos[0]:t_pos[1]] + ['[/E2]'] + data['tokens'][t_pos[1]:]
                else:
                    tokens = data['tokens'][:t_pos[0]] + ['[E2]'] + data['tokens'][t_pos[0]:t_pos[1]] \
                            + ['[/E2]'] + data['tokens'][t_pos[1]:h_pos[0]] + ['[E1]'] + \
                            data['tokens'][h_pos[0]:h_pos[1]] + ['[/E1]'] + data['tokens'][h_pos[1]:]
                
                assert len(tokens) == (len(data['tokens']) + 4)
                sents.append(tokens)
                labels.append(relation)
        return sents, labels
        
    with open('./data/fewrel/train_wiki.json') as f:
        train_data = json.load(f)
        
    with  open('./data/fewrel/val_wiki.json') as f:
        test_data = json.load(f)
    
    train_sents, train_labels = process_data(train_data)
    test_sents, test_labels = process_data(test_data)
    
    df_train = pd.DataFrame(data={'sents': train_sents, 'labels': train_labels})
    df_test = pd.DataFrame(data={'sents': test_sents, 'labels': test_labels})
    
    rm = Relations_Mapper(list(df_train['labels'].unique()))
    save_as_pickle('relations.pkl', rm)
    df_train['labels'] = df_train.progress_apply(lambda x: rm.rel2idx[x['labels']], axis=1)
    
    return df_train, df_test

class fewrel_dataset(Dataset):
    def __init__(self, df, tokenizer, seq_pad_value, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.N = 5
        self.K = 1
        self.df = df
        
        logger.info("Tokenizing data...")
        self.df['sents'] = self.df.progress_apply(lambda x: tokenizer.encode(" ".join(x['sents'])),\
                                      axis=1)
        self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['sents'],\
                                                       e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
        print("\nInvalid rows/total: %d/%d" % (self.df['e1_e2_start'].isnull().sum(), len(self.df)))
        self.df.dropna(axis=0, inplace=True)
        
        self.relations = list(self.df['labels'].unique())
        
        self.seq_pad_value = seq_pad_value
            
    def __len__(self,):
        return len(self.df)
    
    def __getitem__(self, idx):
        target_relation = self.df['labels'].iloc[idx]
        relations_pool = copy.deepcopy(self.relations)
        relations_pool.remove(target_relation)
        sampled_relation = random.sample(relations_pool, self.N - 1)
        sampled_relation.append(target_relation)
        
        target_idx = self.N - 1
    
        e1_e2_start = []
        meta_train_input, meta_train_labels = [], []
        for sample_idx, r in enumerate(sampled_relation):
            filtered_samples = self.df[self.df['labels'] == r][['sents', 'e1_e2_start', 'labels']]
            sampled_idxs = random.sample(list(i for i in range(len(filtered_samples))), self.K)
            
            sampled_sents, sampled_e1_e2_starts = [], []
            for sampled_idx in sampled_idxs:
                sampled_sent = filtered_samples['sents'].iloc[sampled_idx]
                sampled_e1_e2_start = filtered_samples['e1_e2_start'].iloc[sampled_idx]
                
                assert filtered_samples['labels'].iloc[sampled_idx] == r
                
                sampled_sents.append(sampled_sent)
                sampled_e1_e2_starts.append(sampled_e1_e2_start)
            
            meta_train_input.append(torch.LongTensor(sampled_sents).squeeze())
            e1_e2_start.append(sampled_e1_e2_starts[0])
            
            meta_train_labels.append([sample_idx])
            
        meta_test_input = self.df['sents'].iloc[idx]
        meta_test_labels = [target_idx]
        
        e1_e2_start.append(get_e1e2_start(meta_test_input, e1_id=self.e1_id, e2_id=self.e2_id))
        e1_e2_start = torch.LongTensor(e1_e2_start).squeeze()
        
        meta_input = meta_train_input + [torch.LongTensor(meta_test_input)]
        meta_labels = meta_train_labels + [meta_test_labels]
        meta_input_padded = pad_sequence(meta_input, batch_first=True, padding_value=self.seq_pad_value).squeeze()
        return meta_input_padded, e1_e2_start, torch.LongTensor(meta_labels).squeeze()

def load_dataloaders(args):
    if args.model_no == 0:
        from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
        model = args.model_size#'bert-large-uncased' 'bert-base-uncased'
        lower_case = True
        model_name = 'BERT'
    elif args.model_no == 1:
        from ..model.ALBERT.tokenization_albert import AlbertTokenizer as Tokenizer
        model = args.model_size #'albert-base-v2'
        lower_case = True
        model_name = 'ALBERT'
    elif args.model_no == 2:
        from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
        model = 'bert-base-uncased'
        lower_case = False
        model_name = 'BioBERT'
        
    if os.path.isfile("./data/%s_tokenizer.pkl" % model_name):
        tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        logger.info("Loaded tokenizer from pre-trained blanks model")
    else:
        logger.info("Pre-trained blanks tokenizer not found, initializing new tokenizer...")
        if args.model_no == 2:
            tokenizer = Tokenizer(vocab_file='./additional_models/biobert_v1.1_pubmed/vocab.txt',
                                  do_lower_case=False)
        else:
            tokenizer = Tokenizer.from_pretrained(model, do_lower_case=False)
        tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])

        save_as_pickle("%s_tokenizer.pkl" % model_name, tokenizer)
        logger.info("Saved %s tokenizer at ./data/%s_tokenizer.pkl" %(model_name, model_name))
    
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1
    
    if args.task == 'semeval':
        relations_path = './data/relations.pkl'
        train_path = './data/df_train.pkl'
        test_path = './data/df_test.pkl'
        if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path):
            rm = load_pickle('relations.pkl')
            df_train = load_pickle('df_train.pkl')
            df_test = load_pickle('df_test.pkl')
            logger.info("Loaded preproccessed data.")
        else:
            df_train, df_test, rm = preprocess_semeval2010_8(args)
        
        train_set = semeval_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
        test_set = semeval_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
        train_length = len(train_set); test_length = len(test_set)
        PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id,\
                          label_pad_value=tokenizer.pad_token_id,\
                          label2_pad_value=-1)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=PS, pin_memory=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=PS, pin_memory=False)
    elif args.task == 'fewrel':
        df_train, df_test = preprocess_fewrel(args, do_lower_case=lower_case)
        train_loader = fewrel_dataset(df_train, tokenizer=tokenizer, seq_pad_value=tokenizer.pad_token_id,
                                      e1_id=e1_id, e2_id=e2_id)
        train_length = len(train_loader)
        test_loader, test_length = None, None
        
    return train_loader, test_loader, train_length, test_length