import itertools
import logging
import math
import os
import re

import joblib
import numpy as np
import pandas as pd
import spacy
import torch
from ml_utils.normalizer import Normalizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AlbertTokenizer

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from src.misc import get_subject_objects

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL,
)
logger = logging.getLogger(__file__)


class DataLoader:
    def __init__(self, config: dict):
        """
        DataLoader for MTB data.

        Args:
            config: configuration parameters
        """
        self.config = config

        tokenizer_path = "data/ALBERT_tokenizer.pkl"
        if os.path.isfile(tokenizer_path):
            with open(tokenizer_path, "rb") as pkl_file:
                self.tokenizer = joblib.load(pkl_file)
            logger.info("Loaded tokenizer from saved path.")
        else:
            self.tokenizer = AlbertTokenizer.from_pretrained(
                "albert-large-v2", do_lower_case=False
            )
            self.tokenizer.add_tokens(
                ["[E1]", "[/E1]", "[E2]", "[/E2]", "[BLANK]"]
            )
            with open(tokenizer_path, "wb") as output:
                joblib.dump(self.tokenizer, output)

            logger.info("Saved ALBERT tokenizer at {0}".format(tokenizer_path))
        e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        if e1_id == e2_id:
            raise ValueError("E1 token equals E2 token")

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.E1_token_id = self.tokenizer.encode("[E1]")[1:-1][0]
        self.E1s_token_id = self.tokenizer.encode("[/E1]")[1:-1][0]
        self.E2_token_id = self.tokenizer.encode("[E2]")[1:-1][0]
        self.E2s_token_id = self.tokenizer.encode("[/E2]")[1:-1][0]
        self.pad_token_id = self.tokenizer.pad_token_id

        self.normalizer = Normalizer("en", config.get("normalization", []))
        self.data = self.load_dataset()
        self.train_generator = PretrainDataset(
            self.data, batch_size=self.config.get("batch_size")
        )

        self.batch_size = config.get("batch_size")
        self.alpha = 0.7
        self.mask_probability = 0.15

    def load_dataset(self):
        """
        Load the data defined in the configuration parameters.
        """
        data_path = self.config.get("data")
        data_file = os.path.basename(data_path)
        data_file_name = os.path.splitext(data_file)[0]
        preprocessed_file = os.path.join("data", data_file_name + ".pkl")

        max_length = self.config.get("max_length", 50000)

        if os.path.isfile(preprocessed_file):
            logger.info("Loaded pre-training data from saved file")
            with open(preprocessed_file, "rb") as pkl_file:
                data = joblib.load(pkl_file)

        else:
            logger.info("Loading pre-training data...")
            with open(data_path, "r", encoding="utf8") as f:
                text = f.readlines()

            text = self._process_textlines(text)

            logger.info("Length of text (characters): {0}".format(len(text)))
            num_chunks = math.ceil(len(text) / max_length)
            logger.info(
                "Splitting into {0} chunks of size {1}".format(
                    num_chunks, max_length
                )
            )
            text_chunks = (
                text[i * max_length : (i * max_length + max_length)]
                for i in range(num_chunks)
            )

            dataset = []
            logger.info("Loading Spacy NLP")
            nlp = spacy.load("en_core_web_lg")

            for text_chunk in tqdm(text_chunks, total=num_chunks):
                dataset.extend(
                    self.create_pretraining_dataset(
                        text_chunk, nlp, window_size=40
                    )
                )

            logger.info(
                "Number of relation statements in corpus: {0}".format(
                    len(dataset)
                )
            )
            dataset = pd.DataFrame(dataset)
            dataset.columns = ["r", "e1", "e2"]

            data = self.preprocess(dataset)
            with open(preprocessed_file, "wb") as output:
                joblib.dump(data, output)
            logger.info(
                "Saved pre-training corpus to {0}".format(preprocessed_file)
            )
        return data

    def preprocess(self, data: pd.DataFrame):
        """
        Preprocess the dataset.

        Normalizes the dataset, tokenizes it and add special tokens

        Args:
            data: dataset to preprocess
        """
        logger.info("Normalizing relations")
        normalized_relations = []
        for _idx, row in data.iterrows():
            relation = self._add_special_tokens(row)
            normalized_relations.append(relation)

        logger.info("Tokenizing relations")
        tokenized_relations = [
            torch.IntTensor(self.tokenizer.convert_tokens_to_ids(n))
            for n in normalized_relations
        ]
        tokenized_relations = pad_sequence(
            tokenized_relations,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        e_span1 = [(r[1][0] + 2, r[1][1] + 2) for r in data["r"]]
        e_span2 = [(r[2][0] + 4, r[2][1] + 4) for r in data["r"]]
        r = [
            (tr.numpy().tolist(), e1, e2)
            for tr, e1, e2 in zip(tokenized_relations, e_span1, e_span2)
        ]
        data["r"] = r
        pools = self.transform_data(data)
        preprocessed_data = {
            "entities_pools": pools,
            "tokenized_relations": data,
        }
        return preprocessed_data

    def _add_special_tokens(self, row):
        r = row.get("r")[0]
        e_span1 = row.get("r")[1]
        e_span2 = row.get("r")[2]
        relation = [self.tokenizer.cls_token]
        for w_idx, w in enumerate(r):
            if w_idx == e_span1[0]:
                relation.append("[E1]")
            if w_idx == e_span2[0]:
                relation.append("[E2]")
            relation.append(self.normalizer.normalize(w))
            if w_idx == e_span1[1]:
                relation.append("[/E1]")
            if w_idx == e_span2[1]:
                relation.append("[/E2]")
        relation.append(self.tokenizer.sep_token)
        return relation

    @classmethod
    def transform_data(cls, df: pd.DataFrame):
        """
        Prepare data for the QQModel.

        Data format:     Question pairs1.     Question pairs2. Negative
        question pool per question.

        Args:
            df: Dataframe to use to generate QQ pairs.
        """
        df["relation_id"] = np.arange(0, len(df))
        logger.info("Generating class pools")
        return DataLoader.generate_entities_pools(df)

    @classmethod
    def generate_entities_pools(cls, data: pd.DataFrame):
        """
        Generate class pools.

        Args:
            data: pandas dataframe containing the relation, entity 1 & 2 and the relation id

        Returns:
            Index of question.
            Index of paired question.
            Common answer id.
        """
        groups = data.groupby(["e1", "e2"])
        pool = []
        for idx, df in groups:
            e1, e2 = idx
            e1_negatives = data[((data["e1"] == e1) & (data["e2"] != e2))][
                "relation_id"
            ]
            e2_negatives = data[((data["e1"] != e1) & (data["e2"] == e2))][
                "relation_id"
            ]
            entities_pool = (
                df["relation_id"].values.tolist(),
                e1_negatives.values.tolist(),
                e2_negatives.values.tolist(),
            )
            pool.append(entities_pool)
        logger.info("Found {0} different pools".format(len(pool)))
        return pool

    def _process_textlines(self, text):
        text = [self._clean_sent(sent) for sent in text]
        text = " ".join([t for t in text if t is not None])
        text = re.sub(" {2,}", " ", text)
        return text

    @classmethod
    def _clean_sent(cls, sent):
        if sent not in {" ", "\n", ""}:
            sent = sent.strip("\n")
            sent = re.sub(
                "<[A-Z]+/*>", "", sent
            )  # remove special tokens eg. <FIL/>, <S>
            sent = re.sub(
                r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#]", " ", sent
            )
            sent = " ".join(sent.split())  # remove whitespaces > 1
            sent = sent.strip()
            sent = re.sub(
                r"([\.\?,!]){2,}", r"\1", sent
            )  # remove multiple puncs
            sent = re.sub(
                r"([A-Z]{2,})", lambda x: x.group(1).capitalize(), sent
            )  # Replace all CAPS with capitalize
            return sent

    def create_pretraining_dataset(
        self, raw_text: str, nlp, window_size: int = 40
    ):
        """
        Input: Chunk of raw text
        Output: modified corpus of triplets (relation statement, entity1, entity2)

        Args:
            raw_text: Raw text input
            nlp: spacy NLP model
            window_size: Maximum windows size between to entities
        """
        logger.info("Processing sentences...")
        sents_doc = nlp(raw_text)
        ents = sents_doc.ents  # get entities

        logger.info("Processing relation statements by entities...")
        entities_of_interest = self.config.get("entities_of_interest")
        length_doc = len(sents_doc)
        data = []
        ents_list = []
        for e1, e2 in tqdm(itertools.product(ents, ents)):
            if e1 == e2:
                continue
            e1start = e1.start
            e1end = e1.end
            e2start = e2.start
            e2end = e2.end
            e1_has_numbers = re.search("[\d+]", e1.text)
            e2_has_numbers = re.search("[\d+]", e2.text)
            if (e1.label_ not in entities_of_interest) or e1_has_numbers:
                continue
            if (e2.label_ not in entities_of_interest) or e2_has_numbers:
                continue
            if e1.text.lower() == e2.text.lower():  # make sure e1 != e2
                continue
            # check if next nearest entity within window_size
            if 1 <= (e2start - e1end) <= window_size:
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
                # sentence should not be longer than window_size
                if (right_r - left_r) > window_size:
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]

                empty_token = all(not token for token in x)
                emtpy_e1 = not e1.text
                emtpy_e2 = not e2.text
                emtpy_span = (e2start - e1end) < 1
                if emtpy_e1 or emtpy_e2 or emtpy_span or empty_token:
                    raise ValueError("Relation has wrong format")

                r = (
                    x,
                    (e1start - left_r, e1end - left_r),
                    (e2start - left_r, e2end - left_r),
                )
                data.append((r, e1.text, e2.text))
                ents_list.append((e1.text, e2.text))

        logger.info(
            "Processing relation statements by dependency tree parsing..."
        )
        doc_sents = list(sents_doc.sents)
        for sent_ in tqdm(doc_sents, total=len(doc_sents)):
            if len(sent_) > (window_size + 1):
                continue

            left_r = sent_[0].i
            pairs = get_subject_objects(sent_)

            if len(pairs) > 0:
                for pair in pairs:
                    e1, e2 = pair[0], pair[1]

                    if (len(e1) > 3) or (
                        len(e2) > 3
                    ):  # don't want entities that are too long
                        continue

                    e1text, e2text = (
                        " ".join(w.text for w in e1)
                        if isinstance(e1, list)
                        else e1.text,
                        " ".join(w.text for w in e2)
                        if isinstance(e2, list)
                        else e2.text,
                    )
                    e1start, e1end = (
                        e1[0].i if isinstance(e1, list) else e1.i,
                        e1[-1].i + 1 if isinstance(e1, list) else e1.i + 1,
                    )
                    e2start, e2end = (
                        e2[0].i if isinstance(e2, list) else e2.i,
                        e2[-1].i + 1 if isinstance(e2, list) else e2.i + 1,
                    )
                    if (e1end < e2start) and (
                        (e1text, e2text) not in ents_list
                    ):
                        assert e1start != e1end
                        assert e2start != e2end
                        assert (e2start - e1end) > 0
                        r = (
                            [w.text for w in sent_],
                            (e1start - left_r, e1end - left_r),
                            (e2start - left_r, e2end - left_r),
                        )
                        data.append((r, e1text, e2text))
                        ents_list.append((e1text, e2text))
        return data


class PretrainDataset(Dataset):
    def __init__(self, dataset, batch_size=None):
        self.batch_size = batch_size
        self.alpha = 0.7
        self.mask_probability = 0.15

        self.df = pd.DataFrame(dataset, columns=["r", "e1", "e2"])

        tokenizer_path = "data/ALBERT_tokenizer.pkl"
        if os.path.isfile(tokenizer_path):
            with open(tokenizer_path, "rb") as pkl_file:
                self.tokenizer = joblib.load(pkl_file)
            logger.info("Loaded tokenizer from saved path.")
        else:
            self.tokenizer = AlbertTokenizer.from_pretrained(
                "albert-large-v2", do_lower_case=False
            )
            self.tokenizer.add_tokens(
                ["[E1]", "[/E1]", "[E2]", "[/E2]", "[BLANK]"]
            )
            with open(tokenizer_path, "wb") as output:
                joblib.dump(self.tokenizer, output)

            logger.info("Saved ALBERT tokenizer at {0}".format(tokenizer_path))
        e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        if not e1_id != e2_id != 1:
            raise ValueError("E1 token == E2 token == 1")

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.E1_token_id = self.tokenizer.encode("[E1]")[1:-1][0]
        self.E1s_token_id = self.tokenizer.encode("[/E1]")[1:-1][0]
        self.E2_token_id = self.tokenizer.encode("[E2]")[1:-1][0]
        self.E2s_token_id = self.tokenizer.encode("[/E2]")[1:-1][0]
        self.PS = Pad_Sequence(
            seq_pad_value=self.tokenizer.pad_token_id,
            label_pad_value=self.tokenizer.pad_token_id,
            label2_pad_value=-1,
            label3_pad_value=-1,
            label4_pad_value=-1,
        )

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
        x = [
            w.lower() for w in x if x != "[BLANK]"
        ]  # we are using uncased model

        ### Include random masks for MLM training
        forbidden_idxs = [i for i in range(s1[0], s1[1])] + [
            i for i in range(s2[0], s2[1])
        ]
        pool_idxs = [i for i in range(len(x)) if i not in forbidden_idxs]
        masked_idxs = np.random.choice(
            pool_idxs,
            size=round(self.mask_probability * len(pool_idxs)),
            replace=False,
        )
        masked_for_pred = [
            token.lower()
            for idx, token in enumerate(x)
            if (idx in masked_idxs)
        ]
        # masked_for_pred = [w.lower() for w in masked_for_pred] # we are using uncased model
        x = [
            token if (idx not in masked_idxs) else self.tokenizer.mask_token
            for idx, token in enumerate(x)
        ]

        ### replace x spans with '[BLANK]' if e is '[BLANK]'
        if (e1 == "[BLANK]") and (e2 != "[BLANK]"):
            x = (
                [self.cls_token]
                + x[: s1[0]]
                + ["[E1]", "[BLANK]", "[/E1]"]
                + x[s1[1] : s2[0]]
                + ["[E2]"]
                + x[s2[0] : s2[1]]
                + ["[/E2]"]
                + x[s2[1] :]
                + [self.sep_token]
            )

        elif (e1 == "[BLANK]") and (e2 == "[BLANK]"):
            x = (
                [self.cls_token]
                + x[: s1[0]]
                + ["[E1]", "[BLANK]", "[/E1]"]
                + x[s1[1] : s2[0]]
                + ["[E2]", "[BLANK]", "[/E2]"]
                + x[s2[1] :]
                + [self.sep_token]
            )

        elif (e1 != "[BLANK]") and (e2 == "[BLANK]"):
            x = (
                [self.cls_token]
                + x[: s1[0]]
                + ["[E1]"]
                + x[s1[0] : s1[1]]
                + ["[/E1]"]
                + x[s1[1] : s2[0]]
                + ["[E2]", "[BLANK]", "[/E2]"]
                + x[s2[1] :]
                + [self.sep_token]
            )

        elif (e1 != "[BLANK]") and (e2 != "[BLANK]"):
            x = (
                [self.cls_token]
                + x[: s1[0]]
                + ["[E1]"]
                + x[s1[0] : s1[1]]
                + ["[/E1]"]
                + x[s1[1] : s2[0]]
                + ["[E2]"]
                + x[s2[0] : s2[1]]
                + ["[/E2]"]
                + x[s2[1] :]
                + [self.sep_token]
            )

        e1_e2_start = (
            [i for i, e in enumerate(x) if e == "[E1]"][0],
            [i for i, e in enumerate(x) if e == "[E2]"][0],
        )

        x = self.tokenizer.convert_tokens_to_ids(x)
        masked_for_pred = self.tokenizer.convert_tokens_to_ids(masked_for_pred)
        return x, masked_for_pred, e1_e2_start

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r, e1, e2 = self.df.iloc[idx]  # positive sample
        pool = self.df[((self.df["e1"] == e1) & (self.df["e2"] == e2))].index
        pool = pool.append(
            self.df[((self.df["e1"] == e2) & (self.df["e2"] == e1))].index
        )
        pos_idxs = np.random.choice(
            pool,
            size=min(int(self.batch_size // 2), len(pool)),
            replace=False,
        )
        ### get negative samples
        """
        choose from option: 
        1) sampling uniformly from all negatives
        2) sampling uniformly from negatives that share e1 or e2
        """
        if np.random.uniform() > 0.5:
            pool = self.df[
                ((self.df["e1"] != e1) | (self.df["e2"] != e2))
            ].index
            neg_idxs = np.random.choice(
                pool,
                size=min(int(self.batch_size // 2), len(pool)),
                replace=False,
            )
            Q = 1 / len(pool)

        else:
            if np.random.uniform() > 0.5:  # share e1 but not e2
                pool = self.df[
                    ((self.df["e1"] == e1) & (self.df["e2"] != e2))
                ].index
                if len(pool) > 0:
                    neg_idxs = np.random.choice(
                        pool,
                        size=min(int(self.batch_size // 2), len(pool)),
                        replace=False,
                    )
                else:
                    neg_idxs = []

            else:  # share e2 but not e1
                pool = self.df[
                    ((self.df["e1"] != e1) & (self.df["e2"] == e2))
                ].index
                if len(pool) > 0:
                    neg_idxs = np.random.choice(
                        pool,
                        size=min(int(self.batch_size // 2), len(pool)),
                        replace=False,
                    )
                else:
                    neg_idxs = []

            if len(neg_idxs) == 0:  # if empty, sample from all negatives
                pool = self.df[
                    ((self.df["e1"] != e1) | (self.df["e2"] != e2))
                ].index
                neg_idxs = np.random.choice(
                    pool,
                    size=min(int(self.batch_size // 2), len(pool)),
                    replace=False,
                )
            Q = 1 / len(pool)

        batch = []
        ## process positive sample
        pos_df = self.df.loc[pos_idxs]
        for idx, row in pos_df.iterrows():
            r, e1, e2 = row[0], row[1], row[2]
            x, masked_for_pred, e1_e2_start = self.tokenize(
                self.put_blanks((r, e1, e2))
            )
            x = torch.LongTensor(x)
            masked_for_pred = torch.LongTensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            # e1, e2 = torch.tensor(e1), torch.tensor(e2)
            batch.append(
                (
                    x,
                    masked_for_pred,
                    e1_e2_start,
                    torch.FloatTensor([1.0]),
                    torch.LongTensor([1]),
                )
            )

        ## process negative samples
        negs_df = self.df.loc[neg_idxs]
        for idx, row in negs_df.iterrows():
            r, e1, e2 = row[0], row[1], row[2]
            x, masked_for_pred, e1_e2_start = self.tokenize(
                self.put_blanks((r, e1, e2))
            )
            x = torch.LongTensor(x)
            masked_for_pred = torch.LongTensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            # e1, e2 = torch.tensor(e1), torch.tensor(e2)
            batch.append(
                (
                    x,
                    masked_for_pred,
                    e1_e2_start,
                    torch.FloatTensor([Q]),
                    torch.LongTensor([0]),
                )
            )
        batch = self.PS(batch)
        return batch


class Pad_Sequence:
    """
    collate_fn for dataloader to collate sequences of different lengths into a
    fixed length batch Returns padded x sequence, y sequence, x lengths and y
    lengths of batch.
    """

    def __init__(
        self,
        seq_pad_value,
        label_pad_value=1,
        label2_pad_value=-1,
        label3_pad_value=-1,
        label4_pad_value=-1,
    ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        self.label3_pad_value = label3_pad_value
        self.label4_pad_value = label4_pad_value

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(
            seqs, batch_first=True, padding_value=self.seq_pad_value
        )
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_value
        )
        y_lengths = torch.LongTensor([len(x) for x in labels])

        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(
            labels2, batch_first=True, padding_value=self.label2_pad_value
        )
        y2_lengths = torch.LongTensor([len(x) for x in labels2])

        labels3 = list(map(lambda x: x[3], sorted_batch))
        labels3_padded = pad_sequence(
            labels3, batch_first=True, padding_value=self.label3_pad_value
        )
        y3_lengths = torch.LongTensor([len(x) for x in labels3])

        labels4 = list(map(lambda x: x[4], sorted_batch))
        labels4_padded = pad_sequence(
            labels4, batch_first=True, padding_value=self.label4_pad_value
        )
        y4_lengths = torch.LongTensor([len(x) for x in labels4])
        return (
            seqs_padded,
            labels_padded,
            labels2_padded,
            labels3_padded,
            labels4_padded,
            x_lengths,
            y_lengths,
            y2_lengths,
            y3_lengths,
            y4_lengths,
        )
