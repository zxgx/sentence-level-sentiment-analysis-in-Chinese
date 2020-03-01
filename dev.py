import csv
import io
import os
from itertools import chain
from collections import Counter

import torch
import jieba
from transformers import BertTokenizer

import logging

logger = logging.getLogger(__name__)

PAD, UNK = '[PAD]', '[UNK]' # align to BERT special tokens

def word_cut(text):
    '''
    Chinese text segmenter.
    
    Refer to https://github.com/fxsjy/jieba for more details.
    '''
    return [word for word in jieba.cut(text) if word.strip()]


def read_embedding(path):
    num, dim, word2vec = 0, None, {}

    with io.open(path, encoding='utf8') as f:
        for line in f:
            entries = line.rstrip().split(" ")
            word, entries = entries[0], entries[1:]
            
            if dim is None and len(entries) > 1:
                dim = len(entries)
            elif len(entries) == 1:
                logger.warning(
                    "Skipping token {} with 1-dimensional "
                    "vector {}; likely a header".format(word, entries)
                )
            elif dim!=len(entries):
                raise RuntimeError(
                    "Vector for token {} has {} dimensions, but previously "
                     "read vectors have {} dimensions. All vectors must have "
                     "the same number of dimensions.".format(
                        word, len(entries), dim
                    )
                )
            
            word2vec[word] = [ float(x) for x in entries ]
            num+=1
    logger.info("Read %d lines from %s"%(num, path))
    return word2vec


class SentTokenizer(object):
    def __init__(self, config):
        self.config = config

        self.tokenize = word_cut

        self.pad_token = PAD
        self.unk_token = UNK
        
        self.ids_to_tokens = []
        self.vocab = {}
        self.embedding = None
        self.build_vocab()

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token, self.vocab.get(self.UNK)) for token in tokens]
        return ids

    def build_vocab(self):
        self.ids_to_tokens.append(self.pad_token)
        self.ids_to_tokens.append(self.unk_token)
        
        tok_cache = self.config.tok_cache
        if not os.path.exists(tok_cache):
            dataset_path = os.path.join(self.config.dataset_dir, 'train.csv')
            logger.info(
                "Building vocab & embedding from training set:%s"%dataset_path
            )
            counter = Counter()
            word2vec = read_embedding(self.config.embedding_path)
            embedding = []
            with io.open(dataset_path, encoding="utf8") as f:
                reader = csv.reader(f)
                if self.config.skip_header:
                    next(reader)
                counter.update(
                    chain.from_iterable(self.tokenize(line[1]) for line in reader)
                )
                for key in counter.keys():
                    if key in word2vec:
                        self.ids_to_tokens.append(key)
                        embedding.append(word2vec[key])
            embedding = torch.tensor(embedding, dtype=torch.float)
            unk_embed = torch.mean(embedding, dim=0, keepdim=True)
            pad_embed = torch.zeros_like(unk_embed, keepdim=True)
            logger.info(pad_embed.shape, unk_embed.shape, embedding.shape)
            self.embedding = torch.cat((pad_embed, unk_embed, embedding), dim=0)
            
            torch.save((self.ids_to_tokens, self.embedding), tok_cache)
        else:
            logger.info("Loading vocab & embedding from %s"%tok_cache)
            self.ids_to_tokens, self, embedding = torch.load(tok_cache)
        
        self.vocab = { k: v for v, k in enumerate(self.ids_to_tokens) }


class SentDataset(object):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class SentDatasetReader(object):
    def __init__(self, config):
        self.config = config
        
        # tokenizer
        if config.is_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_identifier)
        else:
            self.tokenizer = SentTokenizer(config)
        
        # datasets
        self._train_set = None
        self._val_set = None
        self._test_set = None

    @property
    def train_set(self):
        if self._train_set is None:
            logger.info('Reading TRAINING set from %s'%(
                os.path.join(self.config.dataset_dir, 'train.csv'))
            )
            
            self._train_set = SentDataset(
                self.read_examples(
                    os.path.join(self.config.dataset_dir, 'train.csv'))
            )
            
        return self._train_set

    @property
    def val_set(self):
        if self._val_set is None:
            logger.info('Reading VALIDATION set from %s'%(
                os.path.join(self.config.dataset_dir, 'val.csv'))
            )
            
            self._val_set = SentDataset(
                self.read_examples(
                    os.path.join(self.config.dataset_dir, 'val.csv'))
            )
            
        return self._val_set

    @property
    def test_set(self):
        if self._test_set is None:
            logger.info('Reading TEST set from %s'%(
                os.path.join(self.config.dataset_dir, 'test.csv'))
            )
            
            self._test_set = SentDataset(
                self.read_examples(
                    os.path.join(self.config.dataset_dir, 'test.csv'))
            )

    def read_examples(self, path):
        examples = []
        with io.open(path, encoding="utf8") as f:
            reader = csv.reader(f)
            if self.config.skip_header:
                next(reader)
            for line in reader:
                if len(line) != 2:
                    logger.warning("Unexpected line: %s"%line)
                    continue
                label, text = int(line[0]), self.tokenizer.encode(line[1])
                data = {'label':label, 'text':text}
                examples.append(data)

        logger.info("Read %d lines from %s"%(len(examples), path))
        return examples


