import argparse
import time 

import torch
import torch.nn as nn

from models import TextCNN, RNN, BERT
from data_utils import SentDatasetReader, BucketIterator

import logging

logging.basicConfig(level=logging.INFO)

model_classes = {
    'cnn':TextCNN,
    'rnn':RNN,
    'bert':BERT
}

class Manager(object):
    '''
    device
    dtype
    '''
    def __init__(self, config):
        self.config = config
        
        self.model = model_classes[config.model_name](config).to(config.device)
        
        self.reader = SentDatasetReader(config)
        train_set = reader.train_set
        val_set = reader.val_set
        test_set = reader.test_set
        tokenizer = reader.tokenizer
        
        self.train_iter = BucketIterator(
            config, train_set, tokenizer, shuffle=True, sort=True
        )
        self.val_iter = BucketIterator(
            config, val_set, tokenizer, shuffle=False, sort=True
        )
        self.test_iter = BucketIterator(
            config, test_set, tokenizer, shuffle=False, sort=True
        )


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Chinese Sentence level Sentiment Analysis")
    
    # Model config
    parser.add_argument('--model_name', default='cnn', type=str, required=True, 
        help='Model name in: cnn / rnn / bert'
    )
    parser.add_argument('--bert_identifier', default='hfl/chinese-bert-wwm-ext', type=str, 
        help="Pretrained BERT model identifier. Refer to hugging face for more details"
    )
    parser.add_argument('--bert_dir', default='caches/chinese_wwm_ext_pytorch', type=str, 
        help="Pretrained BERT model cache directory, download model directly to this dir"
    )
    parser.add_argument('--freeze', default=True, type=bool, required=True,
        help="Whether to freeze weights of embeddings"
    )
    
    # Dataset config
    parser.add_argument('--dataset_dir', default='data/hotel', type=str, required=True,
        help="Data splits directory"
    )
    parser.add_argument('--skip_header', default=True, type=bool, required=True,
        help="Whether to skip one line for every data splits"
    )
    
    # Tokenizer
    parser.add_argument('--tok_cache', default='caches/hotel_tokenizer.pt', type=str, 
        help="Tokenizer cache, including vocab & embedding, used by rnn & cnn"
    )
    parser.add_argument('--embedding_path', default='caches/sgns.renmin.bigram-char', type=str,
        help="Pretrained embedding file, used by rnn & cnn"
    )
    
    # Iterator
    parser.add_argument('--batch_size', default=64, type=int,
        help="Batch size for iterator"
    )
    parser.add_argument('--max_len', default=None, type=int,
        help="Max length for every example. Neccessary for bert, including cls & sep token"
    )
    parser.add_argument('--include_length', default=False, type=bool,
        help="Whether to include length when reading examples. Neccessary for bert & rnn"
    )
    
    # Running Config
    parser.add_argument('--epochs', default=20, type=int, 
        help="Training epochs"
    )
    parser.add_argument('--device', default=None, type=str,
        help="Training device"
    )
    parser.add_argument('--save_dir', default=None, type=str,
        help="Directory where fine-tuned models are saved"
    )
    
    config = parser.parse_args()
    
    if config.model_name == 'bert':
        config.is_bert = True
    else:
        config.is_bert = False
    
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if \
        config.device is None else torch.device(config.device)
    
    print(vars(config))
    
    ####################################################################
    
    print(torch.cuda.current_device())
    reader = SentDatasetReader(config)
    
    tokenizer = reader.tokenizer
    print("vocab size:", len(tokenizer.ids_to_tokens))
    
    train_set, val_set, test_set = reader.train_set, reader.val_set, reader.test_set
    print(len(train_set), len(val_set), len(test_set))
    
    train_iter = BucketIterator(config, train_set[:50], tokenizer, shuffle=True)
    
    
    # cnn & rnn
    # model = model_classes[config.model_name](tokenizer.embedding, tokenizer.pad_token_id, config.freeze).to(config.device)
    
    
    # bert
    model = model_classes[config.model_name](config.bert_dir, config.freeze).to(config.device)
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad)
    print("totoal params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # for rnn, larger lr is a better choice
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config.epochs):
        for batch in train_iter:
            optimizer.zero_grad()
            text, label = batch['text'], batch['label']
            if config.include_length:
                length = batch['length']
                pred = model(text, length)
            else:
                pred = model(text)
            loss = criterion(pred, label)
            loss.backward()
            print(loss.item(), torch.cuda.memory_allocated())
            optimizer.step()

