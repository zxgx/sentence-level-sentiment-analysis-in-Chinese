import argparse
from importlib import import_module
import time 

import torch
import torch.nn as nn

from models import CNN, RNN, BERT
from data_utils import SentDatasetReader, BucketIterator

model_classes = {
    'cnn':CNN,
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
            config, val_set, tokenizer, shuffle=False, sort=True)
        )
        self.test_iter = BucketIterator(
            config, test_set, tokenizer, shuffle=False, sort=True)
        )
    
    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        best_acc, stop_asc = 0, 0
        for epoch in range(self.config.epochs):
            print('epoch: %d'%(epoch+1), end=' | ')
            epoch_acc, epoch_loss = self.train_step(
                train_iter, optimizer, criterion
            )
            
        
    def train_step(self, data_iter, optimizer, criterion):
        '''
        optimizer, criterion
        '''
        epoch_corrects, epoch_loss = 0, 0.0
        self.model.train()
        st = time.time()
        for it, batch in enumerate(data_iter):
            optimizer.zero_grad()
            
            review = batch['text']
            if self.config.include_length:
                length = batch['length']
                pred = self.model(review, length)
            else: 
                pred = self.model(review)
            label = batch['label']
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            corrects = torch.sum(torch.argmax(pred, dim=1)==label).item()
            epoch_corrects += corrects
            epoch_loss += loss.item()
        
        print("training loss: %.4f, training acc: %.4f"%(
                epoch_loss/len(train_iter), epoch
        print('epoch: %d | train_loss: %.4f, train_acc: %.4f'%
              (epoch+1, epoch_loss/len(train_iter), 
              epoch_corrects/len(train_iter.dataset)),
              end=' | ')
        
        if not self.val_iter is None:
            val_acc, val_loss = evaluate(model, val_iter, criterion, device, include_length)
            print('val_loss: %.4f, val_acc: %.4f'%(val_loss, val_acc), end=' | ')
            if val_acc > best_acc:
                best_acc, stop_asc = val_acc, 0
                if not save_dir is None:
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_path = os.path.join(save_dir, 'best.pth')
                    torch.save(model.state_dict(), save_path)
            else:
                stop_asc += 1
                if stop_asc == 5:
                    break

        et = time.time()
        print('time: %.2fs'%(et-st))        
            
        return best_acc
        
    def evaluate(self):
        '''
        criterion
        '''
        epoch_corrects, epoch_loss = 0, 0
        model.to(device)
        model.eval()
        dataset_size = len(data_iter.dataset)
        with torch.no_grad():
            for it, batch in enumerate(data_iter):
                if include_length:
                    (review, length), label = batch.review, batch.label
                    review, length, label = review.to(device), length.to(device), label.to(device)
                    pred = model(review, length)
                else:
                    review, label = batch.review.to(device), batch.label.to(device)
                    pred = model(review)
                    
                loss = criterion(pred, label)
                
                corrects = torch.sum((torch.argmax(pred, dim=1)) == label).item()
                epoch_corrects += corrects
                epoch_loss += loss.item()
                
        return epoch_corrects/dataset_size, epoch_loss/len(data_iter)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Chinese Sentence level Sentiment Analysis")
    
    # Model config
    parser.add_argument('--model_name', default='cnn', type=str, required=True, 
        help='Model name in: cnn / rnn / bert'
    )
    parser.add_argument('--bert_identifier', default='hfl/chinese-bert-wwm-ext', type=str, 
        help="Pretrained BERT model identifier. refer to hugging face for more details"
    )
    parser.add_argument('--bert_dir', default='caches/chinese_wwm_ext_pytorch', type=str, 
        help="Pretrained BERT model cache directory, download model directly to this dir"
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
    parser.add_argument('--include_length', default=True, type=bool,
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
        help="Fine-tuned model saved directory"
    )
    
    config = parser.parse_args()
    if config.model_name == 'bert':
        config.is_bert = True
    else:
        config.is_bert = False
    
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if \
        config.device is None else torch.device(config.device)
    
    print(vars(config))
    
    Manager(config)

