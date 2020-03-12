import argparse
import time 
import os

import torch
import torch.nn as nn

from models import TextCNN, RNN, BERT
from data_utils import SentDatasetReader, BucketIterator

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    def __init__(self):
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
        parser.add_argument('--repeat', default=5, type=int, required=True,
            help="Repeat for stable statistic of model"
        )
        parser.add_argument('--epochs', default=20, type=int, 
            help="Training epochs"
        )
        parser.add_argument('--device', default=None, type=str,
            help="Training device"
        )
        parser.add_argument('--save_dir', default=None, type=str,
            help="Directory where fine-tuned models are saved"
        )
        
        self.config = parser.parse_args()
        
        if self.config.model_name == 'bert':
            self.config.is_bert = True
        else:
            self.config.is_bert = False
        
        self.config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if \
            self.config.device is None else torch.device(self.config.device)
        
        for k, v in vars(self.config).items():
            print(k.upper(), ":", v)
            
        self.reader = SentDatasetReader(self.config)
    
    def train(self, model, optimizer, criterion, train_iter, val_iter):
        best_acc, stop_asc = 0, 0
        tmp_cache = self.config.model_name + ".tmp"
        for epoch in range(self.config.epochs):
            epoch_corrects, epoch_loss = 0, 0
            model.train()
            st = time.time()
            for batch in train_iter:
                optimizer.zero_grad()
                text, label = batch['text'], batch['label']
                if self.config.include_length:
                    length = batch['length']
                    pred = model(text, length)
                else:
                    pred = model(text)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                
                corrects = torch.sum(torch.argmax(pred, dim=1)==label).item()
                epoch_corrects += corrects
                epoch_loss += loss.item()
                
            print('epoch: %d | train loss: %.4f, train acc: %.4f'%
                (epoch+1, epoch_loss/len(train_iter), epoch_corrects/len(train_iter.dataset)), 
                end =' | '
            )
            val_acc, val_loss = self.evaluate(model, criterion, val_iter)
            print('val_loss: %.4f, val_acc: %.4f'%(val_loss, val_acc), end=' | ')
            print('time cost: %.2fs'%(time.time()-st))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), tmp_cache)
                stop_asc = 0
            else:
                stop_asc += 1

            if stop_asc == 10:
                print("Early stopping at epoch:", epoch+1)
                break
        model.load_state_dict(torch.load(tmp_cache))
        os.remove(tmp_cache)
        return best_acc
        
    def evaluate(self, model, criterion, val_iter):
        epoch_corrects, epoch_loss = 0, 0
        model.eval()
        with torch.no_grad():
            for batch in val_iter:
                text, label = batch['text'], batch['label']
                if self.config.include_length:
                    length = batch['length']
                    pred = model(text, length)
                else:
                    pred = model(text)
                
                loss = criterion(pred, label)
                corrects = torch.sum((torch.argmax(pred, dim=1)) == label).item()
                epoch_corrects += corrects
                epoch_loss += loss.item()
                
        return epoch_corrects/len(val_iter.dataset), epoch_loss/len(val_iter)
    
    def run(self):
        criterion = nn.CrossEntropyLoss()

        train_set = self.reader.train_set
        val_set = self.reader.val_set
        test_set = self.reader.test_set
        tokenizer = self.reader.tokenizer
        train_iter = BucketIterator(
            self.config, train_set, tokenizer, shuffle=True, sort=True
        )
        val_iter = BucketIterator(
            self.config, val_set, tokenizer, shuffle=False, sort=True
        )
        test_iter = BucketIterator(
            self.config, test_set, tokenizer, shuffle=False, sort=True
        )
        
        val_max_acc, val_avg_acc = 0, 0
        test_max_acc, test_avg_acc = 0, 0
        for it in range(self.config.repeat):
            if self.config.is_bert:
                model = model_classes[self.config.model_name](
                    self.config.bert_dir, 
                    self.config.freeze
                ).to(self.config.device)
            else:
                model = model_classes[self.config.model_name](
                    tokenizer.embedding, 
                    tokenizer.pad_token_id, 
                    self.config.freeze
                ).to(self.config.device)
            optimizer = torch.optim.Adam(model.parameters()) # lr
            
            it_st = time.time()
            print('='*30, "Iteration: ", (it+1), '='*30)
            val_acc = self.train(model, optimizer, criterion, train_iter, val_iter)
            test_acc, _ = self.evaluate(model, criterion, test_iter)
            print('Time cost this iteration: %.2fs | VAL acc: %.4f | TEST acc: %.4f'%
                ((time.time()-it_st), val_acc, test_acc)
            )
            if val_acc > val_max_acc:
                val_max_acc = val_acc
            if test_acc > test_max_acc:
                if self.config.save_dir is not None:
                    pth = os.path.join(self.config.save_dir, self.config.model_name+".pt")
                    torch.save(
                        model.state_dict(), 
                        pth
                    )
                    logger.info("Best model saved at %s, in it %d"%(
                        pth, it+1)
                    )
                test_max_acc = test_acc
            val_avg_acc += val_acc
            test_avg_acc += test_acc
        val_avg_acc /= self.config.repeat
        test_avg_acc /= self.config.repeat
        print("MAX VAL acc: %.4f, AVG VAL acc: %.4f"%(val_max_acc, val_avg_acc))
        print("MAX TEST acc: %.4f, AVG TEST acc: %.4f"%(test_max_acc, test_avg_acc))
    
    def insight(self):
        if self.config.save_dir is None:
            print("Model is not saved")
            return
        
        tokenizer = self.reader.tokenizer
        test_set = self.reader.test_set
        test_iter = BucketIterator(
            self.config, test_set, tokenizer, 
            shuffle=False, sort=True
        )
        lt = tokenizer.ids_to_tokens
        if self.config.is_bert:
            model = model_classes[self.config.model_name](
                self.config.bert_dir, 
                self.config.freeze
            ).to(self.config.device)
        else:
            model = model_classes[self.config.model_name](
                tokenizer.embedding, 
                tokenizer.pad_token_id, 
                self.config.freeze
            ).to(self.config.device)
        pth = os.path.join(self.config.save_dir, self.config.model_name+".pt")
        model.load_state_dict(torch.load(pth))
        model.eval()
        
        correct = 0
        with torch.no_grad():
            for batch in test_iter:
                text, label = batch['text'], batch['label']
                if self.config.include_length:
                    length = batch['length']
                    pred = model(text, length)
                else:
                    pred = model(text)
                y = torch.argmax(pred, dim=1)
                correct += torch.sum(y==label).item()
                wrong = text[y!=label].tolist()
                wrong_label = y[y!=label].tolist()
                for sent, wl in zip(wrong, wrong_label):
                    print("wrong predict to", wl, end=":\t")
                    for id in sent:
                        print(lt[id], end=" ")
                    print('\n')
        print("acc: %.4f"%(correct/len(test_iter.dataset)))


if __name__=='__main__':

    ####################################################################
    """ Sanity check
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
    # for rnn, larger lr is a better choice, while for bert, smaller lr is better
    optimizer = torch.optim.Adam(model.parameters())
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
    """
    
    m = Manager()
    m.run()
    m.insight()
