import argparse
import time 
import os
import io
import copy

import torch
import torch.nn as nn

from sklearn import metrics

import seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt

from models import TextCNN, RNN, BERT, Att_RNN, RCNN
from data_utils import SentDatasetReader, BucketIterator

#import logging
#logging.basicConfig(level=logging.INFO)

model_classes = {
    'cnn':TextCNN,
    'rnn':RNN,
    'att-rnn':Att_RNN,
    'rcnn':RCNN,
    'bert':BERT
}

def str2bool(v):
    if v=='True':
        return True
    else:
        return False

#mpl.rcParams['font.sans-serif'] = ['SimHei']
#mpl.rcParams['axes.unicode_minus'] = False
#seaborn.set(font='SimHei')
#print(seaborn.axes_style())
def draw_attention(data, x, y, ax, cbar):
    seaborn.heatmap(data, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
        cbar=cbar, ax=ax)

class Manager(object):
    '''
    '''
    def __init__(self):
        parser = argparse.ArgumentParser(description="Chinese Sentence level Sentiment Analysis")
        
        # Model config
        parser.add_argument('--model_name', type=str, required=True, 
            help='Model name in: cnn / rnn / bert'
        )
        parser.add_argument('--bert_identifier', default='hfl/chinese-bert-wwm-ext', type=str, 
            help="Pretrained BERT model identifier. Refer to hugging face for more details"
        )
        parser.add_argument('--bert_dir', default='caches/chinese_wwm_ext_pytorch', type=str, 
            help="Pretrained BERT model cache directory, download model directly to this dir"
        )
        parser.add_argument('--freeze', type=str2bool, required=True,
            help="Whether to freeze weights of embeddings"
        )

        # Dataset config
        parser.add_argument('--dataset_dir', type=str, required=True,
            help="Data splits directory"
        )
        parser.add_argument('--skip_header', default=True, type=str2bool,
            help="Whether to skip the first line for every data splits"
        )
        
        # Tokenizer
        parser.add_argument('--tok_cache', default='caches/tokenizer.pt', type=str, 
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
        parser.add_argument('--include_length', default=False, type=str2bool,
            help="Whether to include length when reading examples. Neccessary for bert & rnn"
        )
        
        # Running Config
        parser.add_argument('--repeat', type=int, required=True,
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
        train_set = self.reader.train_set
        val_set = self.reader.val_set
        test_set = self.reader.test_set
        self.tokenizer = self.reader.tokenizer
        self.train_iter = BucketIterator(
            self.config, train_set, tokenizer, shuffle=True, sort=True
        )
        self.val_iter = BucketIterator(
            self.config, val_set, tokenizer, shuffle=False, sort=True
        )
        self.test_iter = BucketIterator(
            self.config, test_set, tokenizer, shuffle=False, sort=True
        )
    
    def train(self, model, optimizer, criterion, train_iter, val_iter):
        best_f1, stop_asc = 0.0, 0
        tmp_cache = self.config.model_name + ".tmp"
        for epoch in range(self.config.epochs):
            epoch_corrects, epoch_loss = 0, 0.0
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
            val_loss, val_acc, val_f1 = self.evaluate(model, criterion, val_iter)
            print('val loss: %.4f, val acc: %.4f, val f1: %.4f'%(val_loss, val_acc, val_f1), end=' | ')
            print('time cost: %.2fs'%(time.time()-st))
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), tmp_cache)
                stop_asc = 0
            else:
                stop_asc += 1

            if stop_asc == 10:
                print("Early stopping at epoch:", epoch+1)
                break
        model.load_state_dict(torch.load(tmp_cache))
        os.remove(tmp_cache)
        return best_f1
        
    def evaluate(self, model, criterion, val_iter):
        epoch_corrects, epoch_loss = 0, 0.0
        y_pred, y_true = None, None
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
                
                if y_pred is None:
                	y_pred = pred
                	y_true = label
                else:
                	y_pred = torch.cat((y_pred, pred), dim=0)
                	y_true = torch.cat((y_true, label), dim=0)

        f1 = metrics.f1_score(y_true.cpu(), torch.argmax(y_pred, dim=1).cpu() )
        return epoch_loss/len(val_iter), epoch_corrects/len(val_iter.dataset), f1
    
    def run(self):
        criterion = nn.CrossEntropyLoss()        
        
        val_max_f1, val_avg_f1 = 0.0, 0.0
        # test_max_acc, test_avg_acc = 0.0, 0.0
        test_max_f1, test_avg_f1 = 0.0, 0.0
        for it in range(self.config.repeat):
            if self.config.is_bert:
                model = model_classes[self.config.model_name](
                    self.config.bert_dir, 
                    self.config.freeze
                ).to(self.config.device)
            else:
                model = model_classes[self.config.model_name](
                    self.tokenizer.embedding, 
                    self.tokenizer.pad_token_id, 
                    self.config.freeze
                ).to(self.config.device)
            #optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.05, lr=2e-5)
            optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.05)
            
            it_st = time.time()
            print('='*30, "Iteration: ", (it+1), '='*30)
            val_f1 = self.train(model, optimizer, criterion, self.train_iter, self.val_iter)
            _, _, test_f1 = self.evaluate(model, criterion, self.test_iter)
            print('Time cost this iteration: %.2fs | VAL f1: %.4f | TEST acc: %.4f, TEST f1: %.4f'%
                ((time.time()-it_st), val_f1, test_acc, test_f1)
            )
            if val_f1 > val_max_f1:
                val_max_f1 = val_f1
            if test_f1 > test_max_f1:
                if self.config.save_dir is not None:
                    pth = os.path.join(self.config.save_dir, self.config.model_name+".pt")
                    torch.save(
                        model.state_dict(), 
                        pth
                    )
                    print("Best model saved at %s, in it %d"%(
                        pth, it+1)
                    )
                test_max_f1 = test_f1
            val_avg_f1 += val_f1
            test_avg_f1 += test_f1
        val_avg_f1 /= self.config.repeat
        test_avg_f1 /= self.config.repeat
        print("MAX VAL f1: %.4f, AVG VAL f1: %.4f"%(val_max_f1, val_avg_f1))
        print("MAX TEST f1: %.4f, AVG TEST f1: %.4f"%(test_max_f1, test_avg_f1))
    
    def insight(self):
        if self.config.save_dir is None:
            print("Model is not saved")
            return

        lt = self.tokenizer.ids_to_tokens
        if self.config.is_bert:
            model = model_classes[self.config.model_name](
                self.config.bert_dir, 
                self.config.freeze
            ).to(self.config.device)
        else:
            model = model_classes[self.config.model_name](
                self.tokenizer.embedding, 
                self.tokenizer.pad_token_id, 
                self.config.freeze
            ).to(self.config.device)
        pth = os.path.join(self.config.save_dir, self.config.model_name+".pt")
        model.load_state_dict(torch.load(pth))
        model.eval()
        
        f = io.open(self.config.model_name+'.log', 'w', encoding='utf-8')
        with torch.no_grad():
            corrects = 0
            for batch in self.test_iter:
                text, label = batch['text'], batch['label']
                if self.config.include_length:
                    length = batch['length']
                    pred = model(text, length)
                else:
                    pred = model(text)
                y = torch.argmax(pred, dim=1)
                corrects += torch.sum(y!=label).item()
                wrong = text[y!=label].tolist()
                wrong_label = y[y!=label].tolist()
                for sent, wl in zip(wrong, wrong_label):
                    f.write("wrong predict to "+str(wl)+":\t")
                    for id in sent:
                        f.write(lt[id]+' ')
                    f.write('\n\n')
            f.write("error ratio: %d / %d"%(corrects, len(self.test_iter.dataset)))
        f.close()

    def attention_map(self, sent):
        if self.config.save_dir is None:
            print("Model is not saved")
            return 

        if not self.config.is_bert:
            print("Only bert model has attention map")
            return

        tokenizer = self.tokenizer
        model = model_classes[self.config.model_name](
            self.config.bert_dir,
            self.config.freeze
        ).to(self.config.device)
        pth = os.path.join(self.config.save_dir, self.config.model_name+".pt")
        model.load_state_dict(torch.load(pth))
        model.eval()
        model.bert.encoder.output_attentions = True
        for layer in model.bert.encoder.layer:
            layer.attention.self.output_attentions=True

        tokens = tokenizer.tokenize(sent)
        if len(tokens) > self.config.max_len-2:
            tokens = tokens[:self.config.max_len-2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        ids = [ tokenizer.vocab[token] for token in tokens]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.config.device)
        length = torch.tensor([len(ids)], dtype=torch.long).to(self.config.device)
        # mask = (x!=tokenizer.pad_token_id).unsqeeze(0).to(self.config.device)
        with torch.no_grad():
           pred, att = model(x, length, output_attentions=True)
        print(pred, len(att))
        for l_no, att_map in enumerate(att):
            att_map = att_map.cpu().detach().numpy()
            for row in range(4):
                fig, axs = plt.subplots(1, 3, figsize=(20, 10))
                for col in range(3):
                    draw_attention(att_map[0, row*3+col], tokens, 
                    tokens if col==0 else [], ax=axs[col], cbar=col==2)
                plt.title("Bert Layer %d %d-%d attention"%(l_no+1, row*3+1, (row+1)*3))
                plt.savefig("layer_%d_%d-%d.jpg"%(l_no+1, row*3+1, (row+1)*3))
                plt.show()


def statistic(dataset):
	n, p = 0, 0
	for item in dataset:
		if item['label'] == 1:
			p+=1
		else:
			n+=1

	print("total: %d, pos: %d, neg: %d"%(len(dataset), p, n))

if __name__=='__main__':
    
    # sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    m = Manager()
    # tokenizer = m.reader.tokenizer
    # print(len(tokenizer.ids_to_tokens))
    statistic(m.reader.train_set)
    statistic(m.reader.val_set)
    statistic(m.reader.test_set)
 	# m.run()
	# m.insight()
    # m.attention_map("周边环境较差，服务的速度慢，态度还可以，价格太高。")

