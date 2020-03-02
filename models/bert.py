import torch
import torch.nn as nn
import transformers

class BERT(nn.Module):
    def __init__(self, bert_dir):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_dir)
    
    def forward(self, text, length):
        att_mask = self.create_attention_mask(text.shape, length)
        
        
    def create_attention_mask(self, shape, length):
        att_mask = torch.zeros(shape, dtype=torch.long)
        att_mask[:, :length] = 1
        return att_mask