import torch
import torch.nn as nn
import transformers

def create_attention_mask(text_shape, length):
    pass

class BERT(nn.Module):
    def __init__(self, bert_dir):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_dir)
        self.fc = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, length):
        att_mask = create_attention_mask(text.shape, length)
        x = self.dropout(self.bert(text, attention_mask=att_mask))
        return self.fc(x)
        
        
        