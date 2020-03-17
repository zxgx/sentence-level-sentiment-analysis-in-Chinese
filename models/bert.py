import torch
import torch.nn as nn
from transformers import BertModel

def create_attention_mask(text, length):
    mask = torch.zeros(text.shape, device=text.device)
    for i in range(len(length)):
        d1 = torch.tensor([i]).repeat(length[i])
        d2 = torch.arange(length[i])
        mask.index_put_((d1,d2), torch.tensor(1, dtype=mask.dtype))
    return mask


class BERT(nn.Module):
    def __init__(self, bert_dir, freeze, output_dim=2):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_dir)
        self.bert.embeddings.requires_grad_(not freeze)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
    
    def forward(self, text, length, output_attentions=False):
        att_mask = create_attention_mask(text, length)
        res = self.bert(text, attention_mask=att_mask)
        x = res[0]
        x = x[:, 0] # cls token for classification
        # sentence mean for classification, recommended by huggingface
        # x = x.mean(dim=1) 
        if output_attentions:
            return self.fc(x), res[2]
        else:
            return self.fc(x)

