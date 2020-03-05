import torch
import torch.nn as nn
import transformers

def create_attention_mask(text_shape, length):
    mask = torch.zeros(text_shape)
    for i in range(length):
        d1 = torch.tensor([i]).repeat(length[i])
        d2 = torch.arange(length[i])
        mask.index_put_((d1,d2), torch.tensor(1))
    return mask


class BERT(nn.Module):
    def __init__(self, bert_dir):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_dir)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
    
    def forward(self, text, length):
        att_mask = create_attention_mask(text.shape, length)
        x = self.bert(text, attention_mask=att_mask)
        x = x[:, 0] # cls token for classification
        # sentence mean for classification, recommended by huggingface
        # x = x.mean(dim=1) 
        return self.fc(x)

