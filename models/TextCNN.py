import torch
import torch.nn as nn
import torch.nn.functional as F
from data_split import hotel_split
from data_utils import load_htl_datasets, get_sanity_check_dataset

class TextCNN(nn.Module):
    
    def __init__(self, num_vocab, embed_dim, pad_idx, pretrained=None, num_filters=200,
                 filter_sizes=[2, 3, 4, 5], output_dim=2, dropout=0.5):
        super().__init__()
        
        if pretrained is None:
            # 随机初始化，可训练
            self.embedding = nn.Embedding(num_vocab, embed_dim, padding_idx=pad_idx)
        else:
            # 不可训练
            self.embedding = nn.Embedding.from_pretrained(pretrained)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=num_filters,
                                              kernel_size=(filter_size, embed_dim))
                                    for filter_size in filter_sizes])    
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters*len(filter_sizes), output_dim)
    
    def forward(self, x):
        # [ batch_size, seq_len ]
        
        #print(torch.sum(torch.isnan(x)))
        x = self.embedding(x).unsqueeze(1)
        #print(torch.sum(torch.isnan(x)))
        # [ batch_size, 1, seq_len, embed_dim ]
        
        conved = [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        
        # conved[i] = [ batch_size, num_filters, conved_height ]
        # conved_width equals to 1, and is squeezed
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # pooled[i] = [ batch_size, num_filters ]
        # pooled_height equals to 1, and is squeezed
        
        x = self.dropout(torch.cat(pooled, 1))
        
        # [ batch_size, num_filters*len(filter_sizes) ]
        
        x = self.fc(x)
        
        return x