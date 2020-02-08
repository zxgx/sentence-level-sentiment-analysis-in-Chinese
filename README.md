# 1.7
## hotel数据集部分的进度 
1. data_split.hotel_split：pandas读取并分割数据集，numpy随机索引，可复现。
2. （1.7 12：00）data_utils.load_htl_datasets：利用TorchText建立了数据集的Iterator，可复现。包含内容：Fields，Datasets，Vocab，Vectors，Iterators。

TODO：  
1. ~~预训练的embedding在vocab的vectors（Tensor）中，unk和pad初始化方式统一，这里设定为随机。但预想的是，unk初始化合理，pad初始化为0。~~
2. rnn可能会用到include_length参数，以及pad_packed_sequeence相关的一对函数
3. 分词用的jieba简单处理，复杂的文本可能需要re模块预先处理一遍
4. 生成batch时，为了复现，固定随机数的操作不知道合不合理
5. weibo数据集好像质量不行，预训练embedding不知道需不需要换，可以尝试char2vec
6. **有时间试下自己建一个数据集类**

## TextCNN模型部分的进度 
models.py：建立了模型的基本结构  

TODO：  
~~embedding这一模块，理想是unk随机初始化，训练可变，pad全0不可变，或者unk合理，pad全0，都不可变~~  
调参方向：单层更多卷积核，学习更多n-gram；尝试深度卷积网络，加更多卷积层  
函数接口  
~~训练时记得.T~~

# 1.8 21：30
pad的embedding全0，unk的embedding为预训练所有embedding的平均值，训练时所有embedding固定，因为vocab规模较小

# 1.9 15：00
utils.train & utils.evaluate：定义了train和evaluate函数的基本结构，能通过完整性检查

TODO：  
训练过程中，输出过程和early stopping需要更细粒度的迭代计数  
**训练集中含有，而预训练词向量中不含的单词，词向量初始化为创建Vectors时定义的初始化方式，尝试从源码解决，或直接手动重写数据集类**  
方案1：unk和pad都为0  
方案2：unk为平均，pad为0

# 1.30
TextCNN的embedding，可以建立一个动态一个静态  

# 2.8
增加了完整性检查相关代码，整合了一下cnn和rnn的接口，还需要进一步改进为面向对象形式。  
RNN训练太慢，应该是模型本身问题，完整性检查如下：
## TextCNN sanity check
代码：
```python
import random
import torch
import torch.nn as nn
import torch.optim as optim

from data_split import hotel_split
from data_utils import load_htl_datasets, get_sanity_check_dataset
from models.TextCNN import TextCNN
from models.RNN import RNN
from utils import train, num_parameters


train_iter, val_iter, test_iter = load_htl_datasets(batch_first=True, 
                                                    include_length=False,
                                                    sort_within_batch=None)
sanity_iter = get_sanity_check_dataset(val_iter.dataset,
                                       num_examples=50,
                                       seed=None,
                                       sort_within_batch=None)

TEXT = sanity_iter.dataset.fields['review']
model = TextCNN(num_vocab=len(TEXT.vocab), 
                embed_dim=300, 
                pad_idx=TEXT.vocab.stoi[TEXT.pad_token],
                pretrained=None,
                num_filters=200,
                filter_sizes=[2, 3, 4, 5], 
                output_dim=2,
                dropout=0.5)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
device = torch.device('cpu')
epochs = 20
print(len(sanity_iter.dataset))
print(num_parameters(model))
train(model, sanity_iter, optimizer, criterion, epochs, device, include_length=False)
```
结果：
```
50
3573002
epoch: 1 | train_loss: 0.8183, train_acc: 0.6000 | time: 7.31s
epoch: 2 | train_loss: 0.7071, train_acc: 0.6000 | time: 5.75s
epoch: 3 | train_loss: 0.2090, train_acc: 0.9200 | time: 5.28s
epoch: 4 | train_loss: 0.2298, train_acc: 0.8800 | time: 6.09s
epoch: 5 | train_loss: 0.1210, train_acc: 0.9400 | time: 5.47s
epoch: 6 | train_loss: 0.0550, train_acc: 1.0000 | time: 5.58s
epoch: 7 | train_loss: 0.0339, train_acc: 1.0000 | time: 5.91s
epoch: 8 | train_loss: 0.0233, train_acc: 1.0000 | time: 5.39s
epoch: 9 | train_loss: 0.0250, train_acc: 1.0000 | time: 5.49s
epoch: 10 | train_loss: 0.0109, train_acc: 1.0000 | time: 5.51s
epoch: 11 | train_loss: 0.0264, train_acc: 1.0000 | time: 5.73s
epoch: 12 | train_loss: 0.0137, train_acc: 1.0000 | time: 7.70s
epoch: 13 | train_loss: 0.0111, train_acc: 1.0000 | time: 6.37s
epoch: 14 | train_loss: 0.0063, train_acc: 1.0000 | time: 6.79s
epoch: 15 | train_loss: 0.0032, train_acc: 1.0000 | time: 7.53s
epoch: 16 | train_loss: 0.0036, train_acc: 1.0000 | time: 5.61s
epoch: 17 | train_loss: 0.0021, train_acc: 1.0000 | time: 6.65s
epoch: 18 | train_loss: 0.0026, train_acc: 1.0000 | time: 5.79s
epoch: 19 | train_loss: 0.0021, train_acc: 1.0000 | time: 5.49s
epoch: 20 | train_loss: 0.0016, train_acc: 1.0000 | time: 5.30s
```

## RNN sanity check
代码：
```
train_iter, val_iter, test_iter = load_htl_datasets(batch_first=False, 
                                                    include_length=True,
                                                    sort_within_batch=True)
sanity_iter = get_sanity_check_dataset(val_iter.dataset,
                                       num_examples=50,
                                       seed=None,
                                       sort_within_batch=True)

TEXT = sanity_iter.dataset.fields['review']
model = RNN(vocab_size=len(TEXT.vocab), 
            embedding_dim=300, 
            pad_idx=TEXT.vocab.stoi[TEXT.pad_token], 
            hidden_dim=100, output_dim=2, dropout=0.5,
            n_layers=2, bidirectional=True)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
device = torch.device('cpu')
epochs = 20
print(len(sanity_iter.dataset))
print(num_parameters(model))
train(model, sanity_iter, optimizer, criterion, epochs, device, include_length=True)
```
结果：
```
50
3294202
epoch: 1 | train_loss: 0.6911, train_acc: 0.4800 | time: 20.92s
epoch: 2 | train_loss: 0.6822, train_acc: 0.6000 | time: 20.51s
epoch: 3 | train_loss: 0.6644, train_acc: 0.7000 | time: 20.37s
epoch: 4 | train_loss: 0.6691, train_acc: 0.6400 | time: 20.57s
epoch: 5 | train_loss: 0.6496, train_acc: 0.7600 | time: 20.51s
epoch: 6 | train_loss: 0.6333, train_acc: 0.7200 | time: 19.31s
epoch: 7 | train_loss: 0.6254, train_acc: 0.7400 | time: 20.70s
epoch: 8 | train_loss: 0.6269, train_acc: 0.7000 | time: 20.24s
epoch: 9 | train_loss: 0.6023, train_acc: 0.7400 | time: 19.65s
epoch: 10 | train_loss: 0.5802, train_acc: 0.7800 | time: 19.82s
epoch: 11 | train_loss: 0.5726, train_acc: 0.7800 | time: 19.24s
epoch: 12 | train_loss: 0.5629, train_acc: 0.8000 | time: 20.45s
epoch: 13 | train_loss: 0.5491, train_acc: 0.8000 | time: 20.85s
epoch: 14 | train_loss: 0.5206, train_acc: 0.7800 | time: 19.26s
epoch: 15 | train_loss: 0.4916, train_acc: 0.8000 | time: 19.08s
epoch: 16 | train_loss: 0.4886, train_acc: 0.8000 | time: 19.31s
epoch: 17 | train_loss: 0.4750, train_acc: 0.8000 | time: 19.29s
epoch: 18 | train_loss: 0.4196, train_acc: 0.8000 | time: 19.33s
epoch: 19 | train_loss: 0.4201, train_acc: 0.8200 | time: 19.51s
epoch: 20 | train_loss: 0.4076, train_acc: 0.8200 | time: 19.25s
```