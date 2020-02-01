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