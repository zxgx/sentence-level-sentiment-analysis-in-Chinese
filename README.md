# Chinese sentence level sentiment analysis
模型：TextCNN、Bi-LSTM、RCNN、BERT  
另外还有个Att-RNN，用self-attention做的LSTM，可以对比下Bi-LSTM、Att-RNN、RCNN的作用。
实验结果是：BERT > RCNN > TextCNN $\approx$ Bi-LSTM > Att-RNN  
BERT很强，RCNN效果出乎意料，一层双向LSTM加池化顶三层双向LSTM，倒是LSTM，很难训练
## Requirements
就自己看看，缺啥pip啥吧，没啥特殊配置，主要记下分词和预训练相关的东西
### 中文分词
适用模型：TextCNN、RNN、RCNN  
[结巴](https://github.com/fxsjy/jieba)
### 词向量
[sgns.renmin.bigram-char](https://github.com/Embedding/Chinese-Word-Vectors)
### 中文BERT
[bert-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)  
使用hugging face提供的接口很方便，但个人觉得学习成本挺高，因为我想用什么库就很想把它弄明白，但这些库为了通用性，搞得很复杂。其实明白BERT原理，看看hugging face的文档也能用。  
### 数据集相关
一点个人的感想。  
数据集我是自定义的。做这个实验的时候（2020.1），Torchtext版本大概是0.4，没有很好的教程，我用读代码的方式学了下，感觉还是为了通用性牺牲了很多可读性，有这个学习时间，自定义一个数据集都足够了，而且做了才发现其实数据集自定义也不是很难的。最坑的是，Torchtext至少0.4版本是不好处理BERT输入的，我没细想，但想跟hugging face的预训练模型结合起来，还是自定义比较好，早知道这点我就直接自定义了。
## 运行
我做的实验参数都列在下面，具体的自行体会吧。
### TextCNN
```bash
python main.py --model_name=cnn --dataset_dir=data/chnsenticorp --freeze=True --repeat=10 --epochs=100 --save_dir=.
```
### Bi-LSTM
```bash
python main.py --model_name=rnn --dataset_dir=data/chnsenticorp --freeze=True --include_length=True --repeat=10 --epochs=100 --save_dir=.
```
### RCNN
```bash
python main.py --model_name=rcnn --dataset_dir=data/chnsenticorp --freeze=True --include_length=True --repeat=10 --epochs=100 --save_dir=.
```
### BERT
```bash
python main.py --model_name=bert --dataset_dir=data/chnsenticorp --freeze=True --include_length=True --max_len=128 --batch_size=32 --repeat=10 --epochs=100 --save_dir=. 
```
BERT设置了max_len和更小的batch_size，主要就是出于显存不够的考虑，这两个值可以量力而行。  
但max_len最长也只能512，因为有个Position embedding，预训练预设最长512。