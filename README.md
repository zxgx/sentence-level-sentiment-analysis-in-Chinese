# Chinese sentence level sentiment analysis
[Chinese version README](README-zh.md)

|Model|Dev|Test|
|:---|:---:|:---:|
|[TextCNN](https://aclanthology.org/D14-1181/)                          | 92.07 / 91.46 | 93.56 / 92.95 |
|Bi-LSTM                                                                | 92.26 / 91.69 | 93.54 / 92.64 |
|[RCNN](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745) | 92.73 / 92.31 | 94.08 / 93.48 |
|[BERT](https://aclanthology.org/N19-1423/)                             | 95.28 / 94.82 | 95.50 / 95.00 |

Notes:
1. The results report the highest accuracy and the mean accuracy of 10 times running respectively.
2. The Bi-LSTM consists of 3 bidirectional LSTM layers, while the RCNN achieves the results with only 1 bidirectional layer

### Dataset
```bash
cd data
tar -zxf chnsenticorp.tgz
```

## Requirements
pip whenever there is any missing module  
Below, I document some notes about Chinese text segmentation and pretrained models

### Chinese text segmentation & Word embeddings
__Needed by: TextCNN, RNN, RCNN__

Since there are no blanks in Chinese, I need some text segmentation tool to preprocess each sentence into a sequence of words.  
Here, I adopt [jieba](https://github.com/fxsjy/jieba), which is popular among Chinese NLP communities.  

Then, I use the pretrained word embeddings [sgns.renmin.bigram-char](https://github.com/Embedding/Chinese-Word-Vectors) to convert each discrete word into a dense word vector.

### BERT pretrained with Chinese corpora
[bert-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)  

It's quite easy to implement a bert-based model with the interfaces provided by [hugging face](https://huggingface.co/).  
But actually it took me a while to read through its implementation. For the seek of trade-off between easy adoption and software engineering, the details behind those interfaces are quite complicated.  

I recommend this [handout](http://nlp.seas.harvard.edu/2018/04/03/attention.html) for breaking down a transformer into pieces and understanding this model.

### Notes about dataloader
I implement a dataloader from scratch to package data samples into batches.  

I knew that there were some helpful utility modules, such as TorchText, which could facilitate the data reading procedure.
But when I started this experiment (2020.1), the TorchText version was 0.4, and there weren't any thorough documentations or tutorials about this module.  
What's more, after investigating this module for a while, I found that it's hard to prepare data samples for BERT.


## Commands
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
Due to the limited memory of GPU, you should carefully set the max_len and batch_size.  
However, the upper limit of max_len is 512, because the position embedding in BERT can only support sentences with at most 512 tokens.