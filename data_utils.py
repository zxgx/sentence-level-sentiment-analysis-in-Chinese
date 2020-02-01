import random
import jieba
import torch
import torch.nn as nn
from torchtext import data, datasets
from torchtext.vocab import Vectors

def word_cut(text):
    '''
    Chinese text segmenter.
    
    Refer to https://github.com/fxsjy/jieba for more details.
    '''
    return [word for word in jieba.cut(text) if word.strip()]
    

def load_htl_datasets(dataset_dir='data', pretrained_path='sgns.renmin.bigram-char',
                        cache_dir='caches'):
    '''
    Load the dataset splits of hotel reviews.
    
    Inputs:
        dataset_dir: The directory of the train/val/test dataset.
        pretrained_path: The path of the pretrained word vectors.
        cache_dir: The dir of the word vectors to be saved.
        
    Outputs:
        (train_iter, val_iter, test_iter): A tuple of torchtext.data.BucketIterator \
        objects, which generate batches of examples for training, validation and test.
    '''
    
    # fields
    LABEL = data.LabelField()
    TEXT = data.Field(tokenize=word_cut, batch_first=True)

    fields = [('label', LABEL), ('review', TEXT)]
    
    # load the original data splits
    train, val, test = data.TabularDataset.splits(
        path=dataset_dir, format='csv', skip_header=True, fields=fields,
        train='htl_train.csv', validation='htl_val.csv', test='htl_test.csv'        
    )
    
    # load pretrained word vectors
    vectors = Vectors(name=pretrained_path, cache=cache_dir, unk_init=nn.init.normal_)
    
    # establish vocabularies for fields
    TEXT.build_vocab(train, vectors=vectors, min_freq=2)
    LABEL.build_vocab(train, val, test)
    
    # init embeddings for pad and unk
    unk_embed = torch.mean(vectors.vectors, dim=0)
    pad_embed = torch.zeros(unk_embed.shape[0], dtype=torch.float)
    
    pidx, uidx = TEXT.vocab.stoi[TEXT.pad_token], TEXT.vocab.stoi[TEXT.unk_token]
    TEXT.vocab.vectors[pidx] = pad_embed
    TEXT.vocab.vectors[uidx] = unk_embed
    
    # create BucketIterators
    batch_size = 128
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        datasets=(train, val, test),
        batch_size=batch_size,
        sort_key = lambda x: len(x.review)
    )
    
    return train_iter, val_iter, test_iter
    

if __name__ == '__main__':
    '''
    test data iterators
    '''
    train_iter, val_iter, test_iter = load_htl_datasets() 
    # these two lines are important for the reproducibility of data.BucketIterators, 
    # but they are perhaps less useful for the performance of training,
    # maybe there exists better solutions, but python random module is intricate:).
    random.seed(731)
    train_iter.random_shuffler.random_state=random.getstate()
    for i in range(3):
        for batch in train_iter:
            print(batch.review.shape[0])
        print()
    
    '''
    test embeddings for pad and unk
    '''
    vectors = Vectors(name='sgns.renmin.bigram-char', cache='caches', unk_init=torch.normal)
    unk_embed = torch.mean(vectors.vectors, dim=0)
    pad_embed = torch.zeros_like(unk_embed)
    print(unk_embed[::20], '\n', pad_embed)
    
    TEXT = train_iter.dataset.fields['review']
    pidx, uidx = TEXT.vocab.stoi[TEXT.pad_token], TEXT.vocab.stoi[TEXT.unk_token]
    print(TEXT.vocab.vectors[uidx][::20])
    print(TEXT.vocab.vectors[pidx])