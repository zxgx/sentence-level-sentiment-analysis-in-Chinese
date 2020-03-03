
class Config(object):
    def __init__(self, model_name='bert', 
                 bert_identifier='hfl/chinese-bert-wwm-ext', 
                 bert_dir='caches/chinese_wwm_ext_pytorch',
                 dataset_dir='data/hotel', skip_header=True, 
                 tok_cache='caches/hotel_tokenizer.pt', 
                 embedding_path='caches/sgns.renmin.bigram-char',
                 batch_size=64, max_len=None, include_length=True
                ):
        # Model
        self.model_name = model_name
        if model_name == 'bert':
            self.is_bert = True
            self.bert_identifier = bert_identifier
            self.bert_dir = bert_dir
        else:
            self.is_bert = False
            
        # Tokenizer
        self.tok_cache = tok_cache
        self.embedding_path = embedding_path
        
        # Dataset
        self.dataset_dir = dataset_dir
        self.skip_header = skip_header
        
        # Iterator
        self.batch_size = batch_size
        self.max_len = max_len
        self.include_length = include_length
        

