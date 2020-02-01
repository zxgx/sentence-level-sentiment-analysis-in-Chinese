import os
import numpy as np
import pandas as pd

def hotel_split(path, dir_name='data', seed=None):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    train_pth = os.path.join(dir_name, 'htl_train.csv')
    val_pth = os.path.join(dir_name, 'htl_val.csv')
    test_pth = os.path.join(dir_name, 'htl_test.csv')
    
    if os.path.exists(train_pth) or os.path.exists(val_pth) or os.path.exists(test_pth):
        print("data splits have already existed in", dir_name)
        print()
        return
    
    # for the reproducibility of data splitting
    if not seed is None:
        np.random.seed(seed)
    
    # load the original dataset
    src = pd.read_csv(path)

    # get dataset balanced
    pos, neg = src[src.label==1][:2400], src[src.label==0][:2400]
    df = pd.concat([pos, neg])
    
    # shuffle the balanced dataset
    idx = np.arange(df.shape[0])
    np.random.shuffle(idx)
    df = df.iloc[idx, :]
    
    # split dataset into train/val/test splits
    train, val, test = df[:3500], df[3500:4000], df[4000:]
    train.to_csv(train_pth, index=False)
    val.to_csv(val_pth, index=False)
    test.to_csv(test_pth, index=False)

    print('='*20, 'Data splitting has been done!', '='*20)
    print()

if __name__ == '__main__':
    hotel_split('data/ChnSentiCorp_htl_all.txt', seed=731)
