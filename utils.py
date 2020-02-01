import torch

def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, data_iter, optimizer, criterion, epochs, device, val_iter, save_dir=None):
    model.to(device)
    best_acc = 0
    dataset_size = len(data_iter.dataset)
    for epoch in range(epochs):
        epoch_corrects, epoch_loss = 0, 0
        model.train()
        for it, batch in enumerate(data_iter):
            review, label = batch.review.to(device), batch.label.to(device)
            #print(review.shape, label.dtype)
            optimizer.zero_grad()
            pred = model(review)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            corrects = torch.sum(torch.argmax(pred, dim=1) == label).item()
            epoch_corrects += corrects
            epoch_loss += loss.item()
        
        val_acc, val_loss = evaluate(model, val_iter, criterion, device)
        
        print('epoch: %d, iterations: %d, train_loss: %.4f, train_acc: %.4f,\
        val_loss: %.4f, val_acc: %.4f' % (epoch+1, (epoch+1)*len(data_iter),
               epoch_loss/len(data_iter), epoch_corrects/dataset_size, val_loss, val_acc))
        
        if val_acc > best_acc:
            best_acc = val_acc
            if not save_dir is None:
                if os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = os.path.join(save_dir, 'best.pth')
                torch.save(model.state_dict(), save_path)
    print("\nThe best validation accuracy is %.4f\n" % best_acc)


def evaluate(model, data_iter, criterion, device):
    epoch_corrects, epoch_loss = 0, 0
    #model.to(device)
    model.eval()
    dataset_size = len(data_iter.dataset)
    with torch.no_grad():
        for it, batch in enumerate(data_iter):
            review, label = batch.review.to(device), batch.label.to(device)
            pred = model(review) # batch_first
            loss = criterion(pred, label) # long
            
            corrects = torch.sum((torch.argmax(pred, dim=1)) == label).item()
            epoch_corrects += corrects
            epoch_loss += loss.item()
            
    return epoch_corrects/dataset_size, epoch_loss/len(data_iter)


if __name__ == '__main__':
    '''
    完整性检查
    '''
    #import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    from data_split import hotel_split
    from data_utils import load_htl_datasets
    from models import TextCNN
    
    hotel_split('data/ChnSentiCorp_htl_all.txt')

    train_iter, val_iter, _ = load_htl_datasets()
    #train_iter.random_shuffler.random_state=random.seed(731)
    TEXT = train_iter.dataset.fields['review']
    embedding = TEXT.vocab.vectors
    vocab_size, embed_dim = embedding.shape
    model = TextCNN(vocab_size, embed_dim, embedding)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    epochs = 3

    train(model, val_iter, optimizer, criterion, epochs, device, val_iter)