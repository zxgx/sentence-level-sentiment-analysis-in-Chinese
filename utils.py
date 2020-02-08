import torch
import time

def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_iter, optimizer, criterion, epochs, device, include_length, val_iter=None, save_dir=None):
    model.to(device)
    best_acc, stop_asc = 0, 0
    for epoch in range(epochs):
        epoch_corrects, epoch_loss = 0, 0
        model.train()
        st = time.time()
        for it, batch in enumerate(train_iter):
            optimizer.zero_grad()
            
            if include_length:
                (review, length), label = batch.review, batch.label
                review, length, label = review.to(device), length.to(device), label.to(device)
                pred = model(review, length)
            else:
                review, label = batch.review.to(device), batch.label.to(device)     
                pred = model(review)
            
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            
            corrects = torch.sum(torch.argmax(pred, dim=1) == label).item()
            epoch_corrects += corrects
            epoch_loss += loss.item()
            
        print('epoch: %d | train_loss: %.4f, train_acc: %.4f'%
              (epoch+1, epoch_loss/len(train_iter), 
              epoch_corrects/len(train_iter.dataset)),
              end=' | ')
        
        if not val_iter is None:
            val_acc, val_loss = evaluate(model, val_iter, criterion, device, include_length)
            print('val_loss: %.4f, val_acc: %.4f'%(val_loss, val_acc), end=' | ')
            if val_acc > best_acc:
                best_acc, stop_asc = val_acc, 0
                if not save_dir is None:
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_path = os.path.join(save_dir, 'best.pth')
                    torch.save(model.state_dict(), save_path)
            else:
                stop_asc += 1
                if stop_asc == 5:
                    break

        et = time.time()
        print('time: %.2fs'%(et-st))        
        
    return best_acc


def evaluate(model, data_iter, criterion, device, include_length):
    epoch_corrects, epoch_loss = 0, 0
    model.to(device)
    model.eval()
    dataset_size = len(data_iter.dataset)
    with torch.no_grad():
        for it, batch in enumerate(data_iter):
            if include_length:
                (review, length), label = batch.review, batch.label
                review, length, label = review.to(device), length.to(device), label.to(device)
                pred = model(review, length)
            else:
                review, label = batch.review.to(device), batch.label.to(device)
                pred = model(review)
                
            loss = criterion(pred, label)
            
            corrects = torch.sum((torch.argmax(pred, dim=1)) == label).item()
            epoch_corrects += corrects
            epoch_loss += loss.item()
            
    return epoch_corrects/dataset_size, epoch_loss/len(data_iter)