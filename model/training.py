import torch
import torch.nn as nn
from tqdm import trange

def train_epoch(model, optimizer, batchsize=16):
    loss_log = []
    model.train()
    for _, (x_batch, y_batch) in zip(trange(len(train_loader)), train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        model_loss = model.compute_lower_bound(x_batch, y_batch)
        print(model_loss)
        model_loss.backward()
        optimizer.step()
        loss = model_loss.item()
        loss_log.append(loss)
    return loss_log

def test(model):
    ce_log, acc_log = [], []
    model.eval()
    for batch_num, (x_batch, y_batch) in enumerate(test_loader):  
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        output = model.sample(x_batch).permute(0,2,3,1)
        loss = nn.CrossEntropyLoss()(output.view(-1, num_classes), target.view(-1))
        
        pred = torch.max(output, dim = -1)
        acc = torch.eq(pred, y_batch).float().mean()
        acc_log.append(acc)
        
        loss = loss.item()
        ce_log.append(loss)
    return ce_log, acc_log

def plot_history(train_history, val_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)
    
    points = np.array(val_history)
    
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')
    
    plt.legend(loc='best')
    plt.grid()

    plt.show()
    
def train(model, opt, n_epochs):
    train_log = []
    ce_log, acc_log = [], []

    batchsize = batch_size

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, opt, batchsize=batchsize)

        val_loss, val_acc = test(model)

        train_log.extend(train_loss)

        steps = train_dataset.train_labels.shape[0] / batch_size
        ce_log.append((steps * (epoch + 1), np.mean(val_loss)))
        acc_log.append((steps * (epoch + 1), np.mean(val_acc)))

        clear_output()
        plt.plot(train_log, ce_log)    
        plt.plot(ce_log, acc_log, title='accuracy')   
            
    print("Final error: {:.2%}".format(1 - acc_log[-1][1]))

