import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np

from model.model import ProbUNet
import visualization_utils

from labels import labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 19

batch_size = 2
latent_space_size = 6


def get_id_to_train_id(labels = labels):
    id_to_train_id = np.zeros((len(labels), 1), int)
    for item in labels:
        if item.id > 0:
            id_to_train_id[item.id] = item.trainId
    return id_to_train_id

def get_train_id_to_id(labels = labels):
    train_id_to_id = np.zeros((len(labels), 1), int)
    for item in labels:
        if item.trainId > 0:
            train_id_to_id[item.trainId] = item.id
    return train_id_to_id

def train_epoch(model, optimizer, batchsize=8):
    loss_log = []
    model.train()
    for _, (x_batch, y_batch) in zip(trange(len(train_loader)), train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        model_loss = model.compute_lower_bound(x_batch, y_batch)
        model_loss.backward()
        optimizer.step()
        loss = model_loss.item()
        loss_log.append(loss)
    return loss_log

def test(model, epoch_num, test_loader, res_dir = None):
    ce_log, acc_log = [], []
    model.eval()
    for batch_num, (x_batch, y_batch) in enumerate(test_loader):  
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_batch[y_batch == 255] = 0
        output = model.sample(x_batch)
        loss = nn.CrossEntropyLoss()(output.permute(0,2,3,1).contiguous().view(-1, num_classes), y_batch.view(-1))
        loss = loss.item()
        ce_log.append(loss)
        
        pred = torch.argmax(output.permute(0,2,3,1), dim = -1)
        acc = torch.eq(pred, y_batch).float().mean()
        acc_log.append(acc)
        
        
        if res_dir and batch_num == 0:
            visualization_utils.plot_batch_with_results(x_batch, 
                                                        y_batch.cpu().view(batch_size, -1, 1, 256, 512), 
                                                        pred.cpu().view(batch_size, -1, 1, 256, 512), 
                                                        outdir = res_dir + "epoch_{}.pdf".format(epoch_num))
    return ce_log, acc_log

def plot_history(train_history, ce_log, acc_log, epoch_num, save_dir = "results/"):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)
    
    points = np.array(ce_log)
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val_ce', zorder=2)
    points = np.array(acc_log)
    plt.scatter(points[:, 0], points[:, 1], marker='+', s=180, c='orange', label='val_acc', zorder=2)
        
    plt.xlabel('train steps')

    plt.legend(loc='best')
    plt.grid()
    
    plt.savefig(save_dir + str(epoch_num) + ".png")
    
    
def train(model, opt, n_epochs, save_path = None):
    train_log = []
    ce_log, acc_log = [], []

    batchsize = batch_size

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, opt, batchsize=batchsize)

        val_loss, val_acc = test(model,epoch, test_loader, res_dir = "results/")

        train_log.extend(train_loss)

        steps = train_dataset.train_labels.shape[0] / batch_size
        ce_log.append((steps * (epoch + 1), np.mean(val_loss)))
        acc_log.append((steps * (epoch + 1), np.mean(val_acc)))

        clear_output()
        plot_history(train_log, ce_log, acc_log, epoch)    
        if save_path:
            torch.save(model.state_dict(), save_path)
            
    print("Final error: {:.2%}".format(1 - acc_log[-1][1]))

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = "/home/glazkova/ProbabilisticUnet/data"

    img_transform_func = transforms.Compose([
                        transforms.Resize((256, 512), interpolation = PIL.Image.BILINEAR),
                        transforms.ToTensor(),

                    ])

    labels_transform_func = transforms.Compose([
                        transforms.Resize((256, 512), interpolation = PIL.Image.NEAREST),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x*255) #better way?

                    ])

    train_dataset = datasets.Cityscapes(root=data_root, 
                               mode="fine",
                               split="train",
                               target_type="semantic",
                               transform = img_transform_func,
                               target_transform = labels_transform_func)

    test_dataset = datasets.Cityscapes(root=data_root, 
                               mode="fine",
                               split="test",
                               target_type="semantic",
                               transform = img_transform_func,
                               target_transform = labels_transform_func)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size)

    #iter_num = 240000
    #n_epochs = iter_num // (len(train_dataset) // batch_size)
    n_epochs = 50

    model = ProbUNet(num_classes, latent_space_size)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(model, opt, n_epochs, save_path = "results/model")