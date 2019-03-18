import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import PIL
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from model.model import ProbUNet
import vis_and_data_utils.visualization_utils as visualization_utils
import train_utils

from vis_and_data_utils.labels import labels


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 19

batch_size = 8
latent_space_size = 6
    
def training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = "/home/glazkova/ProbabilisticUnet/data"
    id_to_train_id = train_utils.get_id_to_train_id()


    img_transform_func = transforms.Compose([
                        transforms.Resize((256, 512), interpolation = PIL.Image.BILINEAR),
                        transforms.ToTensor(),

                    ])

    labels_transform_func = transforms.Compose([
                        transforms.Resize((256, 512), interpolation = PIL.Image.NEAREST),
                        transforms.Lambda(lambda x: id_to_train_id[x]),
                        transforms.ToTensor()
                        #transforms.Lambda(lambda x: x*255) #better way?

                    ])

    train_dataset = datasets.Cityscapes(root=data_root, 
                               mode="fine",
                               split="train",
                               target_type="semantic",
                               transform = img_transform_func,
                               target_transform = labels_transform_func)

    test_dataset = datasets.Cityscapes(root=data_root, 
                               mode="fine",
                               split="val",
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
    model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_utils.train(model, opt, n_epochs, train_loader, test_loader, batch_size, save_path = "results/model")

if __name__ == "__main__":
    training()