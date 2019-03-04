import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from model.model import ProbUNet
from model.training import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 21
batch_size = 8

transform_func = transforms.Compose([
                    transforms.Resize((256, 512)),
                    transforms.ToTensor()
                ])

train_dataset = datasets.Cityscapes(root="/home/glazkova/prob_unet/data", 
                           mode="fine",
                           split="train",
                           target_type="semantic",
                           transform = transform_func,
                           target_transform = transform_func)

test_dataset = datasets.Cityscapes(root="/home/glazkova/prob_unet/data", 
                           mode="fine",
                           split="test",
                           target_type="semantic",
                           transform = transform_func,
                           target_transform = transform_func)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size)

iter_num = 240000
n_epochs = iter_num // (len(train_dataset) // batch_size)

model = ProbUNet(21, 6)
model.to(device)
opt = torch.optim.RMSprop(model.parameters(), lr=0.00001)
train(model, opt, n_epochs)