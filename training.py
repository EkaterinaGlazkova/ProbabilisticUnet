import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
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
num_classes = 25

train_batch_size = 8
test_batch_size = 4
latent_space_size = 3

id_to_train_id = train_utils.get_id_to_train_id()

class TransformedCityDataset(datasets.Cityscapes):
    def __getitem__(self, i):
        input, target = super(TransformedCityDataset, self).__getitem__(i)
        
        angle_arg = np.random.randint(-30, 30)
        translate_arg = tuple(np.random.randint(-10, 10, 2))
        scale_arg = np.random.random()*0.3 + 0.7
        shear_arg = np.random.randint(-30, 30)
        
        input_transforms = transforms.Compose([
            transforms.Lambda(lambda x: transforms.functional.affine(x, angle = angle_arg, 
                                                                     translate = translate_arg, 
                                                                     scale = scale_arg, 
                                                                     shear = shear_arg, 
                                                                     resample=PIL.Image.BILINEAR)),
            transforms.Resize((256, 512), interpolation = PIL.Image.BILINEAR),
            transforms.ColorJitter(),
            transforms.ToTensor()
        ])
        
        target_transforms = transforms.Compose([
            transforms.Lambda(lambda x: transforms.functional.affine(x, angle = angle_arg, 
                                                                     translate = translate_arg, 
                                                                     scale = scale_arg, 
                                                                     shear = shear_arg, 
                                                                     resample=PIL.Image.NEAREST)),
            transforms.Resize((256, 512), interpolation = PIL.Image.NEAREST),
            transforms.Lambda(lambda x: id_to_train_id[x]),
            transforms.ToTensor(),
            train_utils.get_segmentation_variant
        ])
        
        input = input_transforms(input)
        target = target_transforms(target)

        return input, target
    
def training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = "/home/glazkova/ProbabilisticUnet/data"
    
    img_transform_func = transforms.Compose([
                        transforms.Resize((256, 512), interpolation = PIL.Image.BILINEAR),
                        transforms.ToTensor(),

                    ])

    labels_transform_func = transforms.Compose([
                        transforms.Resize((256, 512), interpolation = PIL.Image.NEAREST),
                        transforms.Lambda(lambda x: id_to_train_id[x]),
                        transforms.ToTensor()
                    ])


    train_dataset = TransformedCityDataset(root=data_root, 
                               mode="fine",
                               split="train",
                               target_type="semantic")

    test_dataset = datasets.Cityscapes(root=data_root, 
                               mode="fine",
                               split="val",
                               target_type="semantic",
                               transform = img_transform_func,
                               target_transform = labels_transform_func)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch_size)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=test_batch_size)

    #iter_num = 240000
    #n_epochs = iter_num // (len(train_dataset) // batch_size)
    n_epochs = 100

    model = ProbUNet(num_classes, latent_space_size)
    #model.load_state_dict(torch.load("results/model"))
    model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(opt, step_size=5, gamma=0.9)
    train_utils.train(model, opt, scheduler, n_epochs, train_loader, test_loader, save_path = "results/final_3D/")

if __name__ == "__main__":
    training()