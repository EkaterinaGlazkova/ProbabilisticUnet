import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class down_block(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(down_block, self).__init__()

        self.down_net = nn.Sequential(
            nn.MaxPool2d(2), # add condition for not only maxpool (avgpool?)
            conv_unit(input_ch, output_ch, kernel_size).to(device)
        )

    def forward(self, x):
        x = self.down_net(x)
        return x


class up_block(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(up_block, self).__init__()

        #self.up_proc = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners = True)
        self.up_proc = nn.ConvTranspose2d(input_ch, input_ch//2, 3, stride=2, padding=1, output_padding=1)
        self.conv = conv_unit(input_ch, output_ch, kernel_size).to(device)

    def forward(self, low_res_inp, high_res_inp):
        low_res_inp = self.up_proc(low_res_inp)
        x = torch.cat([high_res_inp, low_res_inp], dim=1)
        x = self.conv(x)
        return x


class conv_unit(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(conv_unit, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size, padding = kernel_size // 2).to(device), 
            nn.ReLU(),
            nn.Conv2d(output_ch, output_ch, kernel_size, padding = kernel_size // 2).to(device), 
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_net(x)
        return x
