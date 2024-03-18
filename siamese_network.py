import os
import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class SiameseNetworkDataset(Dataset):

    def __init__(self, data_dir, data_indices='train', transform=None, should_invert=True):
        self.data_dir = data_dir
        self.indices = np.loadtxt(data_dir + '/' + data_indices + '_idx.txt').astype(int)
        self.labels = np.loadtxt(data_dir + '/labels').astype(int)
        self.labels = 1 - self.labels  # note that the original labels (1 = duplicate) need to be inverted for the
        # contrastive loss function to work (1 = dissimilar)
        self.transform = transform
        self.should_invert = should_invert

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_pair_path = os.path.join(self.data_dir, 'features', str(self.indices[idx] + 1))
        img_1 = Image.open(img_pair_path + '/img_1.png')
        img_2 = Image.open(img_pair_path + '/img_2.png')
        img_1 = img_1.convert("L")
        img_2 = img_2.convert("L")

        label = self.labels[self.indices[idx]]

        if self.should_invert:
            img_1 = PIL.ImageOps.invert(img_1)
            img_2 = PIL.ImageOps.invert(img_2)

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2, torch.from_numpy(np.array([label], dtype=np.float32))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 50),
            nn.ReLU(inplace=True),

            nn.Linear(50, 20))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
