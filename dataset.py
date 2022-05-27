"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
# import openslide
import os
import random


class GA_Datas(data_utils.Dataset):
    def __init__(self, data_list):

        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
        # if self.train:
        #     return len(self.train_labels_list)
        # else:
        #     return len(self.test_labels_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        # if self.train:
        #     bag = self.train_bags_list[index]
        #     label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        # else:
        #     bag = self.test_bags_list[index]
        #     label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return data