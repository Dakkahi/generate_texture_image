import os
import random
import numpy
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DataSets_forGAN(Dataset): 
    def __init__(self, data_dir, ImageSize):
        self.ImageSize = ImageSize
        self.data_dir = data_dir
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data = []
        #process X
        for filename in tqdm(os.listdir(data_dir), desc = "data loading"):
            if filename.endswith('.pt'):
                tensor = torch.load(os.path.join(data_dir, filename))
                tensor = tensor.to(self.device)
                self.data.append(tensor)

            

    def __getitem__(self, index):
        tensorX = self.data[index]
        return tensorX

    def __len__(self):
        return len(self.data)