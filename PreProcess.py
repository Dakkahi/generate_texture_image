import os
import csv
import torch
import shutil
import sys
sys.path.append("./src")
import torch.utils.data
import torchvision.transforms as transforms
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from torch.utils.data import DataLoader, Dataset
import DataSets
import torchvision.utils as utils
from tqdm import tqdm
import pandas as pd
import random
import math
import numpy as np
import collections
from PIL import Image
import torch
import torchvision
import io
from torchvision.transforms import functional as F


class PreProcess:
    def __init__(self, OutputFolderPath, ImageSize):
        if os.path.isdir(OutputFolderPath) is False: #OutputFolderPathがなかったら作成する
            os.makedirs(OutputFolderPath)
        self.OutputFolderPath = OutputFolderPath
        self.ImageSize = ImageSize
    

    def PreProcess_ImageEdit_withaugmentation(self, InputFolderPath, augment_count):
        transform = transforms.Compose([                    #変換の定義
            transforms.ToTensor(),  # テンソルに変換
            transforms.Normalize(mean=[0.5], std=[0.5]),  # -1～1に正規化
            transforms.RandomRotation(degrees=(-90, 90)),  # -π/2からπ/2の範囲でランダムに回転
            transforms.RandomCrop(self.ImageSize),  # 256x256にランダムクロップ
            transforms.RandomHorizontalFlip(p=0.5),  # 1/2の確率で左右反転
            transforms.RandomVerticalFlip(p=0.5),  # 1/2の確率で上下反転
            transforms.ColorJitter(brightness=0.2, contrast=0.2)  # 明るさとコントラストの調整のみ
        ])

        for filename in tqdm(os.listdir(InputFolderPath), desc = "File Processing"):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                # 画像の読み込み
                image_path = os.path.join(InputFolderPath, filename)
                image = Image.open(image_path).convert('L')
                for file_idx in range(augment_count):
                    augmented_tensor = transform(image)
                    tensor_path = os.path.join(f"{self.OutputFolderPath}", filename.split('.')[0] + f'_{file_idx}' + '.pt')
                    torch.save(augmented_tensor, tensor_path)