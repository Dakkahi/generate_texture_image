import os
import csv
import torch
import shutil
import sys
sys.path.append("./src")
import torch.utils.data
import torchvision.transforms as transforms
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
    

    def pt2png(self, folder_path):
        if os.path.isdir("PreProcessedImage") is False:
            os.makedirs("PreProcessedImage")

        for filepath in tqdm(os.listdir(folder_path), desc = "extension converting"):
            data = torch.load(filepath)

            # テンソルが画像データの場合を想定 (例: shape [C, H, W])
            if isinstance(data, torch.Tensor):
                # 必要ならデータを正規化または変換
                data = data.squeeze(0)  # チャンネル次元が1の場合削除 (例: [1, H, W] → [H, W])
                data = data.numpy()  # NumPy配列に変換

                # ピクセル値を0-255の範囲にスケール
                if data.max() <= 1.0:
                    data = (data * 255).astype(np.uint8)
                else:
                    data = data.astype(np.uint8)

                # PIL Imageとして保存
                img = Image.fromarray(data)
                output_file = f"{filepath.split(".")[0]}_image.png"
                img.save(f"PreProcessedImage/{output_file}")
                print(f"画像が保存されました: {output_file}")