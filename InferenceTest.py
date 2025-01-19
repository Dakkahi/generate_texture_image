import torch
import torch.nn as nn
import IPython
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import csv
sys.path.append("./src")
import DataSets
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchvision.utils as vutils


class InferenceTest:
    def __init__(self, gen_model_path, ImageSize):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gen_model = torch.load(gen_model_path).to(self.device)
        self.gen_model.eval()
        self.dis_model.eval()
        self.ImageSize = ImageSize
        if os.path.isdir("Images") is False:
            os.makedirs("Images")

    def begin_inference(self, num_images, z_dim, filename):
        noise = torch.randn(num_images, z_dim, 1, 1).to(self.device)  # ランダムノイズ
        with torch.no_grad():
            generated_images = self.gen_model(noise)
        
        # 画像の出力（-1 ～ 1 の範囲を 0 ～ 1 に変換）
        generated_images = (generated_images + 1) / 2.0  # 正規化を解除
        for i, img in enumerate(generated_images):
            file_path = os.path.join("Images", f"{filename}_{i + 1}.png")
            vutils.save_image(img, file_path, normalize=True)
            print(f"Saved: {file_path}")
        
        return generated_images


    def show_image(self, generated_images):
        grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title("Generated Images")
        plt.show()