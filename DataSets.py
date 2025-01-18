import os
import random
import numpy
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DataSets_forGAN(Dataset): 
    def __init__(self, DataDir, ImageSize):
        self.ImageSize = ImageSize
        self.DataDir = DataDir
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.DirX = os.path.join(DataDir, "X")
        self. DirY = os.path.join(DataDir, "Y")

        filesX = os.listdir(self.DirX)
        filesY = os.listdir(self.DirY)
        self.DataPathListX = []
        self.DataPathListY = []
        self.DataX = []
        self.DataY = []

        self.ImageR_in = torch.zeros(self.ImageSize, self.ImageSize).to(self.device)
        self.ImageG_in = torch.zeros(self.ImageSize, self.ImageSize).to(self.device)
        self.ImageB_in = torch.zeros(self.ImageSize, self.ImageSize).to(self.device)
        # self.ImageR_out = torch.zeros(1).to("cuda")
        # self.ImageG_out = torch.zeros(1).to("cuda")
        # self.ImageB_out = torch.zeros(1).to("cuda")

        for i in range(ImageSize):
            for j in range(ImageSize):
                rnd1 = random.random()
                rnd2 = random.random()
                rnd3 = random.random()
                if rnd1 > 0.5:
                    self.ImageR_in[i][j] = 1
                
                if rnd2 > 0.5:
                    self.ImageG_in[i][j] = 1
                
                if rnd3 > 0.5:
                    self.ImageB_in[i][j] = 1

        self.rng = numpy.random.default_rng()

        #process X
        for file in tqdm(filesX, desc = "Loading Image_X"):
            DataPath = os.path.join(self.DirX, file)
            self.DataX.append(torch.load(DataPath).to(self.device))
            self.DataPathListX.append(DataPath)
        # process Y

        for file in tqdm(filesY, desc = "Loading Image_Y"):
            DataPath = os.path.join(self.DirY, file)
            self.DataY.append(torch.load(DataPath).to(self.device))
            self.DataPathListY.append(DataPath)

    def __getitem__(self, index):
        tensorX = self.DataX[index]
        tensorY = self.DataY[index]

        offsetR = self.rng.uniform(-0.5, 0.5)
        offsetG = self.rng.uniform(-0.5, 0.5)
        offsetB = self.rng.uniform(-0.5, 0.5)

        #オフセットを一旦なくす
        tensorX[0] += offsetR*self.ImageR_in
        tensorX[1] += offsetG*self.ImageG_in
        tensorX[2] += offsetB*self.ImageB_in 


        return tensorX, tensorY

    def __len__(self):
        return len(self.DataPathListX)

class MyDataSets_Grayscale(Dataset): 
    def __init__(self, DataDir, ImageSize):
        self.ImageSize = ImageSize
        self.DataDir = DataDir
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.DirX = os.path.join(DataDir, "X")
        self. DirY = os.path.join(DataDir, "Y")

        filesX = os.listdir(self.DirX)
        filesY = os.listdir(self.DirY)
        self.DataPathListX = []
        self.DataPathListY = []
        self.DataX = []
        self.DataY = []

        #process X
        for file in tqdm(filesX, desc = "Loading Image_X"):
            DataPath = os.path.join(self.DirX, file)
            self.DataX.append(torch.load(DataPath).to(self.device))
            self.DataPathListX.append(DataPath)
        # process Y

        for file in tqdm(filesY, desc = "Loading Image_Y"):
            DataPath = os.path.join(self.DirY, file)
            self.DataY.append(torch.load(DataPath).to(self.device))
            self.DataPathListY.append(DataPath)

    def __getitem__(self, index):
        tensorX = self.DataX[index]
        tensorY = self.DataY[index]

        std_factor = 0.05  # データ範囲に対して5%の標準偏差
        data_std = tensorX.std() 
        noise = torch.normal(mean=0, std=std_factor * data_std, size=tensorX.shape).to(self.device)
        tensorX += noise

        return tensorX, tensorY

    def __len__(self):
        return len(self.DataPathListX)


class DataSets_forMLP(Dataset):
    def __init__(self,DataDir, feature_num, output_size):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.feature_num = feature_num
        self.output_size = output_size
        self.DataDir = DataDir
        self.dirX = os.path.join(DataDir, "X") #directory
        self.dirY = os.path.join(DataDir, "Y")
        filesX = os.listdir(self.dirX)
        filesY = os.listdir(self.dirY)
        self.dataPathListX = []
        self.dataPathListY = []
        self.dataX = []
        self.dataY = []

        #offset用の軸操作？
        self.dirX_in = torch.zeros(self.feature_num)
        self.dirX_out = torch.zeros(self.output_size)
            
        #to cuda
        self.dirX_in = self.dirX_in.to(self.device)
        self.dirX_out = self.dirX_out.to(self.device)

        #process X
        for file in tqdm(filesX, desc = "Loading Feature_X"):
            dataPath = os.path.join(self.dirX, file)
            self.dataX.append(torch.load(dataPath).to(self.device))
            self.dataPathListX.append(dataPath)

        # process Y
        for file in tqdm(filesY, desc = "Loading Feature_Y"):
            dataPath = os.path.join(self.dirY, file)
            self.dataY.append(torch.load(dataPath).to(self.device))
            self.dataPathListY.append(dataPath)
        
        # 入力データのスケール調整（標準化）
        self.dataX = torch.stack(self.dataX)  # リストをテンソルに変換
        self.mean = self.dataX.mean(dim=0)  # 各成分の平均
        self.std = self.dataX.std(dim=0)  # 各成分の標準偏差
        self.dataX = (self.dataX - self.mean) / self.std  # 標準化

    def __getitem__(self, index):
        tensorX = self.dataX[index]
        tensorY = self.dataY[index]

        std_factor = 0.05  # データ範囲に対して5%の標準偏差
        data_std = tensorX.std() 
        noise = torch.normal(mean=0, std=std_factor * data_std, size=tensorX.shape).to(self.device)
        tensorX += noise

        return tensorX, tensorY

    def __len__(self):
        return len(self.dataPathListX)
    

class ImageDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        image = Image.open(file_path).convert('L')
        transform = transforms.Compose([                    #変換の定義
            transforms.RandomCrop(454, 680),    # 画像のリサイズ
            transforms.PILToTensor(),  # 正規化なしで 0～255 のままテンソル形式へ変換
        ])
        return transform(image), os.path.basename(file_path)

class DataSetsforAnalysis(Dataset): 
    def __init__(self, DataDir, ImageSize):
        self.ImageSize = ImageSize
        self.DataDir = DataDir
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device("cpu")

        
        #self. DirY = os.path.join(DataDir, "Y")

        filesX = os.listdir(DataDir)
        #filesY = os.listdir(self.DirY)
        self.DataPathListX = []
        #self.DataPathListY = []
        self.DataX = []
        #self.DataY = []

        self.ImageR_in = torch.zeros(self.ImageSize, self.ImageSize).to(self.device)
        self.ImageG_in = torch.zeros(self.ImageSize, self.ImageSize).to(self.device)
        self.ImageB_in = torch.zeros(self.ImageSize, self.ImageSize).to(self.device)
        # self.ImageR_out = torch.zeros(1).to("cuda")
        # self.ImageG_out = torch.zeros(1).to("cuda")
        # self.ImageB_out = torch.zeros(1).to("cuda")

        for i in range(ImageSize):
            for j in range(ImageSize):
                rnd1 = random.random()
                rnd2 = random.random()
                rnd3 = random.random()
                if rnd1 > 0.5:
                    self.ImageR_in[i][j] = 1
                
                if rnd2 > 0.5:
                    self.ImageG_in[i][j] = 1
                
                if rnd3 > 0.5:
                    self.ImageB_in[i][j] = 1

        self.rng = numpy.random.default_rng()

        #process X
        for file in tqdm(filesX, desc = "Loading X"):
            DataPath = os.path.join(DataDir, file)
            self.DataX.append(torch.load(DataPath).to(self.device))
            self.DataPathListX.append(DataPath)
        # process Y

        # for file in tqdm.tqdm(filesY):
        #     DataPath = os.path.join(self.DirY, file)
        #     self.DataY.append(torch.load(DataPath).to("cuda"))
        #     self.DataPathListY.append(DataPath)

    def __getitem__(self, index):
        tensorX = self.DataX[index]
        #tensorY = self.DataY[index]

        offsetR = self.rng.uniform(-0.5, 0.5)
        offsetG = self.rng.uniform(-0.5, 0.5)
        offsetB = self.rng.uniform(-0.5, 0.5)

        #オフセットを一旦なくす
        # tensorX[0] += offsetR*self.ImageR_in
        # tensorX[1] += offsetG*self.ImageG_in
        # tensorX[2] += offsetB*self.ImageB_in 


        return tensorX#, tensorY

    def __len__(self):
        return len(self.DataPathListX)