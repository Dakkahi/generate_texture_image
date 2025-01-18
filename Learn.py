import os
import sys
sys.path.append("./src")
import DataSets
#from Models import FiveLayerNet_dropout
import Models
import datetime
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import glob
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.datasets import ImageFolder


class Learn:
    def __init__(self, ImageSize, feature_num, OutputSize, batch_size, modelSaveSpan, use_existing_folder, loadEpoch):
        #諸パラメータの設定
        #self.predictionFutureFrame = predictionFutureFrame
        self.ImageSize = ImageSize
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.OutputSize = OutputSize
        self.modelSaveSpan = modelSaveSpan
        self.useAugmentation = True
        self.useCustomLossFunction = True
        self.d = self.present_time()
        if use_existing_folder is None:
            self.ModelFolderPath = "Models/Models" + "_"+ str(self.d)
            os.makedirs(f"{self.ModelFolderPath}/gen")
            os.makedirs(f"{self.ModelFolderPath}/dis")
        else:
            self.ModelFolderPath = use_existing_folder
        self.loadEpoch = loadEpoch
        self.writerPath = 'runs/' + ("Aug_" if self.useAugmentation else "") + ("CF_" if self.useCustomLossFunction else "") + "_" + self.d
        print("tensorboard --logdir=" + self.writerPath)
        #解析用のやつ
        self.writer = SummaryWriter(self.writerPath)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def present_time(self):
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        d = now.strftime('%Y%m%d%H%M%S')   
        return d
    

    # 要修正
    def data_loader_GAN(self,FolderPath, DataSetsClass):
        #データローダの生成
        if DataSetsClass == DataSets.MyDataSets_Grayscale:
            self.dataloader = DataLoader(DataSets.MyDataSets_Grayscale(DataDir = f"{FolderPath}/Train", ImageSize = self.ImageSize), batch_size=self.batch_size, shuffle=True)
            X,Y = self.trainLoader_Image.dataset.__getitem__(0)
            print(X.size(),Y.size())
            return (X, Y)
        else:
            print('the class imported from DataSets.py can not be compatible to this loader.')



    def network_GAN(self, gen_model, dis_model, z_dim, lr, weight_decay):
        # ネットワークの設定
        ImageSize = self.ImageSize
        OutputSize = self.OutputSize
        self.gen_model = gen_model.to(self.device)
        self.dis_model = dis_model.to(self.device)
        if self.loadEpoch != 0:
            self.gen_model = torch.load(os.path.join(self.ModelFolderPath,str(self.loadEpoch) + "_gen.pth"))
            self.dis_model = torch.load(os.path.join(self.ModelFolderPath,str(self.loadEpoch) + "_dis.pth"))

        # モデルのグラフをTensorBoardに追加
        example_input_gen = torch.randn(self.batch_size, z_dim, 1, 1).to(self.device)  # ダミーの入力データ
        self.writer.add_graph(self.gen_model, example_input_gen)

        example_input_dis = torch.randn(self.batch_size, 1, self.ImageSize, self.ImageSize).to(self.device)  # ダミーの入力データ
        self.writer.add_graph(self.gen_model, example_input_gen)
        
        # 損失関数と最適化手法の定義
        criterion = nn.BCELoss()
        optimizer_gen = optim.Adam(self.gen_model.parameters(), lr = lr, weight_decay=weight_decay)
        optimizer_dis = optim.Adam(self.di_model.parameters(), lr = lr, weight_decay=weight_decay)

        if(os.path.isdir(self.ModelFolderPath) is False):
            os.makedirs(self.ModelFolderPath)

        return criterion, optimizer_gen, optimizer_dis

    def learn(self, criterion, optimizer_gen, optimizer_dis, n_epochs):
        # 学習の実行
        print(f"device : {self.device}")
        for epoch in tqdm.tqdm(range(self.loadEpoch,n_epochs)):
            #discriminatorの学習
            self.gen_model.train()
            trainLossSum = 0
            for x_train, y_train in self.trainLoader_Image:
                rnd_val = random.randint(1, 4)
                x_train = TF.rotate(x_train, 90*rnd_val)
                y_pred = self.model(x_train)
                # print(y_tensor)
                # print(y_pred)
                loss = criterion(y_pred, y_train)
                trainLossSum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.writer.add_scalar('training loss',
                            trainLossSum,
                                epoch)
            if (epoch + 1) % self.modelSaveSpan == 0:
                #print("Eval with EvalLoader:")
                self.model.eval()
                lossEachPart = torch.zeros(self.OutputSize).to(self.device)
                evalLossSum = 0
                evalCount = 0
                for x_test, y_test in self.testLoader:
                    y_pred = self.model(x_test)
                    loss = criterion(y_pred, y_test)
                    # diff = torch.abs(y_pred - y_test)
                    # diffSum = torch.sum(diff,0)
                    # lossEachPart += diffSum
                    evalLossSum += loss.item()
                    evalCount += 1
                #print("Avg Eval Loss:" + str(evalLossSum/evalCount))
                modelPath = os.path.join(self.ModelFolderPath,str(epoch+1) + ".pth")
                torch.save(self.model, modelPath)
                #print("saving model at " + modelPath)
                self.writer.add_scalar('evalLoss',
                                evalLossSum,
                                    epoch)
                # for i in range(6,len(lossEachPart)):
                #     self.writer.add_scalar("Diff_" + str(i),lossEachPart[i],epoch)
        self.writer.close()
    