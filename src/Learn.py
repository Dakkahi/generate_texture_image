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
    def __init__(self, ImageSize, z_dim, batch_size, modelSaveSpan, use_existing_folder, loadEpoch_gen, loadEpoch_dis):
        #諸パラメータの設定
        #self.predictionFutureFrame = predictionFutureFrame
        self.ImageSize = ImageSize
        self.z_dim = z_dim
        self.batch_size = batch_size
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
        self.loadEpoch_gen = loadEpoch_gen
        self.loadEpoch_dis = loadEpoch_dis
        self.writerPath = 'runs/' + ("Aug_" if self.useAugmentation else "") + ("CF_" if self.useCustomLossFunction else "") + "_" + self.d
        print("tensorboard --logdir=" + self.writerPath)
        #解析用のやつ
        self.writer = SummaryWriter(self.writerPath)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ModelFolderPath_gen = f"{self.ModelFolderPath}/gen"
        self.ModelFolderPath_dis= f"{self.ModelFolderPath}/dis"

    def present_time(self):
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        d = now.strftime('%Y%m%d%H%M%S')   
        return d
    

    # 要修正
    def data_loader_GAN(self, FolderPath, DataSetsClass):
        #データローダの生成
        if DataSetsClass == DataSets.DataSets_forGAN:
            self.dataloader = DataLoader(DataSets.DataSets_forGAN(data_dir = FolderPath, ImageSize = self.ImageSize), batch_size=self.batch_size, shuffle=True)
            X = self.dataloader.dataset.__getitem__(0)
            print(X.size())
            return X
        else:
            print('the class imported from DataSets.py can not be compatible to this loader.')



    def network_GAN(self, gen_model, dis_model, lr_gen, lr_dis, betas, weight_decay):
        # ネットワークの設定
        ImageSize = self.ImageSize
        self.gen_model = gen_model.to(self.device)
        self.dis_model = dis_model.to(self.device)
        if self.loadEpoch_gen != 0:
            self.gen_model = torch.load(os.path.join(f"{self.ModelFolderPath}/gen" ,str(self.loadEpoch_gen) + "_gen.pth"))
        if self.loadEpoch_dis != 0:
            self.dis_model = torch.load(os.path.join(f"{self.ModelFolderPath}/dis" ,str(self.loadEpoch_dis) + "_dis.pth"))

        # モデルのグラフをTensorBoardに追加
        example_input_gen = torch.randn(self.batch_size, self.z_dim, 1, 1).to(self.device)  # ダミーの入力データ
        self.writer.add_graph(self.gen_model, example_input_gen)

        example_input_dis = torch.randn(self.batch_size, 1, self.ImageSize, self.ImageSize).to(self.device)  # ダミーの入力データ
        self.writer.add_graph(self.gen_model, example_input_gen)
        
        # 損失関数と最適化手法の定義
        criterion = nn.BCELoss()
        optimizer_gen = optim.Adam(self.gen_model.parameters(), lr = lr_gen, betas = betas, weight_decay = weight_decay)
        optimizer_dis = optim.Adam(self.dis_model.parameters(), lr = lr_dis, betas = betas, weight_decay = weight_decay)

        if(os.path.isdir(self.ModelFolderPath) is False):
            os.makedirs(self.ModelFolderPath)

        return criterion, optimizer_gen, optimizer_dis

    def learn(self, criterion, optimizer_gen, optimizer_dis, n_epochs):
        # 学習の実行
        print(f"device : {self.device}")
        real_label = 1.0
        fake_label = 0.0
        for epoch in tqdm.tqdm(range(n_epochs)):
            #discriminatorの学習
            TrainDisLossSum = 0
            TrainGenLossSum = 0
            self.gen_model.train()
            self.dis_model.train()
            for x_train in self.dataloader:
                x_train = x_train.to(self.device)

                # 偽物データの損失
                noise = torch.randn(self.batch_size, self.z_dim, 1, 1).to(self.device)  # ランダムノイズ
                fake_images = self.gen_model(noise)
                fake_output = self.dis_model(fake_images.detach())  # Generatorを更新しないようdetach
                # fake_loss = criterion(fake_output, torch.full_like(fake_output, fake_label))

                # ======== Discriminatorの学習 強くなりすぎないように2回に1回だけ学習========
                # if epoch % 2 == 0:
                y_real = self.dis_model(x_train)
                # 本物データの損失
                # real_loss = criterion(y_real, torch.full_like(y_real, real_label))

                
                # 全体の損失
                # loss_dis = real_loss + fake_loss
                loss_dis = torch.mean(F.relu(1.0 - y_real)) + torch.mean(F.relu(1.0 + fake_output))
                optimizer_dis.zero_grad()
                loss_dis.backward()
                optimizer_dis.step()
                TrainDisLossSum += loss_dis.item()



                # ======== Generatorの学習 ========
                y_fake = self.dis_model(fake_images)
                # loss_gen = criterion(y_fake, torch.full_like(y_fake, real_label))
                loss_gen = -torch.mean(y_fake)
                optimizer_gen.zero_grad()
                loss_gen.backward()
                optimizer_gen.step()
                TrainGenLossSum += loss_gen.item()


            self.writer.add_scalar('training discriminator loss',
                            TrainDisLossSum/len(self.dataloader),
                                epoch)
            
            self.writer.add_scalar('training generator loss',
                            TrainGenLossSum/len(self.dataloader),
                                epoch)
            

            # 検証
            # 一旦は検証は行わない
            # if (epoch + 1) % self.modelSaveSpan == 0:
            #     #print("Eval with EvalLoader:")
            #     self.gen_model.eval()
            #     self.dis_model.eval()
            #     eValDisLossSum = 0
            #     EvalGenLossSum = 0
            #     with torch.no_grad():  # 検証中は勾配計算を無効化
            #         for x_eval in self.evalLoader_Image:  # 検証用データローダ
            #             x_eval = x_eval.to(self.device)

            #             # ======== Discriminatorの検証 ========
            #             y_real = self.dis_model(x_eval)
            #             # 本物データの損失
            #             real_loss = criterion(y_real, torch.full_like(y_real, real_label))

            #             # 偽物データの損失
            #             noise = torch.randn(self.batch_size, self.z_dim, 1, 1).to(self.device)  # ランダムノイズ
            #             fake_images = self.gen_model(noise)
            #             fake_output = self.dis_model(fake_images)
            #             fake_loss = criterion(fake_output, torch.full_like(fake_output, fake_label))

            #             # 全体の損失
            #             loss_dis = real_loss + fake_loss
            #             EvalDisLossSum += loss_dis.item()

            #             # ======== Generatorの検証 ========
            #             y_fake = self.dis_model(fake_images)
            #             loss_gen = criterion(y_fake, torch.full_like(y_fake, real_label))
            #             EvalGenLossSum += loss_gen.item()

            #     # 平均損失をTensorBoardに記録
            #     self.writer.add_scalar('validation discriminator loss',
            #                     EvalDisLossSum / len(self.evalLoader_Image),
            #                     epoch)

            #     self.writer.add_scalar('validation generator loss',
            #                     EvalGenLossSum / len(self.evalLoader_Image),
            #                     epoch)

            if (epoch + 1) % self.modelSaveSpan == 0:
                torch.save(self.gen_model, f"{self.ModelFolderPath_gen}/{epoch+1}_gen.pth")
                torch.save(self.dis_model, f"{self.ModelFolderPath_dis}/{epoch+1}_dis.pth")
                
        self.writer.close()
    