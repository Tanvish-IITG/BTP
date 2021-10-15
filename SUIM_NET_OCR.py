import torch, torchvision, torchsummary
import torch.nn as nn
import torch.nn.functional as F
from OCR import OCR_Module

import torch
import torchvision.transforms as transforms
import glob
import os
from torch.utils import data
from cv2 import imread, IMREAD_GRAYSCALE
import matplotlib.pyplot as plt


import os
import logging

logging.basicConfig(filename="./log.txt",level=logging.INFO,format='%(asctime)s:%(level)s:%(message)s')



class RSB(nn.Module):
    def __init__(self,ch,kernel,skip = True,stride = 1):
      super().__init__()
      ch0,ch1,ch2,ch3 = ch
      self.skip = skip
      self.conv1 = nn.Conv2d(ch0,ch1,(1,1),stride = stride)
      self.normalize1 = nn.BatchNorm2d(ch1)

      self.conv2 = nn.Conv2d(ch1,ch2,kernel_size=kernel,padding='same')
      self.normalize2 = nn.BatchNorm2d(ch2)

      self.conv3 = nn.Conv2d(ch2,ch3,kernel_size=kernel,padding='same')
      self.normalize3 = nn.BatchNorm2d(ch3)

      self.conv4 = nn.Conv2d(ch0,ch3,kernel_size=(1,1),stride = stride)
      self.normalize4 = nn.BatchNorm2d(ch3)


    def forward(self,input_image):
      x = self.conv1(input_image)
      x = self.normalize1(x)
      x = F.relu(x)

      x = self.conv2(x)
      x = self.normalize2(x)
      x = F.relu(x)

      x = self.conv3(x)
      x = self.normalize3(x)
      
      if(self.skip):
        shortcut = input_image

      else:
        shortcut = self.conv4(input_image)
        shortcut = self.normalize4(shortcut)
      
      output = F.relu(x + shortcut)
      return output

class RSB_Encoder(nn.Module):
    def __init__(self,input_channel):
      super().__init__()
      self.conv1 = nn.Conv2d(input_channel,64,(5,5),stride=1,padding='same')
      self.norm1 = nn.BatchNorm2d(64)
      self.maxpool1 = nn.MaxPool2d((2,2), stride = 2)

      self.rsb1 = RSB((64,128,128,128),(3,3),skip=False,stride=2)
      self.rsb2 = RSB((128,128,128,128),(3,3))
      self.rsb3 = RSB((128,128,128,128),(3,3))

      

      self.rsb4 = RSB((128,256,256,256),(3,3),skip=False,stride=2)
      self.rsb5 = RSB((256,256,256,256),(3,3))
      self.rsb6 = RSB((256,256,256,256),(3,3))
      self.rsb7 = RSB((256,256,256,256),(3,3))



    def forward(self,input_img):
      x = self.conv1(input_img)
      enc1 = x
      x = self.norm1(x)
      x = F.relu(x)
      x = self.maxpool1(x)
      x = self.rsb1(x)
      x = self.rsb2(x)
      x = self.rsb3(x)
      enc2 = x
      x = self.rsb4(x)
      x = self.rsb5(x)
      x = self.rsb6(x)
      x = self.rsb7(x)
      enc3 = x
      return [enc1,enc2,enc3]

class RSB_Decoder(nn.Module):
    def __init__(self,output_channel):
        super().__init__()
        self.conv1   = nn.Conv2d(256,256,(3,3), padding = 'same')
        self.bnorm1  = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256,256,(2,2),stride = 2, padding = 0)

        self.conv2   = nn.Conv2d(256 + 128,256,(3,3), padding = 'same')
        self.bnorm2  = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256,128,(2,2),stride = 2, padding = 0)

        self.conv3   = nn.Conv2d(128,128,(3,3), padding = 'same')
        self.bnorm3  = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128,128,(2,2),stride = 2, padding = 0)

        self.conv4   = nn.Conv2d(128 + 64,64,(3,3), padding = 'same')
        self.bnorm4  = nn.BatchNorm2d(64)

        self.conv5   = nn.Conv2d(64,output_channel,(3,3), padding = 'same')
        self.bnorm5  = nn.BatchNorm2d(output_channel)

    def forward(self,input_image):
        enc1, enc2, x = input_image
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.deconv1(x)
        x = torch.cat((x,enc2),1)

        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.deconv2(x)

        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.deconv3(x)

        x = torch.cat((x,enc1),1)

        x = self.conv4(x)
        x = self.bnorm4(x)
        x = self.conv5(x)
        x = self.bnorm5(x)

        return x





class SUIM(nn.Module):
      def __init__(self):
            super().__init__()
            self.rsb_encoder = RSB_Encoder(3)
            self.rsb_decoder = RSB_Decoder(8)
            
      def forward(self,input_image):
            x = self.rsb_encoder(input_image)
            x = self.rsb_decoder(x)
            return x

      def Step(self, batch, opt):

            img, label = batch
            print(type(img))
            preds = self(img)
            loss = F.mse_loss(preds,label)
            loss.backward()
            opt.step()
            opt.zero_grad()
            return loss

      def fit(self, train_dl,opt,PATH, epochs = 10):

          for epoch in range(epochs):

              self.train()
              for batch in train_dl:
                  batch = batch[0], batch[1]
                  loss = self.Step(batch, opt)
                  print(epoch,loss)
              torch.save(model.state_dict(), PATH)



class SUIM_OCR(torch.nn.Module):
      def __init__(self,path=None):
            super().__init__()
            self.backbone = SUIM()
            self.ocr = OCR_Module(64,8,8,8)
            if(path):
                self.backbone.load_state_dict(path)
                for param in self.backbone.parameters():
                    param.requires_grad = False



      def forward(self, img):
            e = self.backbone(img)
            x = self.ocr(e)
            return x


if __name__ == "__main__":
    model = SUIM_OCR()
    t = torch.zeros((1,3,256,256))
    ans = model(t)
    torchsummary(model,(1,3,256,256))

    # trainDataset = DatasetSegmentation(file_path,"images","masks",transformSeg)


    # model = RSB_Network()
    # model = model.cuda()
    # opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    # traindl = data.DataLoader(trainDataset,batch_size = 25,shuffle = True)
    # model.fit(traindl,opt,10)


    # PATH = "./First_Train"
    # torch.save(model.state_dict(), PATH)
