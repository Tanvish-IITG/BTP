import torch
from torchvision import models
from OCR import OCR_Module

class Encoder(torch.nn.Module):
      def __init__(self):
          super().__init__()
          vgg16 = models.vgg16(pretrained=True)
          f = vgg16.features
          self.B1 = f[0:5]
          self.B2 = f[5:10]
          self.B3 = f[10:17]
          self.B4 = f[17:24]
          

      def forward(self,img):
          e1 = self.B1(img)
          e2 = self.B2(e1)
          e3 = self.B3(e2)
          e4 = self.B4(e3)
          return e1,e2,e3,e4

class DecoderBlock(torch.nn.Module):
      def __init__(self,input_channel,output_channel):
          super().__init__()
          self.deconv =  torch.nn.ConvTranspose2d(input_channel,output_channel,(2,2),stride = 2, padding = 0)
          self.conv = torch.nn.Conv2d(output_channel,output_channel,(3,3),padding = 'same')
          self.batchnorm = torch.nn.BatchNorm2d(output_channel)



      def forward(self,img):
          x = self.deconv(img)
          x = self.conv(x)
          x = self.batchnorm(x)
          x = torch.nn.functional.relu(x)

          return x

class Decoder(torch.nn.Module):
      def __init__(self,input_channel,mid_channels,output_channel):
            super().__init__()
            mid1,mid2,mid3 = mid_channels
            self.block1 = DecoderBlock(input_channel,mid1)
            self.block2 = DecoderBlock(2*mid1,mid2)
            self.block3 = DecoderBlock(2*mid2,mid3)
            self.block4 = DecoderBlock(2*mid3,output_channel)

      def forward(self, e):
            e1,e2,e3,e4 = e
            x = self.block1(e4)
            x = torch.cat((x,e3),dim = 1)
            x = self.block2(x)
            x = torch.cat((x,e2),dim = 1)
            x = self.block3(x)
            x = torch.cat((x,e1),dim = 1)
            x = self.block4(x)
            return x

class VGG16(torch.nn.Module):
      def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder(512,(256,128,64),8)
            for param in self.encoder.parameters():
                    param.requires_grad = False

      def forward(self, img):
            e = self.encoder(img)
            x = self.decoder(e)
            return x

class VGG16_OCR(torch.nn.Module):
      def __init__(self,path=None):
            super().__init__()
            self.backbone = VGG16()
            self.ocr = OCR_Module(64,8,8,8)
            if(path):
                self.backbone.load_state_dict(path)
                for param in self.backbone.parameters():
                    param.requires_grad = False



      def forward(self, img):
            e = self.backbone(img)
            x = self.ocr(e)
            return x
