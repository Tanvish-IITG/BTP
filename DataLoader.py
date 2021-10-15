import torch
from torch._C import Argument
from torch.utils import data
import glob
import torchvision.transforms as transforms
from cv2 import imread, IMREAD_GRAYSCALE
import os

import matplotlib.pyplot as plt
import sys


def splitchannel(img,shape):
    ans = torch.zeros(6,shape[1],shape[2])
    for i in range(shape[1]):
      for j in range(shape[2]):
        pixel = img[:,i,j]
        if(pixel[0] > 0.5 and pixel[1] <= 0.5 and pixel[2] <= 0.5):
          ans[0][i][j] = 1
        elif(pixel[0] <= 0.5 and pixel[1] > 0.5 and pixel[2] <= 0.5):
          ans[1][i][j] = 1
        elif(pixel[0] <= 0.5 and pixel[1] <= 0.5 and pixel[2] > 0.5):
          ans[2][i][j] = 1
        elif(pixel[0] > 0.5 and pixel[1] > 0.5 and pixel[2] <= 0.5):
          ans[3][i][j] = 1
        elif(pixel[0] > 0.5 and pixel[1] <= 0.5 and pixel[2] > 0.5):
          ans[4][i][j] = 1
        elif(pixel[0] <= 0.5 and pixel[1] > 0.5 and pixel[2] > 0.5):
          ans[5][i][j] = 1
    
    return ans

def combinechannel(img):
    
    
    return (img[0,:,:] + 2*img[1,:,:] + 4*img[2,:,:]).unsqueeze(0)



transformSeg = torch.nn.Sequential(
                transforms.RandomCrop((256,256),padding=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                               degrees=30, translate=(0.02, 0.02), scale=(0.9, 1.1),
                               shear=30),
)

class DatasetSegmentation(data.Dataset):
    def __init__(self, folder_path, images, masks, transform = None):

        super(DatasetSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,images,'*.jpg'))
        self.mask_files = []
        self.transform = transform
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,masks,os.path.basename(img_path)[:-3] + "bmp") )

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        img =  torch.from_numpy(imread(img_path)).type(torch.float32) / 255
        label = torch.from_numpy(imread(mask_path)).type(torch.float32) / 255
        img = torch.permute(img,[2,0,1]) 
        label = torch.permute(label,[2,0,1])
        label = combinechannel(label)
        if(torch.cuda.is_available()):
            data_tensor = torch.cat([img,label],0).cuda()
        else:
            data_tensor = torch.cat([img,label],0)
        if self.transform:
            data_tensor = self.transform(data_tensor)
        return data_tensor[0:3], data_tensor[3].type(torch.long)

    def __len__(self):
        return len(self.img_files)

if __name__ == "__main__":
    train_dl = DatasetSegmentation("./data/train/SUIMDATA/train_val","images","masks",transform=transformSeg)
    index = int(sys.argv[1])
    img, label = train_dl[index]
    plt.imshow(img.permute(1,2,0))


    plt.show()