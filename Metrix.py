import torch

def IOU(y,label):
    predicted_label = torch.argmax(y,dim = 1)
    intersection = torch.stack([torch.logical_and(label == i,predicted_label == i).sum() for i in range(8)])
    union = torch.stack([torch.logical_or(label == i,predicted_label == i).sum() for i in range(8)])
    return torch.div(intersection,union)

def F_Score(y,label):
    predicted_label = torch.argmax(y,dim = 1)
    TP = torch.stack([torch.logical_and(label == i,predicted_label == i).sum() for i in range(8)])
    FP = torch.stack([torch.logical_and(label != i,predicted_label == i).sum() for i in range(8)])
    TN = torch.stack([torch.logical_and(label != i,predicted_label != i).sum() for i in range(8)])
    FN = torch.stack([torch.logical_and(label == i,predicted_label != i).sum() for i in range(8)])

    Precison = torch.div(TP,torch.add(TP , FP))
    Recall  = torch.div(TP,torch.add(TP , FN))

    return Precison, Recall

def HarmonicMean(x,y):
    return torch.div(torch.mul(2*x,y) , torch.add(x,y))

if __name__ == "__main__":
    from VGG16 import VGG16_OCR
    import DataLoader
    trainDataset = DataLoader.DatasetSegmentation("./data/train/SUIMDATA/train_val","images","masks",transform=DataLoader.transformSeg)
    model = VGG16_OCR()





