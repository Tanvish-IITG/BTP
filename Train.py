import torch
import torch.utils.data as data
from SUIM_NET_OCR import SUIM, SUIM_OCR
from VGG16 import VGG16,VGG16_OCR
from DataLoader import DatasetSegmentation, transformSeg
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def Step(model, batch, opt, loss_fun):

    img, label = batch
    print(type(img))
    preds = model(img)
    loss = loss_fun(preds,label)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss

def train(model,epochs,opt,train_dl,PATH,loss_fun):

    for epoch in range(epochs):

        model.train()
        for batch in train_dl:
            batch = batch[0], batch[1]
            loss = Step(model,batch, opt, loss_fun)
            print("Current ",epoch,loss)

        print("Saving model")
        torch.save(model.state_dict(), PATH)

def TwoStageTrain(model_name,epochs1,epochs2,batch,PATH):
    trainDataset = DatasetSegmentation("./data/train/SUIMDATA/train_val","images","masks",transform=transformSeg)
    train_dl = data.DataLoader(trainDataset,batch_size = batch,shuffle = True)
    print("Stage1:")
    Model = eval(model_name)().to(device)
    opt = torch.optim.SGD(Model.parameters(), lr= 0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    train(Model,epochs1,opt,train_dl,PATH + model_name + ".pt",loss_fn)

    print("Stage2:")
    Model = eval(model_name + '_OCR')(PATH + model_name + ".pt").to(device)
    opt = torch.optim.SGD(Model.paramters)
    loss_fn = torch.nn.CrossEntropyLoss()
    train(Model,epochs2,opt,train_dl,PATH + model_name + "_OCR.pt",loss_fn)

    print("Train complete")

if __name__ == "__main__":
    TwoStageTrain(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])
