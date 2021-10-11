import torch
from SUIM_NET_OCR import SUIM, SUIM_OCR
from VGG16 import VGG16,VGG16_OCR
from DataLoader import DatasetSegmentation, transformSeg




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

def TwoStageTrain(model_name,epochs1,epochs2,PATH):
    train_dl = DatasetSegmentation("./data/train/SUIMDATA/train_val","images","masks",transform=transformSeg)
    print("Stage1:")
    Model = eval(model_name)()
    opt = torch.optim.SGD(Model.paramters)
    loss_fn = torch.nn.CrossEntropyLoss()
    train(Model,epochs1,opt,train_dl,PATH + ".pt",loss_fn)

    print("Stage2:")
    Model = eval(model_name + '_OCR')(PATH + ".pt")
    opt = torch.optim.SGD(Model.paramters)
    loss_fn = torch.nn.CrossEntropyLoss()
    train(Model,epochs1,opt,train_dl,PATH + "_OCR.pt",loss_fn)

    print("Train complete")
