import torch
import torch.utils.data as data
from SUIM_NET_OCR import SUIM, SUIM_OCR
from VGG16 import VGG16,VGG16_OCR
from DataLoader import DatasetSegmentation, transformSeg
import sys
import numpy as np
from Metrix import IOU, HarmonicMean, F_Score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def EarlyStopping(model,valid_loss,PATH):
    if(valid_loss < EarlyStopping.best_loss): 
        print("Loss has reduced so saving the model")
        torch.save(model.state_dict(), PATH)
        EarlyStopping.best_loss = valid_loss
    else:
        EarlyStopping.count += 1

    if(EarlyStopping.count > 5):
        EarlyStopping.stop = True






def Step(model, batch, opt, loss_fun):

    img, label = batch
    print(type(img))
    preds = model(img)
    loss = loss_fun(preds,label)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss

def train(model,epochs,opt,train_dl,val_dl,PATH,loss_fun):

    for epoch in range(epochs):

        model.train()
        for batch in train_dl:
            batch = batch[0], batch[1]
            loss = Step(model,batch, opt, loss_fun)
            print("Current ",epoch,loss)

        print("Saving model")
        torch.save(model.state_dict(), PATH)

def TwoStageTrain(model_name,epochs1,epochs2,batch,PATH):

    print("Stage1:")
    Model = eval(model_name)().to(device)
    opt = torch.optim.SGD(Model.parameters(), lr= 0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_model(Model,batch,epochs1,opt,loss_fn,PATH + model_name + ".pt")

    print("Stage2:")
    Model = eval(model_name + '_OCR')(PATH + model_name + ".pt").to(device)
    opt = torch.optim.SGD(Model.paramters)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_model(Model,batch,epochs1,opt,loss_fn,PATH + model_name + "_OCR.pt")

    print("Train complete")


def train_model(model, batch, n_epochs, opt, loss_fn, PATH):


    EarlyStopping.stop = False
    EarlyStopping.count = 0
    EarlyStopping.best_loss = 20
    Dataset = DatasetSegmentation("./data/train/SUIMDATA/train_val","images","masks",transform=transformSeg)
    trainDataset , valDataset= torch.utils.data.random_split(Dataset,[1500,25],generator=torch.Generator().manual_seed(42))
    train_dl = torch.utils.data.DataLoader(trainDataset,batch_size = batch,shuffle = True)
    val_dl = torch.utils.data.DataLoader(valDataset,batch_size = batch,shuffle = True)
    
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 
    IOUs = []
    Precision = []
    Recall = []
    F_Scores = []
    
    # initialize the early_stopping object
    
    for epoch in range(1, n_epochs + 1):

        model.train() # prep model for training
        for batch, (data, target) in enumerate(train_dl, 1):
            # clear the gradients of all optimized variables
            opt.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = loss_fn(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            opt.step()
            # record training loss
            train_losses.append(loss.item())

        model.eval() # prep model for evaluation
        for data, target in val_dl:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = loss_fn(output, target)
            iou = IOU(target,output)
            p,r = F_Score(output, target)
            f_score = HarmonicMean(p,r)

            # record validation loss
            valid_losses.append(loss.item())
            IOUs.append(iou)
            Precision.append(p)
            Recall.append(r)
            F_Scores.append(f_score)

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        mIOU = torch.mean(torch.stack(IOUs),dim=0)
        mPrecision = torch.mean(torch.stack(Precision),dim=0)
        mRecall = torch.mean(torch.stack(Recall),dim=0)
        mF_score = torch.mean(torch.stack(F_Scores),dim=0)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        print("mIOU: ",mIOU)
        print("mPrecision: ",mPrecision)
        print("mRecall: ",mRecall)
        print("mF_score: ",mF_score)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        EarlyStopping(valid_loss, model,PATH)
        
        if EarlyStopping.stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses

if __name__ == "__main__":
    TwoStageTrain(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])
