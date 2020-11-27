from options.train_options import Training_options
import numpy as np
import importlib
from create_dataset import *
from matplotlib import pyplot as plt
import sklearn
import sklearn.ensemble
import scipy.stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
import scipy.stats
from sklearn.model_selection import train_test_split 
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

if __name__=='__main__':
    opt=Training_options().parse()
    if opt.model=='cnn':
        from models import CNN
        net=CNN(opt)
        train_predictors,train_predictands=assemble_predictors_predictands(opt,train=True)
        train_dataset = ENSODataset(train_predictors, train_predictands)
        trainloader = DataLoader(train_dataset, batch_size=opt.batch_size)
        optimizer = optim.Adam(net.parameters(), lr=opt.lr)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        net = net.to(device)
        best_loss = np.infty
        train_losses = []
        net.train()
        criterion= nn.MSELoss()
        for epoch in range(opt.epoch):
            running_loss = 0.0
            for i,data in enumerate(trainloader):
                batch_predictors, batch_predictands = data
                batch_predictands = batch_predictands.to(device)
                batch_predictors = batch_predictors.to(device)
                optimizer.zero_grad()
                predictions = net(batch_predictors).squeeze()
                loss = criterion(predictions, batch_predictands.squeeze())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('Training Set: Epoch {:02d}. loss: {:3f}'.format( epoch+1, \
                                            running_loss/len(trainloader)))
        torch.save(net,'./checkpoints/{}.pt'.format(opt.name))
            
