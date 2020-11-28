import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import copy
from matplotlib import pyplot as plt
import os
from sklearn import linear_model
import pandas as pd
import xarray as xr

class CNN(nn.Module):
    def __init__(self, opt, print_feature_dimension=False):
        """
        inputs
        -------
            num_input_time_steps        (int) : the number of input time
                                                steps in the predictor
            print_feature_dimension    (bool) : whether or not to print
                                                out the dimension of the features
                                                extracted from the conv layers
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(opt.num_input_time_steps, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.print_layer = Print()
        
        #ATTENTION EXERCISE 9: print out the dimension of the extracted features from 
        #the conv layers for setting the dimension of the linear layer!
        #Using the print_layer, we find that the dimensions are 
        #(batch_size, 16, 42, 87)
        self.fc1 = nn.Linear(16 * 42 * 87, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.print_feature_dimension = print_feature_dimension

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.print_feature_dimension:
          x = self.print_layer(x)
        x = x.view(-1, 16 * 42 * 87)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Print(nn.Module):
    """
    This class prints out the size of the features
    """
    def forward(self, x):
        print(x.size())
        return x

def select_time(xr):
    return str(xr.time.values.flatten())[2:12]

def classify_pd(x,threshold=1.5):
    # x: pd.Series
    index = x.index
    result = classify(x,threshold=threshold)
    result = pd.Series(result, index=index)
    return result
    

def lin_reg(x_train,y_train,x_test):
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    predictions = regr.predict(x_test)
    return predictions

def cnn_predict(url,testloader):
    # url: link to saved model
    # x: predictant
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        net = torch.load(url)
    else:
        net = torch.load(url, map_location=torch.device('cpu'))
    net.eval()
    net.to(device)
    predictions = np.asarray([])
    for i, data in enumerate(testloader):
        batch_predictors, batch_predictands = data
        batch_predictors = batch_predictors.to(device)

        batch_predictions = net(batch_predictors).squeeze()
    #Edge case: if there is 1 item in the batch, batch_predictions becomes a float
    #not a Tensor. the if statement below converts it to a Tensor
    #so that it is compatible with np.concatenate
        if len(batch_predictions.size()) == 0:
          batch_predictions = torch.Tensor([batch_predictions])
        predictions = np.concatenate([predictions, batch_predictions.detach().cpu().numpy()])
    return predictions


def classify(x,threshold=1.5):
    threshold = abs(threshold)
    result = []
    for i in x:
        if i <= -threshold:
            result.append(-1)
        elif i < threshold:
            result.append(0)
        else:
            result.append(1)
    return np.array(result)

def plot_nino_time_series(y, predictions, title,fname, label='Ground Truth'):
    predictions = pd.Series(predictions, index=y.index)
    predictions = predictions.sort_index()
    y = y.sort_index()
    plt.plot(y, label=label)
    plt.plot(predictions, '--', label='ML Predictions')
    plt.legend(loc='best')
    plt.title(title)
    plt.ylabel('Nino3.4 Index/ONI')
    plt.xlabel('Date')
    if not os.path.isdir("./results/"):
        os.mkdir('./results')
    plt.savefig(fname)

def parse_file(url):
    # get reforecast files and reforecast years
    f=open(url)
    files,years=[],[]
    for i in f.readlines():
        i=i.split(',')
        files.append(i[0].strip())
        years.append(i[1].strip())
    return files,years

##def get_dates(xr,year,leadtime, period):
##    # xr: xarray object
##    year=str(year)
##    dates=[]
##    temp=''
##    for i in range(len(xr)):
##        time = str(xr[i].time.values.flatten())[2:12]
##        if time[:4]==year:
##            if temp!='' and temp>time:
##                break
##            dates.append(time)
##            temp=time
##        else:
##            break
##    return dates

def get_dates(xr,year,leadtime, period):
    # xr: xarray object
    cover =  leadtime * period # number of slices covered
    year=str(year)
    temp=''
    idx = -1
    for i in range(len(xr)):
        idx += 1
        time = select_time(xr[i])
        time1 = select_time(xr[i+1])
        if time[:4]==year and time==time1:
            break
    for i in range(idx+1, len(xr)):
        t1 = select_time(xr[i-1])
        t2 = select_time(xr[i])
        t3 = select_time(xr[i+1])
        if t1!=t2 and t2!=t3:
            break
    return [i,i+cover-1]

def merge_date_nino(dict_, dates,nino):
    
    for i,j in zip(dates,nino):
        dict_[i]=j
    
        

##def read_reforecast(opt):
##
##  fname = opt.reforecast_data
##  num_input_time_steps = opt.num_input_time_steps
##  ref = opt.ref
##  variable_name = opt.variable_name
##  file_leadtime = opt.file_leadtime
##  period = opt.period
##  
##  files,years=parse_file(fname)
##  dates,nino34=[],[]
##  dict_ = {}
##  for i,j in zip(files,years):
##    ds = xr.open_dataset(i)
##    date_ = get_dates(ds[variable_name],j)
##    start_date=0
##    end_date=len(date_)
##    subsetted_ds = ds[variable_name].sel(dim0=slice(start_date,
##                                                   end_date))-ref # convert to anomaly
##    num_samples = subsetted_ds.shape[0]
##    #subsetted_ds = np.stack([subsetted_ds.values[n-num_input_time_steps:n] for n in range(num_input_time_steps,
##    #                                                          num_samples+1)])
##    #subsetted_ds[subsetted_ds>1000] = float('NaN')
##    #Calculate the Nino3.4 index
##    y = subsetted_ds.sel(latitude=slice(5,-5), longitude=slice(360-170,360-120)).mean(dim=('latitude','longitude'))
##    
##    y = pd.Series(y.values).rolling(window=3).mean()[2:].values
##    y = y.astype(np.float32)
##    merge_date_nino(dict_,date_,y)
##    ds.close()
##  return  dict_


def read_reforecast(opt):

  fname = opt.reforecast_data
  num_input_time_steps = opt.num_input_time_steps
  ref = opt.ref
  variable_name = opt.variable_name
  file_leadtime = opt.file_leadtime
  period = opt.period
  lead_time = opt.leadtime
  
  files,years=parse_file(fname)
  dates,nino34=[],[]
  dict_ = {}
  for i,j in zip(files,years):
    y = []
    date_ = []
    ds = xr.open_dataset(i)
    start_date,end_date = get_dates(ds[variable_name],j,file_leadtime,period)
    index = lead_time - 1 + start_date
    for k in range(period):
      temp = ds[variable_name][index]
      date_.append(select_time(temp))
      y.append(temp.sel(latitude=slice(5,-5), longitude=slice(360-170,360-120)).mean(dim=('latitude','longitude')).values.flatten()[0]-ref)
      index += file_leadtime
    y = pd.Series(y).rolling(window=3).mean()[2:].values
    y = y.astype(np.float32)
    merge_date_nino(dict_,date_[:-2],y)
    ds.close()
  return  dict_

def accuracy(x,y):
    # x: np.array
    # y: pd.Series
    if len(x)==0 or len(y)==0:
        return 0
    x1,y1=[],[]
    for i,j in zip(x,y):
        x1.append(int(i))
        y1.append(int(j))
    y1=np.array(y1)
    x1=np.array(x1)
    match = x1 == y1
    return sum(match)/len(match)
    
