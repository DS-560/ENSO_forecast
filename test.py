from options.test_options import Test_options
from models import *
import importlib
from create_dataset import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import copy
from matplotlib import pyplot as plt
import os
from sklearn import linear_model
import pandas as pd
import numpy as np

def proper_mon(m):
    if m<10:
        mon = '0'+str(int(m))
    else:
        mon = str(int(m))
    return mon

def get_date_str(start,end,lead):
    start = start.split('-')
    end = end.split('-')
    start_y,start_m = int(start[0]), int(start[1])
    start_m1= int(start_m)
    end_y, end_m = int(end[0]), int(end[1])
    months = int((end_y - start_y)*12 + (end_m - start_m))
    dates = []
    start_m += lead
    if start_m>12:
        start_y += int(start_m/12)
        start_m = start_m%12
    
    dates.append(str(start_y)+'-'+proper_mon(start_m)+'-01')
    for i in range(start_m1+lead+1,start_m1+months+lead+1):
        start_m += 1
        if start_m>12:
            start_y += 1
            start_m -= 12
        
        dates.append(str(start_y)+'-'+proper_mon(start_m)+'-01')
    return dates

def plot_and_write(dates,pred,name,lead):
    dates1 = []
    for i in dates:
        dates1.append(np.datetime64(i))
    fname = './results/'+name+'_Nino34index_prediction_leadtime'+str(lead)+'.txt'
    f = open(fname,'w')
    for i,j in zip(dates,pred):
        f.write(str(i)+'   '+str(j)+'\n')
    f.close()
    series = pd.Series(pred, index=pd.to_datetime(dates1))
    plt.plot(series)
    plt.xlabel('Datetime')
    plt.ylabel('Nino 3.4 Index')
    plt.title('Predicted Nino3.4 Index with leadtime = '+str(lead))
    if not os.path.isdir("./results/"):
        os.mkdir('./results')
    plt.savefig('./results/'+name+'_forecast.png')
    


if __name__=='__main__':
    opt=Test_options().parse()
    opt.startdate = strTodate(opt.startdate)
    opt.enddate = strTodate(opt.enddate)
    opt.test_start = strTodate(opt.test_start)
    opt.test_end = strTodate(opt.test_end)
    dates = get_date_str(opt.test_start, opt.test_end , opt.leadtime)
    if '.' in opt.name:
        fname = opt.name[:opt.name.index('.')]
    else:
        fname = opt.name
    if opt.model=='cnn':
        dates = dates[1:]
        test_predictors, test_predictands = assemble_predictors_predictands(opt)
        test_dataset = ENSODataset(test_predictors, test_predictands)
        testloader = DataLoader(test_dataset, batch_size=opt.batch_size)
        pred_CNN=cnn_predict('./checkpoints/'+opt.name,testloader)
        
        if opt.classification:
            pred_CNN = classify(pred_CNN,threshold= opt.threshold)
            if opt.compare_ground_truth:
                test_predictands=classify_pd(test_predictands,threshold= opt.threshold)
            experiment_name = 'CNN Classification'
        else:
            experiment_name = 'CNN Regression'
        plot_and_write(dates,pred_CNN,fname,opt.leadtime)
        if opt.compare_ground_truth:
            corr, _ = pearsonr(test_predictands, pred_CNN)
            rmse = mean_squared_error(test_predictands, pred_CNN) ** 0.5
            plot_nino_time_series(test_predictands, pred_CNN, '{} Predictions. Corr: {:3f}. RMSE: {:3f}.'.format(experiment_name,
                                                                      corr, rmse),'./results/'+fname)
        
    elif opt.model == 'linear_regression':
        x_train,y_train=assemble_basic_predictors_predictands(opt,train=True)
        opt.startdate=opt.test_start
        opt.enddate=opt.test_end
        opt.variable_name = opt.variable_name_ref
        opt.dataroot = opt.dataroot1
        x_test,y_test=assemble_basic_predictors_predictands(opt)
        pred_reg=lin_reg(x_train, y_train,x_test)
        if opt.classification:
            pred_reg = classify(pred_reg,threshold= opt.threshold)
            if opt.compare_ground_truth:
                y_test = classify_pd(y_test,threshold= opt.threshold)
            experiment_name = 'Linear Classification'
        else:
            experiment_name = 'Linear Regression'
        plot_and_write(dates,pred_reg,fname,opt.leadtime)
        if opt.compare_ground_truth:
            corr, _ = pearsonr(y_test, pred_reg)
            rmse = mean_squared_error(y_test, pred_reg) ** 0.5
            plot_nino_time_series(y_test, pred_reg, '{} Predictions. Corr: {:3f}. RMSE: {:3f}.'.format(experiment_name,
                                                                      corr, rmse),'./results/'+fname)
        
        
        
