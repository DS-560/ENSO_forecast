from options.reforecast_options import Reforecast_options
import numpy as np
from create_dataset import *
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
import datetime

def get_date_idx(dates):
    result=[]
    for i in dates:
        date = datetime.datetime.fromtimestamp(i/1e9)
        month = date.month
        if month<10:
            month = '0'+str(month)
        else:
            month = str(month)
        result.append(str(date.year)+'-'+month+'-01')
    return result
    

if __name__ == '__main__':
    opt=Reforecast_options().parse()
    opt.startdate = strTodate(opt.startdate)
    opt.enddate = strTodate(opt.enddate)
    opt.test_start = strTodate(opt.test_start)
    opt.test_end = strTodate(opt.test_end)
    
    if opt.model=='cnn':
        test_predictors, test_predictands = assemble_predictors_predictands(opt)
        test_dataset = ENSODataset(test_predictors, test_predictands)
        testloader = DataLoader(test_dataset, batch_size=opt.batch_size)
        idx = get_date_idx(test_predictands.index.values.tolist())
        pred_CNN=cnn_predict('./checkpoints/'+opt.name,testloader)
        if opt.classification:
            pred_CNN = classify(pred_CNN,threshold= opt.threshold)
            
            experiment_name = 'CNN Classification'
        else:
            experiment_name = 'CNN Regression'
        #corr, _ = pearsonr(test_predictands, pred_CNN)
        #rmse = mean_squared_error(test_predictands, pred_CNN) ** 0.5
        
        
        #plot_nino_time_series(test_predictands, pred_CNN, '{} Predictions. Corr: {:3f}. RMSE: {:3f}.'.format(experiment_name,
         #                                                             corr, rmse),'./results/'+fname)
        pred_dict ={}
        merge_date_nino(pred_dict,idx,pred_CNN)
    elif opt.model == 'linear_regression':
        x_train,y_train=assemble_basic_predictors_predictands(opt,train=True)
        opt.startdate=opt.test_start
        opt.enddate=opt.test_end
        opt.dataroot = opt.dataroot1
        opt.dataset = 'observations'
        x_test,y_test=assemble_basic_predictors_predictands(opt)
        idx = get_date_idx(y_test.index.values.tolist())
        pred_reg=lin_reg(x_train, y_train,x_test)
        if opt.classification:
            pred_reg = classify(pred_reg,threshold= opt.threshold)
            y_test = classify_pd(y_test,threshold= opt.threshold)
            experiment_name = 'Linear Classification'
        else:
            experiment_name = 'Linear Regression'
        #corr, _ = pearsonr(y_test, pred_reg)
        #rmse = mean_squared_error(y_test, pred_reg) ** 0.5
        ##plot_nino_time_series(y_test, pred_reg, '{} Predictions. Corr: {:3f}. RMSE: {:3f}.'.format(experiment_name,
        #                                                              corr, rmse),'./results/'+opt.name)
        pred_dict ={}
        merge_date_nino(pred_dict,idx,pred_reg)
    if '.' in opt.name:
        fname = opt.name[:opt.name.index('.')]
    else:
        fname = opt.name

    ## get reforecast
    opt.variable_name = opt.variable_name_ref
    refore_data = read_reforecast(opt)
    reforecast = []
    model_pred = []
    time_series = []
    for i in refore_data:
        try:
            mod = pred_dict[i]
            time_series.append(np.datetime64(i))
            reforecast.append(refore_data[i])
            model_pred.append(mod)
        except:
            continue
    if opt.classification:
        reforecast = classify(reforecast,threshold= opt.threshold)
    reforecast = pd.Series(reforecast, index=pd.to_datetime(time_series))
    corr, _ = pearsonr(model_pred, reforecast)
    rmse = mean_squared_error(model_pred, reforecast) ** 0.5
    #plot_nino_time_series(reforecast,model_pred, '{} Predictions vs. Reforecast. Corr: {:3f}. RMSE: {:3f}.'.format(experiment_name,
    #                                                                  corr, rmse),'./results/'+fname+'_Reforecast', label='Reforecast')
    if opt.classification:
        acc = accuracy(reforecast, model_pred)
        plot_nino_time_series(reforecast, model_pred, '{} Predictions vs. Reforecast. Acc: {:.3} Corr: {:.3}. RMSE: {:.4}.'.format(experiment_name,
                                                                      acc, corr, rmse),'./results/'+fname+'_Reforecast', label='Reforecast')
    else:
        plot_nino_time_series(reforecast, model_pred, '{} Predictions vs. Reforecast. Corr: {:.3}. RMSE: {:.4}.'.format(experiment_name,
                                                                      corr, rmse),'./results/'+fname+'_Reforecast', label='Reforecast')
    
