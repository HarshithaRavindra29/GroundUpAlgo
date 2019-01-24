# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:21:17 2019
Naive Bayes implementation
@author: Harshitha R
"""
import pandas as pd
import numpy as np

def My_Guassian_naive_bayes(x_train,y_train,x_test,y_test):
    y_train.columns = ['class']
    y_test.columns = ['class']
    
    def calc_gaussian(tr_col,test_x):
      col = np.sort(tr_col)
      col_mean = np.nanmean(col)
      col_var = np.nanvar(col)
    #  col_std = np.std(col)
      Px = []
      for l in range(len(test_x)):
          #print(test_x.iloc[l])
          Px.append((1.0/np.sqrt(2*np.pi*col_var))*np.exp(-((test_x.iloc[l] - col_mean)**2)/(2.0*col_var)))
      return Px
    
    Prediction_classes = y_train['class'].unique()
    class_likely, Prior = {},{}
    for j in Prediction_classes:
        print j
        x_train_class = x_train[y_train['class']==j]
        ind_likely = {}
        for k in x_train_class.columns.values:
            ind_likely[k]=calc_gaussian(x_train_class[k],x_test[k])
        ind_likely_df = pd.DataFrame(ind_likely)
        class_likely[j]= ind_likely_df.prod(axis=1)
        Prior[j]=x_train_class.shape[0]*1.0/x_train.shape[0]
        
    class_likely_df = pd.DataFrame(class_likely)
    Prior_df = pd.DataFrame(Prior,index=[0])
    
    num = {}
    for m in class_likely_df.columns:
        num[m] = class_likely_df[m]*Prior_df.iloc[0,m]
    num_df = pd.DataFrame(num)    
    num_df['y_pred']=num_df.idxmax(axis=1)
     
    
    resul_dit = {'y_code_pred':num_df.y_pred,'y_test':y_test.reset_index()['class']}
    result = pd.DataFrame(resul_dit)
    
    result['truth_code'] = np.where((result['y_test']==1) & (result['y_code_pred']==1),"TP",
                               np.where((result['y_test']==1) & (result['y_code_pred']==0),"FN",
                                 np.where((result['y_test']==0) & (result['y_code_pred']==1),"FP",'TN')))
        
    TT_code = result.groupby(['truth_code']).size()
    TT_code = TT_code.reset_index()
    Accuracy = (float(TT_code[TT_code.truth_code=='TP'][0])+float(TT_code[TT_code.truth_code=='TN'][0]))/len(x_test)
    return(result,Accuracy)
