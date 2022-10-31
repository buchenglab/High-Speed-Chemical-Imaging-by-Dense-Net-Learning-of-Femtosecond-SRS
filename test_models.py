#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:15:24 2020
@author: jingzhang
"""
import numpy as np
#from model_BN import unet
from deep_chem import deep_chem, precision, recall, crossent_recall
from tensorflow.keras.models import load_model, model_from_json
import matplotlib.pyplot as plt
from supp_func import get_pr_re,get_pr_re_all, psudo_clr, merged,one_hot_label, one_hot_label_porb, stitch_win, global_norm_noise 
from supp_func import im_adj, fuse_img, write_metrics
import os 
import glob
from time import gmtime, strftime, localtime
import imageio
import scipy.io
import re
from limit_gpu_memory import limit_gpu_memory
limit_gpu_memory(0.25, allow_growth=True)
from sklearn.model_selection import train_test_split
from skimage import measure 
from supp_func import psudo_clr, get_pr_re, global_norm_noise,morph_open_cls,\
 merged,one_hot_label, one_hot_label_porb, stitch_win, im_adj, make_save_path,\
 save_model_log, model_test, save_metrics, reformat_metrics, model_test_aug,\
 one_hot_2_2d, get_conf_mat
from get_splited_dataset_all import get_splited_dataset
#%% 
model_path = '/home/'
model_lipid_path = model_path+'lipid.hdf5'
model_ER_path = model_path+'ER.hdf5'
model_nuclei_path = model_path+'nuclei.hdf5'
model_cytop_path = model_path+'cytop.hdf5'
model_path_all = [model_lipid_path,model_ER_path,model_nuclei_path,model_cytop_path]
model_all = []
dependencies = {'precision': precision, 'recall':recall,'custom loss':crossent_recall(0.2)}
for classno in range(4):
    if classno == 0:    
#        model_all.append(load_model(model_path_all[classno],custom_objects=dependencies))
        model_all.append(load_model(model_path_all[classno],compile = False))
    else:
        model_all.append(load_model(model_path_all[classno]))    
model_input_size = [128,128]
#%% load testset
mat_dir = '/home/'
avg_mat_ = scipy.io.loadmat(mat_dir + 'xxx.mat')
label_mat_ = scipy.io.loadmat(mat_dir + 'xxxx.mat')
avg_mat_ = avg_mat_['weighted_avg_mat_']
avg_mat_ = np.transpose(avg_mat_,[2,0,1])
label_mat_ = label_mat_['label_spec_spat_mat_']
label_mat_ = np.transpose(label_mat_,[3,2,0,1])
#%%
save_path = '/home/'
save_test_path = make_save_path(save_path)## make new path with current time
save_model_log(save_test_path,model_path_all)
pred_metrics, pred_mat_mean,pred_mat_std, ori_mat, label_gt_mat, conf_mat, ssim_mat, mssim_mat\
 = [], [], [], [], [], [], [], []
for ii in range(100):
    srs_temp = avg_mat_[ii,:,:]
    srs_temp = global_norm_noise(srs_temp,mean_new=0.5,std_new=0.125,noise_level = 0)
    pred_label_mat_mean, pred_label_mat_std = [], []
    for classno in range(4):#classno
        pred_mean, res_x, res_y = model_test_aug(srs_temp,model_input_size,20,model_all[classno])
        pred_label_mat_mean.append(pred_mean)
#        pred_label_mat_std.append(pred_std)
    pred_label_mat = one_hot_label_porb(np.transpose(np.asarray(pred_label_mat_mean),[1,2,0]))
    srs_temp_adj = im_adj(srs_temp)#[0:-1 * res_x, 0:-1 * res_y])
    pred_label_clr = psudo_clr(pred_label_mat,4)             
    merge_clr = merged(srs_temp_adj, pred_label_clr,0.3)
    pred_mat_mean.append(pred_label_mat)
#    pred_mat_std.append(np.asarray(pred_label_mat_std))
    ori_mat.append(srs_temp)#[0:-1 * res_x, 0:-1 * res_y])
    pred_metrics.append(get_pr_re_all(label_mat_[ii,:,:,:],pred_label_mat))
    label_gt_mat.append(label_mat_[ii,:,:,:])
    conf_mat.append(get_conf_mat(one_hot_2_2d(label_mat_[ii,:,:,:]),one_hot_2_2d(pred_label_mat)))
    ssim_mat.append(measure.compare_ssim(one_hot_2_2d(label_mat_[ii,:,:,:])/4,one_hot_2_2d(pred_label_mat)/4))
    mssim_mat.append(measure.compare_ssim(np.transpose(label_mat_[ii,:,:,:],[1,2,0]),pred_label_mat,multichannel=True))
    imageio.imwrite(save_test_path +'/' + str(ii) + '.png',(merge_clr))
    imageio.imwrite(save_test_path +'/' + str(ii) + '_pred.png',(pred_label_clr))
save_metrics(save_test_path,pred_metrics, conf_mat, ssim_mat)    ##pred_mat_std = pred_mat_std,
np.savez(save_test_path + 'result_arr.npz',pred_mat_mean = pred_mat_mean, \
         ori_mat = ori_mat,label_mat = label_gt_mat)    
rrr, precision_mat,recall_mat, ssim_mmm = reformat_metrics(pred_metrics, conf_mat, ssim_mat)#%% get quick result
fuse_img(save_test_path +'/',20)


    
    
    
    
    
    
    