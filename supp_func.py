#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:58:19 2019
@author: jingzhang
"""
import numpy as np
from scipy.signal import butter, lfilter, freqz
from skimage.morphology import binary_closing, binary_opening
import matplotlib.pyplot as plt
from skimage import measure
import glob
import math
import imageio
import re
from sklearn import metrics
import os 
from time import gmtime, strftime, localtime
from sklearn.metrics import confusion_matrix
from scipy.ndimage.morphology import binary_dilation
def label_dila(label_arr): #label_arr, #samples x W x H
    label_arr_dila_ = []
    for ii in range(np.size(label_arr,0)):
        label_arr_dila_.append(binary_dilation(label_arr[ii,:,:],))
    return np.asarray(label_arr_dila_) #samples x W x H

def one_hot_2_2d(label_mat): 
    if np.argmin(np.shape(label_mat)) == 0:## make sure labe_mat in W x H x class 
        label_mat = np.transpose(label_mat,[1,2,0])
    size_x, size_y = np.size(label_mat,0),np.size(label_mat,1)
    label_mat_2d = np.zeros([size_x * size_y,1])
    for ii in range(np.size(label_mat,2)):
        label_mat_temp = np.squeeze(label_mat[:,:,ii]).reshape(size_x*size_y,1)
        locs = np.where(label_mat_temp==1)
        label_mat_2d[locs[0]] = ii + 1
    return np.reshape(label_mat_2d,[size_x,size_y])
def get_conf_mat(label_mat,pred_mat):
    ## label_mat and pred_mat all in shape of W x H x class
    size_x, size_y = np.size(label_mat,0),np.size(label_mat,1)
    ccc = confusion_matrix(label_mat.reshape([size_x*size_y,1]),pred_mat.reshape([size_x*size_y,1]))
    return ccc    
def make_save_path(save_path):
    save_test_dir = save_path + strftime("%Y-%m-%d %H:%M:%S",localtime()) + '/'
    if not os.path.exists(save_test_dir):
        os.mkdir(save_test_dir)
        os.mkdir(save_test_dir + 'ori_gt/')
    return save_test_dir
def save_model_log(save_path,model_path_all):
    log_txt_path = save_path + 'log.txt'
    log_txt_file = open(log_txt_path,'w')
    for classno in range(4):
        log_txt_file.write(model_path_all[classno] + '\n')
    log_txt_file.close()
def model_test_aug(srs_temp,model_input_size,aug_no,model = None):
    ## aug_no is augmentation number
    ## test time dropout... seems no effect ??? check codes later
#    pred_mat_aug = []
#    for kkk in range(aug_no):
#        result_stitch_, res_x, res_y = model_test(srs_temp,model_input_size,model = model)
#        pred_mat_aug.append(result_stitch_)
#    pred_mat_aug_mean = np.mean(pred_mat_aug,axis = 0)
#    pred_mat_aug_std = np.std(pred_mat_aug,axis = 0)
#    return pred_mat_aug_mean, pred_mat_aug_std, res_x, res_y
    ## test time augmentation 
    pred_mat_aug = []
    for kkk in range(5):
        ## augmentation
        if kkk == 0: srs_temp_aug = srs_temp
        if kkk == 1: srs_temp_aug = np.flip(srs_temp,0)
        if kkk == 2: srs_temp_aug = np.flip(srs_temp,1)
        if kkk == 3: srs_temp_aug = np.rot90(srs_temp,1)
        if kkk == 4: srs_temp_aug = np.rot90(srs_temp,2)
        result_stitch_, res_x, res_y = model_test(srs_temp_aug,model_input_size,model = model)
        ## transfer back
        if kkk == 1: result_stitch_ = np.flip(result_stitch_,0)
        if kkk == 2: result_stitch_ = np.flip(result_stitch_,1)
        if kkk == 3: result_stitch_ = np.rot90(np.rot90(result_stitch_,1),2)
        if kkk == 4: result_stitch_ = np.rot90(result_stitch_,2)
        pred_mat_aug.append(result_stitch_)
    pred_mat_aug_mean = np.mean(pred_mat_aug,axis = 0)
    return pred_mat_aug_mean, res_x, res_y
def model_test(srs_temp,model_input_size,model = None):
    ## model == None means no model as input, just return 0's array
    crop_h,crop_w = model_input_size[0],model_input_size[1]#128, 128
    crop_h_, crop_w_ = int(crop_h/2), int(crop_w/2)
    #size_x,size_y = label_mat_all.shape[0],label_mat_all.shape[1]
    size_x, size_y = np.size(srs_temp,0), np.size(srs_temp,1)
    noimg_x, noimg_y = int(np.floor(size_x/crop_h)),int(np.floor(size_y/crop_w))
    res_x, res_y = size_x - noimg_x * crop_h, size_y - noimg_y * crop_w
    ## only use the [0:-1*res_x,0:-1*res_y] part !!!
    if model == None:
        result_stitch_ = np.zeros((noimg_x * crop_h, noimg_y * crop_w))
    else:
        stitch_temp = []
        for temp_ii in range(noimg_x*2 - 1):
            for temp_jj in range(noimg_x*2 - 1):
                patch_temp = srs_temp[temp_ii*crop_h_:(temp_ii+2)*crop_h_,temp_jj*crop_w_:(temp_jj+2)*crop_w_]
                pred_temp_ = model.predict(patch_temp.reshape([1,crop_h,crop_w,1]))
                stitch_temp.append(pred_temp_.reshape([crop_h,crop_w]))
        result_stitch = stitch_win(np.asarray(stitch_temp).reshape(noimg_x*2-1,noimg_y*2-1,crop_h,crop_w),noimg_x,noimg_y,crop_h_,crop_w_)
        result_stitch_ = np.asarray(result_stitch).reshape(noimg_x*2,noimg_y*2, crop_h_, crop_w_).swapaxes(1, 2).reshape(size_x - res_x, size_y - res_y)
    return result_stitch_, res_x, res_y

def save_metrics(save_dir,pred_metrics,conf_mat, ssim_mat):
    np.save(save_dir + '/' + 'four_model_metrics.npy',np.asarray(pred_metrics))
    test_metrics_txt_path = save_dir + '/' + 'four_model_metrics_.txt'
#    with open(test_metrics_txt_path,'w') as filehandle:   
#        for index, item in enumerate(pred_metrics):
#            filehandle.write("%s" % index + "   " + str(item).strip('[]') + "\n")
    write_metrics(test_metrics_txt_path, pred_metrics)
    np.save(save_dir + '/' + 'conf_mat.npy',np.asarray(conf_mat))
    np.save(save_dir + '/' + 'ssim_mat.npy',np.asarray(ssim_mat))

def reformat_metrics(pred_metrics, conf_mat, ssim_mat):
    pred_metrics_ = np.asarray(pred_metrics)  
    pred_metrics_exclude = np.delete(pred_metrics_,[11,12,13,14,16,17],axis = 0)
    pred_metrics_exclude_mean = np.mean(pred_metrics_exclude,axis=0).reshape([1,32])
    pred_metrics_mean = np.mean(pred_metrics_,axis=0).reshape([1,32])
    rrr = np.concatenate((pred_metrics_mean.reshape(4,8),pred_metrics_exclude_mean.reshape(4,8)),axis=0)    
    
    conf_mat = np.asarray(conf_mat)
    conf_mat_mean = np.mean(conf_mat,axis = 0)
    conf_mat_mean_norm_p = conf_mat_mean/conf_mat_mean.sum(axis=0)## axis = 1: recall, axis =0: precision
    conf_mat_exclude = np.delete(conf_mat,[11,12,13,14,16,17],axis = 0)
    conf_mat_exclude_mean = np.mean(conf_mat_exclude,axis=0).reshape([5,5])
    conf_mat_exclude_mean_norm_p = conf_mat_exclude_mean/conf_mat_exclude_mean.sum(axis=0)
    precision_mat = np.concatenate((conf_mat_mean_norm_p,conf_mat_exclude_mean_norm_p),axis = 0)
    conf_mat_mean_norm_r = conf_mat_mean/conf_mat_mean.sum(axis=1)
    conf_mat_exclude_mean_norm_r = conf_mat_exclude_mean/conf_mat_exclude_mean.sum(axis=1)
    recall_mat = np.concatenate((conf_mat_mean_norm_r,conf_mat_exclude_mean_norm_r),axis = 0)     
    
    ssim_mat_mean = np.mean(np.asarray(ssim_mat),axis = 0)
    ssim_mat_exclude = np.delete(ssim_mat,[11,12,13,14,16,17],axis = 0)
    ssim_mat_exclude_mean = np.mean(np.asarray(ssim_mat_exclude),axis = 0)
    ssim_mmm = np.asarray([ssim_mat_mean, ssim_mat_exclude_mean])
    return rrr, precision_mat,recall_mat, ssim_mmm
        
def write_metrics(test_metrics_txt_path, pred_metrics):
    with open(test_metrics_txt_path,'w') as filehandle:   
        for index, item in enumerate(pred_metrics):
            filehandle.write("%s" % index + "   " ) 
            for index_, item_ in enumerate(item):#str(item_).strip('[]')
                filehandle.write(np.array2string(item_,max_line_width=100,formatter={'float_kind':lambda x: "%.3f" % x}).strip('[]') + '  ')
            filehandle.write("\n")  
def plot_std(img_dir,pred_std_temp):
    ## img_dir is the specific name to the current image, not only the directory 
    plt.figure(22,figsize=(6,6),dpi=100)
    class_name = ['lipid','ER','nuclei','cytoplasm']
    for ii in range(4):
        plt.subplot(2,2,ii + 1)
        plt.imshow(pred_std_temp[:,:,ii])
        plt.title(class_name[ii] + ' std')
    plt.savefig(img_dir)
#file_dir = '/home/jingzhang/hyperspec/model_4class/densenet_0126/denseunet-tmi-2020-01-28 17:06:08-/testset/ER/'
def fuse_img(img_dir,nstep,total_num,mode = 'snake'):
    temp_list = (glob.glob(img_dir + '*_pred.png'))
    temp_list = sorted(temp_list, key = lambda name: int(re.findall(r'\d+',name)[-1]))#name[153:-4]
#    stitch_all = []
    for ii in range(total_num):#len(temp_list)):
        temp_name = temp_list[ii]
        temp_img = plt.imread(temp_name)
        if (ii+1)%nstep == 1:
            stitch = temp_img
        else:
            if math.floor((ii+1-0.1)/nstep)%2 == 0:
                stitch = np.concatenate((stitch,temp_img),axis = 1)
            else:
                stitch = np.concatenate((temp_img,stitch),axis = 1)
        if (ii+1)%nstep == 0:
            if ii+1 == nstep:## stitch_all is empty
                stitch_all = stitch
            else:
                stitch_all = np.concatenate((stitch_all,stitch),axis = 0)
    if len(temp_list) != 0:
        imageio.imwrite(img_dir +'/' + '_stitch_all.png',(stitch_all))
#    return 
def morph_open_cls(label_img,n_clst):
    struct = np.ones((3,3))
    for ii in range(np.size(label_img,0)):
        label_temp = label_img[ii,:,:]
        label_temp_mor_ = binary_opening(label_temp)#, selem = struct)
        label_temp_mor__ = binary_closing(label_temp_mor_)
        label_temp_mor__ = label_temp_mor__[:,:,np.newaxis]#, selem = struct)
        if ii == 0:
            label_mor = label_temp_mor__#[:,:,np.newaxis]
        else:
            label_mor = np.append(label_mor,label_temp_mor__,axis = 2)            
    return label_mor
def im_adj(img_temp):
    cmin = np.percentile(img_temp,0.3)
    cmax = np.percentile(img_temp,99.7)
    if (cmax - cmin) > 0.05*cmax:
        img_adj = (img_temp - cmin)/(cmax - cmin)
    else:
        img_adj = img_temp
    return img_adj
        

def global_norm_noise(img,mean_new=0.5,std_new=0.125,noise_level = 3/255):
    if len(np.shape(img)) > 2:
        img_new = []
        for ii in range(np.shape(img)[0]):
            img_temp = img[ii,:,:]
#            img_temp_new = (img_temp-np.mean(img_temp))/np.std(img_temp)*std_new + mean_new + np.random.normal(0,noise_level,np.shape(img_temp))
            img_temp_new = (img[ii,:,:]-np.mean(img[ii,:,:]))/np.std(img[ii,:,:])*std_new + mean_new #+ np.random.normal(0,noise_level,np.shape(img_temp))
            img_new.append(img_temp_new)
    else:        
        img_mean = np.mean(img)
        img_std = np.std(img)
        img_new = ((img-img_mean)/img_std*std_new + mean_new)
    return np.asarray(img_new)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data,axis = 0)
    return y

def stitch_win(pred_arr,noimg_x, noimg_y,crop_h_, crop_w_):
    stitched = []
    for ii in range(noimg_x*2):
        for jj in range(noimg_y*2):
            if ii==0:
                if jj == 0 :
                    img_temp = pred_arr[ii,jj,0:crop_h_,0:crop_w_]
                elif jj == noimg_y*2 - 1:
                    img_temp = pred_arr[ii,jj-1,0:crop_h_,crop_w_:crop_w_*2]
                else:
                    img_temp = 0.5*pred_arr[ii,jj-1,0:crop_h_,crop_w_:crop_w_*2] + 0.5*pred_arr[ii,jj,0:crop_h_,0:crop_w_]
            elif ii== noimg_x*2 - 1:
                if jj == 0:
                    img_temp = pred_arr[ii-1,jj,crop_h_:crop_h_*2,0:crop_w_]
                elif jj == noimg_y*2 -1:
                    img_temp = pred_arr[ii-1,jj-1,crop_h_:crop_h_*2,crop_w_:crop_w_*2]
                else:
                    img_temp = 0.5*pred_arr[ii-1,jj-1,crop_h_:crop_h_*2,crop_w_:crop_w_*2] + 0.5*pred_arr[ii-1,jj,crop_h_:crop_h_*2,0:crop_w_]
            elif (jj == 0) & (ii != 0) & (ii != noimg_x*2 - 1):
                img_temp = 0.5*pred_arr[ii-1,jj,crop_h_:crop_h_*2,0:crop_w_] + 0.5*pred_arr[ii,jj,0:crop_h_,0:crop_w_]
            elif (jj == noimg_y*2 - 1) & (ii != 0) & (ii != noimg_x*2 - 1):
                img_temp = 0.5*pred_arr[ii-1,jj-1,crop_h_:crop_h_*2,crop_w_:crop_w_*2] + 0.5*pred_arr[ii,jj-1,0:crop_h_,crop_w_:crop_w_*2]
            else:
                img_temp = 0.25*pred_arr[ii-1,jj-1,crop_h_:crop_h_*2,crop_w_:crop_w_*2] + 0.25*pred_arr[ii-1,jj,crop_h_:crop_h_*2,0:crop_w_]\
                +0.25*pred_arr[ii,jj-1,0:crop_h_,crop_w_:crop_w_*2] + 0.25*pred_arr[ii,jj,0:crop_h_,0:crop_w_]
            stitched.append(img_temp)
#            print([ii,jj])
#            print(np.shape(stitched))
    return stitched
                
# result_lipid_stitch = stitch_win(result_lipid_,noimg_x, noimg_y,crop_h_, crop_w_)                
# noimg_x, noimg_y = 6,6
# crop_h_, crop_w_ = 32,32                 
                
def one_hot_label_porb(label_mat):
    locs = np.argmax(label_mat, axis = 2)
    locs_locs = np.where(np.max(label_mat, axis = 2)>0.5)
    locs_val= np.zeros((np.shape(label_mat)[0],np.shape(label_mat)[1]))
    locs_val[locs_locs[0],locs_locs[1]] = 1
    locs_val = np.concatenate((locs_val[:,:,np.newaxis],locs_val[:,:,np.newaxis],locs_val[:,:,np.newaxis],locs_val[:,:,np.newaxis]),axis = 2)
    label_temp = np.zeros(np.shape(label_mat))
    for ii in range(np.shape(label_mat)[2]):
        locs_temp = np.where(locs == ii)
        label_temp[locs_temp[0],locs_temp[1],ii] = 1
    label_one_hot = np.multiply(label_temp,locs_val)
    return label_one_hot

def one_hot_label(label_mat):
    label_lipid = label_mat[:,:,0]
    label_ER = label_mat[:,:,1]
    label_nuclei = label_mat[:,:,2]
    label_cytop = label_mat[:,:,3]
    
    label_ER_ = np.multiply((1 - label_lipid),label_ER)
    label_nuclei_ = np.multiply((1 - label_lipid),label_nuclei)
    label_nuclei_ = np.multiply((1 - label_ER),label_nuclei_)
    label_cytop_ = np.multiply((1 - label_lipid),label_cytop)
    label_cytop_ = np.multiply((1 - label_ER),label_cytop_)
    label_cytop_ = np.multiply((1 - label_nuclei),label_cytop_)
    label_one_hot = np.concatenate((label_lipid[:,:,np.newaxis],label_ER_[:,:,np.newaxis],label_nuclei_[:,:,np.newaxis],label_cytop_[:,:,np.newaxis]),axis = 2)
    return label_one_hot 

def merged(ori_img,label_clr,alpha):
    
    ori_img_3 = np.concatenate((ori_img[:,:,np.newaxis],ori_img[:,:,np.newaxis],ori_img[:,:,np.newaxis]),axis = 2)#mlib.repmat(ori_img,1,3)
    merge_clr = ori_img_3*(1-alpha) + label_clr*alpha
    return merge_clr

def merged_old(ori_img,pred_label,n_clr):
#    color_code = np.asarray([[1,0,0],[0.5,0,0.5],[0,1,0],[0,0,1]])
    color_code = np.asarray([[1,0,0],[0,1,0],[1,1,0],[0,0,1]])
#    ori_img = ori_img[8：384，8：384]
    ori_img = ori_img.reshape(ori_img.shape[0]*ori_img.shape[1])
    img_color = np.zeros((ori_img.shape[0],3))
    for kk in range(n_clr):
        img_temp = pred_label[:,:,kk].reshape(pred_label.shape[0]*pred_label.shape[1],1)
        locs = np.where(img_temp == 1)
        locs_temp = np.zeros((pred_label.shape[0]*pred_label.shape[1],1))
        locs_temp[locs] = 1
        img_color = img_color + np.matmul(np.multiply(img_temp,locs_temp),color_code[kk,:].reshape(1,3))
    img_color = img_color/n_clr
    img_color = np.reshape(img_color,(384,384,3))
    return img_color#.reshape((384,384,3))
def get_pr_re_all(GT4, pred_):
    if np.argmin(np.shape(pred_)) == 2:
        pred_ = np.transpose(pred_, [2, 0, 1])
    if np.argmin(np.shape(GT4)) == 2:
        GT4 = np.transpose(GT4, [2, 0, 1])
    pr_re_ = []
    for ii in range(np.min(np.shape(pred_))):
        pr_re_temp = get_pr_re(GT4[ii,:,:],pred_[ii,:,:])
        pr_re_.append(pr_re_temp)
    return pr_re_


def get_pr_re(label_temp,bi_result_temp):
    #% label_temp is GT, bi_reslt_temp is prediction
    true_pos_map = np.multiply(label_temp,bi_result_temp)
    true_pos = np.size(np.where(true_pos_map==1)[0])
    false_neg_map = np.multiply(label_temp,1-bi_result_temp)
    false_neg = np.size(np.where(false_neg_map==1)[0])
    act_pos = np.size(np.where(label_temp[:,:]==1)[0])
    if act_pos == 0 :
            act_pos = -0.1
    pred_pos = np.size(np.where(bi_result_temp[:,:])[0])
    
    if pred_pos == 0 :
        pred_pos = -0.1
    precision = true_pos/pred_pos 
    recall = true_pos/act_pos    
    if true_pos == 0:
        f1 = -0.1
    else:
        f1 = 2*precision*recall/(precision + recall)    
    jac = true_pos/(pred_pos + false_neg)
    ssim_temp = measure.compare_ssim(label_temp,bi_result_temp)
    mae_temp = metrics.mean_absolute_error(label_temp,bi_result_temp)
    mse_temp = metrics.mean_squared_error(label_temp,bi_result_temp)
    ## iou
    inter_area = np.sum(true_pos_map)
    union_area = np.sum(label_temp) + np.sum(bi_result_temp) - inter_area
    if union_area == 0:
        iou = -1
    else:
        iou = inter_area/union_area    
    result = np.asarray([precision,recall,f1,jac,ssim_temp,mae_temp,mse_temp,iou])
#    plt.figure(10, figsize=(6, 6), dpi=60)
#    plt.subplot(121)
#    plt.imshow(label_temp)
#    plt.subplot(122)
#    plt.imshow(bi_result_temp)
#    plt.show()
    return result
#get_pr_re(label_temp[:,:,0],bi_lipid_temp)
#def get_pr_re_4(label_true,label_pred):
#    #label_true = []
def psudo_clr(img_clsted,n_clst):
#    color_code = np.asarray([[255,0,0],[0,250,140],[255,255,10],[0,180,255]])/255
#    np.asarray([[245,165,200],[195,240,210],[190,215,240],[255,240,200]])/255 #too dim
#    color_code = np.asarray([[1,0,0],[0,1,0],[0.5,0,0.5],[0,0,1]]) green ER, purple nuclei, blue cytop
#    color_code = np.asarray([[1,0,0],[1,1,0],[0,1,0],[0,0,1]])
#    color_code = np.asarray([[250,255,35],[110,235,25],[248,25,250],[5,175,185]])/255
    color_code = np.asarray([[1,0,0],[0,1,0],[1,1,0],[0,0,1]])
    color_bkg = 0#240/255 #R = G = B = 240, FOR SILVER
    img_color = color_bkg * np.ones((img_clsted.shape[0]*img_clsted.shape[1],3))#np.zeros((img_clsted.shape[0]*img_clsted.shape[1],3))
    for kk in range(n_clst-1,-1,-1):
        img_temp = np.squeeze(img_clsted[:,:,kk]).reshape(img_clsted.shape[0]*img_clsted.shape[1],1)
        locs = np.where(img_temp == 1);
        img_color[locs[0],:] = color_code[kk,:]
    img_color = np.reshape(img_color,(img_clsted.shape[0],img_clsted.shape[1],3))
    return img_color


#result_clr = psudo_clr(bi_result_temp,4)