# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:16:24 2019
@author: jing6
"""

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import math
import random 
from supp_func import global_norm_noise
def get_new_range(size_x,angle,crop_width):
    new_range = np.arange(get_B(size_x,angle),-get_B(size_x,angle)+get_L(size_x,angle)) 
    return [np.min(new_range),np.max(new_range)-crop_width-1]
def get_B(xx,alpha):
    BB = xx*math.cos(alpha)*math.sin(alpha)/(math.cos(alpha)+math.sin(alpha))
    BB = math.ceil(BB)
    return BB
def get_L(xx,angle):
    LL = xx*(1/math.sin(0.7854+angle)/math.sqrt(2))
    if math.ceil(LL)%2 ==0:
        LL = math.ceil(LL)
    else:
        LL = math.ceil(LL) + 1
    return LL
def test_crop_(label_):
    label_ = np.asarray(label_)
    for ii in range(4):
        plt.figure(12, figsize=(6, 6), dpi=60)
        plt.subplot(1,5,ii+1)
        plt.imshow(label_[ii,:,:])
    plt.show()    
#%%
def get_splited_dataset(ori_train, label_train, norm = 1, norm_m = 0.5, norm_std = 0.125, noise_level = 0):    
    
    ori_train = global_norm_noise(ori_train,norm_m,norm_std,noise_level)
    size_x, size_y = ori_train.shape[1],ori_train.shape[2]
    crop_width, crop_height = 128, 128
    classlist = ['lipid','ER','nuclei','cytop']
    label_train_crop = []
    ori_train_crop = []
    for imgno in range(ori_train.shape[0]):
        img_cell = ori_train[imgno,:,:].reshape((size_x,size_y))
        mean_img = np.mean(img_cell)
        kk,kk_rot = 0,0
        locs_ii, locs_jj = [], []
        rot_ii, rot_jj, rot_angle = [], [], []
        for nn in range(10):## just cropping 
            ii = math.floor(random.uniform(0,size_x - crop_width - 1))
            jj = math.floor(random.uniform(0,size_y - crop_height - 1))
            crop_img_cell = img_cell[ii:ii+crop_height,jj:jj+crop_width]
            if np.mean(crop_img_cell) > 1.05 * mean_img:#1.05 for lipid, nuclei, er, 
                locs_ii.append(ii)
                locs_jj.append(jj)
                ori_train_crop.append(crop_img_cell)
#                im = Image.fromarray(crop_img_cell)
    #            im.save('label_rot_64x64_noise/ori_png/'+str(imgno) +'_'+ str(kk) + '.png')
                kk += 1
        for nn in range(10):# if imgno!= 0 else 10): ## rotation and cropping 
            angle = random.uniform(5/180*math.pi,40/180*math.pi)
            tttt = get_new_range(size_x,angle,crop_width)
            ii = math.floor(random.uniform(tttt[0],tttt[1]))
            jj = math.floor(random.uniform(tttt[0],tttt[1]))
            img_cell_rot = Image.fromarray(img_cell).rotate(angle*180/math.pi,expand=1)
            crop_img_cell = np.asarray(img_cell_rot)[ii:ii+crop_height,jj:jj+crop_width]
            if np.mean(crop_img_cell) > 1.05 * mean_img:
                rot_ii.append(ii)
                rot_jj.append(jj)
                rot_angle.append(angle)
                ori_train_crop.append(crop_img_cell)
#                im = Image.fromarray(crop_img_cell)
    #            im.save('label_rot_64x64_noise/ori_png/'+str(imgno) +'_'+ str(kk) + '.png')
                kk += 1
        for mmm in range(len(locs_ii)):
            iitemp = locs_ii[mmm]
            jjtemp = locs_jj[mmm]
            crop_temp = label_train[imgno,:,iitemp:iitemp+crop_height,jjtemp:jjtemp+crop_width].reshape((len(classlist),crop_height,crop_width))
#            test_crop_(crop_temp) ## looks good while debug [0:10,]
            label_train_crop.append(crop_temp)
            for classno in range(len(classlist)):
                crop_img_temp = crop_temp[classno,:,:].reshape(crop_height,crop_width)
#                im = Image.fromarray(crop_img_temp)
#            im.save('label_rot_64x64_noise/label_'+classlist[classno]+'/'+str(imgno)+'_'+str(mmm) + '.png')
        for mmm in range(len(rot_ii)):
            iitemp = rot_ii[mmm]
            jjtemp = rot_jj[mmm]
            angletemp = rot_angle[mmm]
            crop_img_temp = []
            for classno in range(len(classlist)):
                crop_img_temp_ = label_train[imgno,classno,:,:].reshape((size_x,size_y))
                crop_img_temp_rot = Image.fromarray(crop_img_temp_).rotate(angletemp*180/math.pi,expand=1)
                ## the mode = 'L' caused unexpected ?overflow ?
                crop_img_temp_rot_crop = np.asarray(crop_img_temp_rot)[iitemp:iitemp+crop_height,jjtemp:jjtemp+crop_width]
                crop_img_temp.append(crop_img_temp_rot_crop)
#                im = Image.fromarray(crop_img_temp_rot_crop)
#            test_crop_(crop_img_temp)
    #            im.save('label_rot_64x64_noise/label_'+classlist[classno]+'/'+str(imgno)+'_'+str(mmm+len(locs_ii)) + '.png')
            label_train_crop.append(np.asarray(crop_img_temp))
             
    for imgno in range(len(ori_train_crop)):
        ori_temp = ori_train_crop[imgno]#,:,:]
        label_temp = label_train_crop[imgno]#,:,:,:]
        if random.uniform(0,1) > 0.8:#rot90
            ori_train_crop.append(np.rot90(ori_temp))
            label_train_crop.append(np.rot90(label_temp,1,(1,2)))
#            test_crop_(np.rot90(label_temp,1,(1,2)))
        if random.uniform(0,1) > 0.8:# flip up and down 
            ori_train_crop.append(np.flip(ori_temp,0))
            label_train_crop.append(np.flip(label_temp,1))
#            test_crop_(np.flip(label_temp,1))
        if random.uniform(0,1) > 0.8:# flip up and down 
            ori_train_crop.append(np.flip(ori_temp,1))
            label_train_crop.append(np.flip(label_temp,2))
#            test_crop_(np.flip(label_temp,2))
#        im = Image.fromarray(np.flip(ori_temp,1))
#        im.show()
    ori_train_crop_rot_flip = np.asarray(ori_train_crop)
    label_train_crop_rot_flip = np.asarray(label_train_crop)
    label_train_crop_rot_flip[label_train_crop_rot_flip>0.5] = 1
    label_train_crop_rot_flip[label_train_crop_rot_flip<0] = 0
    label_train_crop = np.transpose(label_train_crop,[0,2,3,1])  
    ori_train_crop_rot_flip = ori_train_crop_rot_flip[:,:,:,np.newaxis]
    return ori_train_crop_rot_flip,label_train_crop_rot_flip
#    savename = 'dataset_rot_flip_array_64x64_raw_no111213'
#    np.savez(savename,ori_train_crop = ori_train_crop_rot_flip,label_train_crop = label_train_crop_rot_flip,ori_test=ori_test,label_test=label_test)        
    #%%
    #import matplotlib.pyplot as plt
    #plt.hist(ori_train_crop_rot_flip.flatten(),bins = np.linspace(0,1.5,100))
    #plt.xlabel('intnesity after normalization')
    #plt.ylabel('pixel counts')
    #plt.title(savename) 
    #plt.savefig(savename + '.png')
    #%%       
    #np.savez('oldataset_rot',ori_train_crop = ori_train_crop,label_train_crop = label_train_crop,ori_test=ori_test,label_test=label_test)        
#img_arr_aug, label_arr_aug = get_splited_dataset(ori_train[0:10,:,:], label_train[0:10,:,:,:], norm = 1, norm_m = 0.5, norm_std = 0.125, noise_level = 0)        
