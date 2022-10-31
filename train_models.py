"""
Created on Mon Feb  3 20:48:41 2020
@author: jingzhang
"""
from deep_chem import deep_chem, precision, recall
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from supp_func import (psudo_clr, get_pr_re, global_norm_noise,morph_open_cls, fuse_img,\
                       merged,one_hot_label, one_hot_label_porb, stitch_win,  im_adj)
from get_splited_dataset_all import get_splited_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math 
import os
import scipy.io
import glob
import imageio
from time import gmtime, strftime, localtime
from limit_gpu_memory import limit_gpu_memory
from livelossplot.keras import PlotLossesCallback
limit_gpu_memory(0.2, allow_growth=True)
def lr_decay(epoch):
    return max(lr_begin*math.pow(0.5,int(epoch/5)),1e-6)
#    return max(0.01*math.pow(0.995,int(epoch)),1e-6)

#%% loaddata to avg_mat_all and label_mat_all
#%% data augmentation
label_arr = np.transpose(label_mat_all,[3,2,0,1])  
img_arr = np.transpose(avg_mat_all,[2,0,1])  
ori_train,ori_test,label_train,label_test = train_test_split(img_arr,label_arr,random_state=0,test_size=0)
img_arr_aug, label_arr_aug = get_splited_dataset(ori_train, label_train, norm = 1, norm_m = 0.5, norm_std = 0.125, noise_level = 0)        
#%%
note_log = '20191127+1220dataset-clean-'#
model_name = 'DeepChem_0203'
save_dir = '/home/' + model_name + strftime("%Y-%m-%d %H:%M:%S",localtime()) + '-/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
#%% 
imgno_all = np.shape(label_mat_all)[3]
size_x,size_y = label_mat_all.shape[0],label_mat_all.shape[1]
classlist = ['lipid','ER','nuclei','cytop']
epochs = 100   
crop_h,crop_w = 128, 128
crop_h_, crop_w_ = int(crop_h/2), int(crop_w/2)
noimg_x, noimg_y = int(np.floor(size_x/crop_h)),int(np.floor(size_y/crop_w))

#%%
for testno in range(1):
    save_dir_2 = save_dir + strftime("%Y-%m-%d %H:%M:%S",localtime()) + '-/'
    if not os.path.exists(save_dir_2):
        os.mkdir(save_dir_2)
    save_test_dir = save_dir_2 + 'testset/'
    if not os.path.exists(save_test_dir):
        os.mkdir(save_test_dir)
    global lr_begin
#    if (testno== 0) | (testno== 2) : lr_begin = 0.001
#    if (testno== 1) | (testno== 3) : lr_begin = 0.0001
    lr_begin = 0.001
    lrrr = []
    for ii in range(100):
        lrrr.append(lr_decay(ii)) 
    kernel_ = {'class':'l1','rate':1e-4}
#    if testno >= 2: kernel_ = {'class':'l1','rate':1e-6}
#    if testno < 2: kernel_ = {'class':'l1','rate':1e-5}
    kernel_size = 3
    batch_size = 2
    alpha4loss = [0.5,0.7,0.9][testno]
    para = {'kernel_regular':kernel_,'loss':'binary_crossentropy','alpha4loss':alpha4loss,\
            'n_block':5,'neck_feature_dim':60,'batch_size':batch_size,\
            'down_dense_layers':(np.asarray([6,12,18,24,32,24])*8).astype(int),'filter_dim':12,\
            'up_layers':[240,288,240,192,32],'learning_rate':lrrr,\
            'dropout_rate':0.02, 'lr_begin':lr_begin, 'kernel_size':kernel_size}
    with open(save_dir_2 + 'para.txt','w') as f: print(para,file=f)
    save_test_dir = save_dir_2 + 'testset/'
    if not os.path.exists(save_test_dir):
        os.mkdir(save_test_dir)
    for classno in range(1):
         # rescale, preprocesssing_function
        train_label_temp = label_arr_aug[:,classno,:,:]
        lr_decay_callback = LearningRateScheduler(lr_decay, verbose=True)
        #    train_label_arr = train_label_arr[:,:,:,classno].reshape((imgno_all,crop_h,crop_w,1))    
    #    train_label_temp= label_train[:,classno,:,:].reshape((imgno_all,size_x,size_y,1)) 
    #    train_label_temp = label_train[:,classno,:,:].reshape((imgno_all,size_x,size_y,1)) 
        model = deep_chem(para,input_size = (128,128,1))
    #    for layer in model.layers:
    #        print(layer.output_shape)
    #    history = model.fit(train_batches,steps_per_epoch=int(np.size(label_mat_all,-1) / 32), epochs=epochs)
        model_checkpoint = ModelCheckpoint(save_dir_2 + classlist[classno] + note_log + '.hdf5', monitor='loss',verbose=0, save_best_only=True)
    
        history = model.fit(img_arr_aug, train_label_temp[:,:,:,np.newaxis], batch_size=batch_size, epochs=epochs, verbose=1,validation_split=0.15, shuffle=True, callbacks=[model_checkpoint,lr_decay_callback,PlotLossesCallback()])#lr_decay_callback
    
        model_checkpoint = ModelCheckpoint(save_dir_2 + classlist[classno] + note_log + '.hdf5', monitor='loss',verbose=0, save_best_only=True)
    #    history = model.fit(ori_train, train_label_temp, batch_size=2, epochs=100, verbose=1,validation_split=0.15, shuffle=True, callbacks=[model_checkpoint,lr_decay_callback])
    #    lr_decay_callback
    
        plt.figure(12, figsize=(6, 6), dpi=60)
        plt.subplot(221)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.title(save_dir[39:] + classlist[classno] + '\n' + '\n' + 'loss') 
        plt.legend()
        plt.subplot(222)
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='val')
        plt.title('acc')
        plt.legend()
        plt.subplot(223)
        plt.plot(history.history['recall'], label='train')
        plt.plot(history.history['val_recall'], label='val')
        plt.title('recall')
        plt.legend()
        plt.subplot(224)
        plt.plot(history.history['precision'], label='train')
        plt.plot(history.history['val_precision'], label='val')
        plt.title('precision')
        plt.legend()
        plt.savefig(save_dir_2 + classlist[classno] + note_log + '_loss_acc.png')
        plt.show()
        np.savez(save_dir_2 + classlist[classno] + note_log + '_loss_acc',[history.history['loss'],history.history['val_loss'],history.history['acc'],history.history['val_acc']],history.history)    
        with open(save_dir_2 + classlist[classno] + '_model_architecture.json','w') as f:
            f.write(model.to_json())
        #%% test on MRF-filtered data 
        file_path_old = '/home/'
        ori_arr_test = scipy.io.loadmat(file_path_old + 'xx.mat')
        label_arr_test = scipy.io.loadmat(file_path_old + 'xx.mat')
        label_arr_test = np.transpose(label_arr_test['label_spec_spat_mat_'],[3,2,0,1])## img_no x class x W x H
        ori_arr_test = np.transpose(ori_arr_test['weighted_avg_mat_'],[2, 0, 1]) ## img_no x W x H

        #%% save test set 
        pred_metrics = []
        if not os.path.exists(save_test_dir + classlist[classno] + '/'):
                os.mkdir(save_test_dir + classlist[classno] + '/')
        for ii in range(len(label_test)):
            srs_temp = ori_test[ii,:,:]
            stitch_temp = []
            for temp_ii in range(noimg_x*2 - 1):
                for temp_jj in range(noimg_x*2 - 1):
                    patch_temp = srs_temp[temp_ii*crop_h_:(temp_ii+2)*crop_h_,temp_jj*crop_w_:(temp_jj+2)*crop_w_]
                    pred_temp = model.predict(patch_temp.reshape([1,crop_h,crop_w,1]))
                    stitch_temp.append(pred_temp.reshape([crop_h,crop_w]))
            stitch_temp = np.asarray(stitch_temp)
            result_stitch = stitch_win(stitch_temp.reshape(noimg_x*2-1,noimg_y*2-1,crop_h,crop_w),noimg_x,noimg_y,crop_h_,crop_w_)
            result_stitch_ = np.asarray(result_stitch).reshape(noimg_x*2,noimg_y*2, crop_h_, crop_w_).swapaxes(1, 2).reshape(size_x, size_y)         
            pred_temp = (result_stitch_ <= 1) & (result_stitch_ > 0.5)        
            pred_label_mat = np.concatenate((np.zeros((256,256,1)),pred_temp[:,:,np.newaxis],np.zeros((256,256,1)),np.zeros((256,256,1))),axis = 2)
            pred_label_clr = psudo_clr(pred_label_mat,4)        
            srs_temp_adj = im_adj(srs_temp)
            merge_clr = merged(srs_temp_adj, pred_label_clr,0.3)
            imageio.imwrite(save_test_dir + classlist[classno] + '/' + str(ii) + '.png',(merge_clr))
            pred_metrics.append(get_pr_re(pred_temp,label_test[ii,classno,:,:]))
        np.save(save_test_dir + classlist[classno] + '/' + classlist[classno] + '_metrics.npy',np.asarray(pred_metrics))
        test_metrics_txt_path = save_test_dir + classlist[classno] + '/' + classlist[classno] + '_metrics.txt'
        with open(test_metrics_txt_path,'w') as filehandle:   
            for index, item in enumerate(pred_metrics):
                filehandle.write("%s" % index + "   " + str(item).strip('[]') + "\n") 

