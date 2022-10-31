% This code is for the Phasor-MRF function in paper: 
% J. Zhang, J. Zhao, H. Lin, Y. Tan, and J.-X. Cheng, "High-Speed Chemical Imaging by Dense-Net Learning of Femtosecond Stimulated Raman Scattering," 
%     J. Phys. Chem. Lett. 11, 8573–8578 (2020).
% Reference:  
% [1] P. Ghamisi, J. A. Benediktsson, and M. O. Ulfarsson, "Spectral–Spatial Classification of Hyperspectral Images Based on Hidden Markov Random Fields," 
%     IEEE Transactions on Geoscience and Remote Sensing 52, 2565–2574 (2014).
% [2] J. Li, J. M. Bioucas-Dias, and A. Plaza, "Spectral–Spatial Hyperspectral Image Segmentation Using Subspace Multinomial Logistic Regression and Markov Random Fields," 
%     IEEE Transactions on Geoscience and Remote Sensing 50, 809–823 (2012).
% [3] D. Fu and X. S. Xie, "Reliable Cell Segmentation Based on Spectral Phasor Analysis of Hyperspectral Stimulated Raman Scattering Imaging Data," 
%     Anal. Chem. 86, 4115–4119 (2014).

%% load hyperspectrum data and initial label
% input: 
% SRS_3D: height x width x lambda % SRS_3d = cat(3, SRS_3d_1, SRS_3d_2); 
% label_init: height x width x nlabel (nlabel = 4)
% load('mat_demo.mat');
[size_x, size_y, nspec] = size(SRS_3d);
[~, ~, nlabel] = size(label_init); 
%% initialization
% parameter for segmentation 
alpha = [1,1,1,1];beta = [1, 1, 1, 1]; gama = [1,1,1,1];
% stack pre-processing 
SRS_2d = reshape(SRS_3d,[size_x*size_y,nspec]);
[coeff2,score2] = pca(SRS_2d);
SRS_pca = reshape(score2(:,1:5),[size_x,size_y,5]);
%% spec spat label
label_phasor_2d = spec_spat_phasor(SRS_pca,label_init,nlabel,1,alpha,beta,gama);
img_color_spec_spat = psudo_clr(label_phasor_2d,nlabel);
%% visualization               
figure,subplot(131),imagesc(mean(SRS_3d,3)),title('SRS unweighted mean'),axis square,axis off
subplot(132),imagesc(psudo_clr(label_init,4)),title('phsor label'),axis square,axis off
subplot(133),imagesc(img_color_spec_spat),title('spec-spat-phasor label'),axis square,axis off
% print(gcf,'-dpng',[saveDir,'\',num2str(nnn),'.png'])
%% save parameters 
para_name = {'alpha','beta','gama'};
txt_name = ['para.txt'];
for ii = 1:length(para_name)
    fid = fopen(txt_name,'a');
    fprintf(fid,[para_name{ii},'\n'])
    fclose(fid);
    eval(sprintf(['data_temp = ',para_name{ii},';']));
    dlmwrite(txt_name,data_temp,'-append','delimiter','\t','precision',6);
end

%% 
figure('Position',[0,0,300,300])
axes('Units','Normalize','Position', [0,0,1,1])
    imagesc(img_color_spec_spat), axis off, axis image%colorbar,     
set(gca,'LooseInset',get(gca,'TightInset'));
print(gcf,'-dpng',['spec_spat_clr','.png'])



