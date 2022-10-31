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

function label_phasor_2d = spec_spat_phasor(SRS_re,label_phasor,nlabel,niter,alpha,beta,gama)
%% gama should be in the size of 1 x nalbel to force the balance between each label (?)
[size_x_,size_y,nspec] = size(SRS_re);
label_phasor_2d = zeros(size_x_,size_y);
for nn = 1:nlabel
    locs_temp = label_phasor(:,:,nn) == 1;
    label_phasor_2d = label_phasor_2d + nn*locs_temp;
end
[xxx,yyy] = find(label_phasor_2d>0);
%% main iterations
% line_color = {'.r--','.g--','.b--','.y--'};
for kk = 1:niter
%     if kk > 1
        label_phasor_2d_old = label_phasor_2d;
%     end
    fff = zeros(size_x_,size_y);
    for iii = 1:size(xxx)
        ii = xxx(iii);jj = yyy(iii);
%     for ii = 1:size_x %- 1
%         for jj = 1:size_y %- 1
            if mod(iii,size_x_) == 1 
                theta = get_mu_sigma(SRS_re,label_phasor_2d,nlabel,nspec);
            end
            patch_coords = get_patch_coords(ii,jj,size_x_,size_y);           
            Ux_ij = get_Ux(ii,jj,patch_coords,label_phasor_2d,nlabel).*alpha; %penalty term -> encourage spatial smoothness    
            Z = sum(Ux_ij);
            f_ij = exp(-Ux_ij)./Z;%>???            
            Uyx_ij = get_Uyx(SRS_re(ii,jj,:),theta,nlabel).*beta; % fit within in label  %intuitively, penalty fit of one label by making beta or alpha of that label bigger           
            [~,label_phasor_2d(ii,jj)] = min((Uyx_ij + Ux_ij).*gama);
            fff_temp = (Uyx_ij + Ux_ij)./sum(Uyx_ij + Ux_ij); 
            fff(ii,jj) = min(exp(-fff_temp)./sum(exp(-fff_temp)));%%
%             if fff(ii,jj) ~=0
%                 ii,jj,
%             end
%         end
%     end
    end
    %% EM step
    close all
    mu = zeros(nlabel,nspec);
    sigma = zeros(nlabel,nspec);
    for nn = 1:nlabel
        locs = find(label_phasor_2d == nn);
        copt_temp = reshape(SRS_re,[size_x_*size_y,nspec]);
        fff_temp = reshape(fff,[size_x_*size_y,1]);
        mu_new(nn,:) = sum(repmat(fff_temp(locs),1,nspec).*copt_temp(locs,:)/sum(fff_temp),1);
        sigma_up = repmat(fff_temp(locs),1,nspec).*(copt_temp(locs,:)-repmat(mu_new(nn,:),length(locs),1));
        sigma_new(nn,:) = sum(sigma_up/sum(fff_temp),1);
    end
    theta.mu = mu_new;
    theta.sigma = sigma_new;   
    %% stopping cue
    if kk > 1 && sum(abs(label_phasor_2d_old - label_phasor_2d),'all') < 50
        sprintf('#iteration = %d',kk)
        break
    end
    
end
%% label_2d to label_3d
if length(size(label_phasor_2d)) == 2
    img_3d = zeros([size(label_phasor_2d),1]);
    for kkk = 1:nlabel
        img_3d(:,:,kkk) = label_phasor_2d == kkk;
    end
    label_phasor_2d = img_3d;
end
%% get Uyx_ij
% Uyx_ij = get_Uyx(copt(1,1),theta,nlabel); %fit within in label
function Uyx_ij = get_Uyx(y_ij,theta,nlabel)
y_ij = squeeze(y_ij);
Uyx_ij = [];
for nn_ = 1:nlabel
    Uyx_ij(:,nn_) = sum((y_ij - theta.mu(nn_,:)').^2./2./theta.sigma(nn_,:)'.^2 + log(theta.sigma(nn_,:)'.^2));
%     copt_bi_2d()
end
end
%% 
function theta = get_mu_sigma(copt,copt_bi_2d,nlabel,nspec)%[mu,sigma]
copt = reshape(copt,[size(copt_bi_2d,1)*size(copt_bi_2d,2),nspec]);
mu_ = zeros(nlabel,nspec);
sigma_ = zeros(nlabel,nspec);
for nn_ = 1:nlabel
    locs_ = find(copt_bi_2d == nn_);
%     if isempty(locs)
%         mu(nn,:) = 0;
%     else 
        mu_(nn_,:) = mean(copt(locs_,:),1);
        std_temp = std(copt(locs_,:),1);
        std_temp(std_temp == 0) = 0.01;  
        sigma_(nn_,:) = std_temp;
%     end        
end
theta.mu = mu_;
theta.sigma = sigma_;
end

%% get Ux
function Ux_ij = get_Ux(ii,jj,patch_coords,copt_bi_2d,nlabel)
Ux_ij = [];
for xx = 1:nlabel %% all possible labels
    Ux_ij(xx) = 0;
    for nn_ = 1:size(patch_coords,1)
        if patch_coords(nn_,2)>400
            1;
        end
        if xx ~= copt_bi_2d(patch_coords(nn_,1),patch_coords(nn_,2))        
            Ux_ij(xx) = Ux_ij(xx) + 1/2;
        end
    end
end
end

%% get patch
function locs = get_patch_coords(ii,jj,sizey,sizex)
    if (ii == 1)&&(jj ==1)
        locs = [1,2;2,1];
    elseif (ii == 1)&&(jj == sizex)
        locs = [1,sizex-1;2,sizex];
    elseif (ii == sizey)&&(jj == 1)
        locs = [sizey-1,1;sizey,2];
    elseif (ii == sizey)&&(jj == sizex)
        locs = [sizey,sizex-1;sizey-1,sizex];
    elseif (ii == 1)&&(jj < sizex)&&(jj > 1)
        locs = [1,jj-1;1,jj+1;2,jj];
    elseif (ii == sizey)&&(jj < sizex)&&(jj > 1)
        locs = [sizey,jj-1;sizey,jj+1;sizey-1,jj];
    elseif (jj == 1)&&(ii < sizey)&&(ii > 1)
        locs = [ii-1,1;ii+1,1;ii,2];
    elseif (jj == sizex)&&(ii < sizey)&&(ii >1)
        locs = [ii-1,sizex;ii+1,sizex;ii,sizex-1];
    else
        locs = [ii-1,jj;ii+1,jj;ii,jj-1;ii,jj+1];
    end
end   

%% edge
% function edge_copt = get_edge_copt(copt)
% 
% [size_x_, size_y, nlabel] = size(copt);
% edge_copt = zeros(size_x_,size_y);
% for nn = 1:nlabel
%     copt_temp = copt(:,:,nn);
%     edge_temp = edge(copt_temp);
%     edge_copt(:,:,nn) = edge_temp;
% end
% end

end