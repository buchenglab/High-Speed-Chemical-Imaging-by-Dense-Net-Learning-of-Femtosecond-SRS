function img_color = psudo_clr(img_clsted,n_clst)

[size_x,size_y,~] = size(img_clsted);
color_code = [[1,0,0];[0,1,0];[1,1,0];[0,0,1]]; % [R, G, B]
color_bkg = 0;
img_color = color_bkg * ones([size_x*size_y,3]);
for kk = n_clst:-1:1
    img_temp = squeeze(reshape(img_clsted(:,:,kk),[size_x*size_y,1]));
    locs = img_temp >= 1;
    img_color(locs,:) = img_color(locs,:) + color_code(kk,:);
end
img_color = reshape(img_color,[size_x,size_y,3]);

end