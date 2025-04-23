%% selective detail enhancement v.s full detail enhancement

clear;clc;close all;
% img = RAW_READ('C:\Users\86188\Desktop\工作\HDR\20220724172936187.raw');
% img = RAW_READ('D:\#20240501暂时\科研\项目\红外细节增强\from阎展\移交\小信号\001.raw'); %1280*1024
% img = RAW_READ('D:\IRBNB\图像数据\中波经开区输出9靶标20240501\中波1280\二楼实验靶标\T10\350mk-0\001.raw');

img = RAW_READ('D:\#20240501暂时\教学\2024-2025-2随机信号分析\课件材料\2-实验系列\实验4参考程序\FROMYZ\大\20220724172937420.raw');  %640*512
img = RAW_READ('D:\#20240501暂时\教学\2024-2025-2随机信号分析\课件材料\2-实验系列\实验4参考程序\FROMYZ\小信号\000.raw');  %1280*1024

% 下面的数据有问题
% img = RAW_READ('D:\#20240501暂时\教学\2024-2025-2随机信号分析\课件材料\2-实验系列\2-实验数据4的数据\空到地\1000.raw');  %640*512  %强干扰



img = imrotate(img,180);

% img = imread('C:\Users\86188\Desktop\工作\改进的引导滤波\数据集\open-sirst-v2-master\images\backgrounds\S4_14.png');
% img = imread('C:\Users\86188\Desktop\工作\改进的引导滤波\数据集\open-sirst-v2-master\images\targets\S20210527_S4_26.png');
[H,W] = size(img);
%%
simg = img;
% 线性拉伸

img_min = mean2(simg) - 4*std2(simg);
img_max = mean2(simg) + 4*std2(simg);
img_linearStretch = linearStretch(simg,img_min,img_max);
figure(9);

imshow(mat2gray(img_linearStretch));title('线性拉伸后');

disp('pause ......');
pause;

imwrite(mat2gray(img_linearStretch),"linearStretch.png")
img_ls = uint8(255 .* (mat2gray(img_linearStretch)));
img_ori = uint8(255 .* (mat2gray(img)));
X = double(img_ls);

r = 8;
% lambda = 1/128;
k1 = 10;
lambda = k1*var(X(:));
% lambda = 0.01;
% theta = 4;
theta = 1;

X_full = FullDetailEnhancement(X, r, lambda, theta);

Iout1 = uint8(255 .* (mat2gray(X_full)));
figure(123)
imshow(Iout1);


imwrite(Iout1,"out1.png")
out = [img_ls, Iout1];
figure('Name', '拉伸对比细节增强后');
imshow(out); 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% only test
% figure;imshow(img_ls);
% figure;imshow(Iout1);
% 
% 
% psfi = fspecial('gaussian',5,0.5);
% [J,psfr] = deconvblind(double(img_ls),psfi);
% figure;imshow(uint8(J));
% 
% psfi = fspecial('gaussian',5,3);
% [J,psfr] = deconvblind(double(Iout1),psfi);
% figure;imshow(uint8(J));
% 
% J = deconvlucy(double(Iout1),psfi,5);%使用deconvlucy滤波
% figure;imshow(uint8(J));
