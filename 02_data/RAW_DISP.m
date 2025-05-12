%% selective detail enhancement v.s full detail enhancement

clear;clc;close all;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 读取强干扰图像数据
ImgFilePath = 'D:\#20240501暂时\教学\2024-2025-2随机信号分析\课件材料\2-实验系列\2-实验数据4的数据\T10-200mk-3\';
% ImgFilePath = 'D:\#20240501暂时\教学\2024-2025-2随机信号分析\课件材料\2-实验系列\2-实验数据4的数据\强干扰\';
% ImgFilePath = 'D:\#20240501暂时\教学\2024-2025-2随机信号分析\课件材料\2-实验系列\2-实验数据4的数据\大\';
% ImgFilePath = 'D:\#20240501暂时\教学\2024-2025-2随机信号分析\课件材料\2-实验系列\2-实验数据4的数据\小信号\';
dirOutput=dir(fullfile(ImgFilePath,'*.raw'));
fileNames={dirOutput.name};
FrameNum = length(fileNames);
for index = 1 : FrameNum
    img = RAW_READ([ImgFilePath '\' fileNames{index}]);    
%     img_min = mean2(img) - 4*std2(img);
%     img_max = mean2(img) + 4*std2(img);
%     img = linearStretch(img,img_min,img_max);    
    imshow(mat2gray(img));
    pause(0.1);
end

disp('pause ......');
pause(0.1);
