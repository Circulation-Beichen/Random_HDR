clear all;close all;

%% 读取文件
% FlatImgFilePath = 'D:\IRBNB\算法仿真与验证\HDR算法验证数据集\烟头\挡板';      %条纹    %烟头    
FlatImgFilePath = 'D:\IRBNB\算法仿真与验证\HDR算法验证数据集\数据集\均匀挡板';
FlatImg = getAverageImg(FlatImgFilePath);

% P500ImgFilePath = 'D:\IRBNB\算法仿真与验证\HDR算法验证数据集\烟头\场景';    
P500ImgFilePath = 'D:\IRBNB\算法仿真与验证\HDR算法验证数据集\数据集\空到地';   %空到地 树叶摇晃';    %强光干扰';   %空地交接';
dirOutput=dir(fullfile(P500ImgFilePath,'*.raw'));
fileNames={dirOutput.name};
FrameNum = length(fileNames);

%% 存储数据
fileout = ['D:\IRBNB\算法仿真与验证\HDR算法验证数据集\randsignal\空到地\'];
% fileout = ['D:\IRBNB\算法仿真与验证\HDR算法验证数据集\randsignal\强干扰\'];
index_out = 1;WW=640; HH=512;

figure;

for index = 1:FrameNum
    %% original data
    index = min(index,FrameNum);
    [img] = RAW_READ([P500ImgFilePath '\' fileNames{index}]); 
    mmax = max(max(img));   mmin = min(min(img));
    img_un = (img - mmin)/(mmax-mmin);
    subplot(321);imshow(mat2gray(img_un));title(['Cur = ' num2str(index) '   Tot =  ' num2str(FrameNum)]);
    subplot(322);hh = imhist(uint16(img),65536);plot(hh);title(['Max = ' num2str(mmax) '   Min =  ' num2str(mmin) '   Diff = ' num2str(mmax-mmin)]);
    
    %% nonuniform corrected data
    imd = img - FlatImg;
    imd = imd + 32768;
    mmax = max(max(imd));   mmin = min(min(imd));
    img_un = (imd - mmin)/(mmax-mmin);
    subplot(323);imshow(img_un);title(['Cur = ' num2str(index) '   Tot =  ' num2str(FrameNum)]);
    subplot(324);hh = imhist(uint16(imd),65536);plot(hh);title(['Max = ' num2str(mmax) '   Min =  ' num2str(mmin) '   Diff = ' num2str(mmax-mmin)]);
    
    %% 存储数据供课堂使用
    filename = [fileout num2str(index_out) '.raw'];
%     filesave(filename,imd,WW,HH);    
    index_out = index_out + 1;
   
    %%
    %双平台直方图均衡
    [img_DPHE,img_GHE] = DPHE_v2(imd,600,200);
%     figure(200);imshow(mat2gray(img_DPHE));title('双平台直方图均衡后mat2gray');
%     bfilt_fuzz(mat2gray(img_DPHE));
    subplot(325);imshow(mat2gray(img_DPHE));title(['Cur = ' num2str(index) '   Tot =  ' num2str(FrameNum)]);
    subplot(326);hh = imhist(img_DPHE,256);plot(hh);title(['Max = ' num2str(mmax) '   Min =  ' num2str(mmin) '   Diff = ' num2str(mmax-mmin)]);
    
    %%
    pause(0.5);

end

function filesave(filename,img,WW,HH)
fp = fopen(filename,'wb');
img = uint16(img);
fwrite(fp,img,'uint16');
fclose(fp);

% fp = fopen(filename,'rb');
% A = fread(fp,[HH,WW],'uint16');   %A = fread(fileID,sizeA,precision)
% fclose(fp);

end





