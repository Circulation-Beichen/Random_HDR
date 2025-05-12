% % % calculate the two point correction parameters for each integral time
% % close all;
% % clear all;
% % IMH = 512;
% % IMW = 640;
% function [img] = RAW_READ(filename) 
% % IMW = 1280;
% %  IMH = 1024;  %%512
% % debug = 0;
%  IMW = 640;
%  IMH = 512;  %%512
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % if(debug)
% % %     filename ='G:\IRB\PR6PIXEL640\matlabdebug\CAMLTEST\INT05\TMP05\20220216172134546.raw';
% %     filename = 'E:\FPGA\bian\HDR_new\test001.raw';
% % end
% 
% fp = fopen(filename);
% 
% a = fread(fp,'uint16');
% % a = bitshift(a,-2);
% img = reshape(a,[IMW,IMH])';
% 
% % img = img_tmp;%(1:512,:); %(1:end,:);
% % img(512,:) = img_tmp(511,:);
% 
% % if(debug)
% %     figure;imshow(mat2gray(img));
% % end
% 
% fclose(fp);

% % calculate the two point correction parameters for each integral time
% close all;
% clear all;
% IMH = 512;
% IMW = 640;
function [img] = RAW_READ(filename) 
debug = 0;
% IMW = 640;
% IMH = 512;  %%512
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(debug)
%     filename ='G:\IRB\PR6PIXEL640\matlabdebug\CAMLTEST\INT05\TMP05\20220216172134546.raw';
    filename = 'F:\CAMLTEST\TIMG\INT25\20220216174832328.raw';
end

fp = fopen(filename);

a = fread(fp,'uint16');
% a = bitshift(a,-2);
if(size(a,1)==327680)   
    IMW = 640; 
    IMH = 512;
else
    IMW = 1280; 
    IMH = 1024;
end

img = reshape(a,[IMW,IMH])'; 




