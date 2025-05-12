%% 自适应双平台直方图算法
clear; clc; close all;

%% 读取并处理RAW数据
filename = "C:\work_space\vscode\Task_Random_HDR\02_实验数据4的数据\T10-200mk-3\000.raw"
%filename = "C:\work_space\vscode\Task_Random_HDR\02_实验数据4的数据\大\20220724172936187.raw"
%filename = 'C:\work_space\vscode\Task_Random_HDR\02_实验数据4的数据\小信号\000.raw';
%filename = 'C:\Users\Lenovo\Desktop\SuiJiXinHao\Small\000.raw';
fp = fopen(filename, 'r');
img = fread(fp, [1280, 1024], 'uint16'); % 读取16位RAW
fclose(fp);
img = img';                               % 转置为[H, W]
img = imrotate(img, 180);                 % 旋转180度
 min_val = double(min(img(:)));
    max_val = double(max(img(:)));
    img8 = uint8(255 * (double(img) - min_val) / (max_val - min_val + eps));
    
%% 图像增强
enhanced_img = NiuBiPlus(img8);

%% 计算原始与增强图像的对比度
C_original = Pingjia(img8);
C_enhanced = Pingjia(enhanced_img);
fprintf('对比度提升: %.2f dB\n', C_enhanced - C_original);
%% 显示结果（自动缩放对比度）
figure;
imshow(img, []), title('原始图像');
figure;
imshow(enhanced_img, []), title('增强图像');

%% 自适应算法函数
function enhanced_img = NiuBiPlus(input_img)
    % 输入：input_img为灰度图像（需确保为uint16类型）
    % 获取图像参数
    [Height, width] = size(input_img);
    N_total = Height * width;       % 总像素数
    M = 65536;  % 旧的最大灰度级


    % 生成直方图
    YuanShiTu = imhist(input_img); 
    
    % 去除零统计值，构建非零直方图N(s)
    QuLing_idx = YuanShiTu > 0;
    N = YuanShiTu(QuLing_idx);
    L = length(N);                  % 非零统计值的灰度级数量

    % 计算上阈值T_UP
    n = 5;                          % 滑动窗口大小
    half_n = floor(n/2);
    polar = [];                     % 存储局部最大值

    % 滑动窗口检测局部最大值
    for s = 1:L
        win_start = max(1, s - half_n);
        win_end = min(L, s + half_n);
        window = N(win_start:win_end);
        center_val = N(s);
        if center_val == max(window) && sum(window == center_val) == 1
            polar = [polar, center_val];
        end
    end

    % 计算T_UP（若未找到局部最大值，则取直方图均值）
    if isempty(polar)
        T_UP = mean(N);
    else
        T_UP = mean(polar);
    end
    fprintf('T_UP= %.2f \n', T_UP);

    % 计算下阈值T_DOWN 
    Sta = min(N_total, T_UP * L);
    T_DOWN = Sta / M;

    fprintf('T_DOWN= %.2f \n', T_DOWN);

    % 修正直方图P_m(k) 
    Pm = zeros(size(YuanShiTu));
    for k = 1:length(YuanShiTu)
        p_k = YuanShiTu(k);
        if p_k == 0
            Pm(k) = 0;
        elseif p_k >= T_UP
            Pm(k) = T_UP;
        elseif p_k <= T_DOWN
            Pm(k) = T_DOWN;
        else
            Pm(k) = p_k;
        end
    end

    % 计算累积直方图并重映射灰度级 
    F = cumsum(Pm);                 % 累积和
    F_max = F(end);                 % 累积最大值
    Dm = zeros(size(F));
    for k = 1:length(F)
        %Dm(k) = floor((M * F(k)) / F_max);
        Dm(k) = floor((256 * F(k)) / F_max); % 这里用新的最大灰度值
    end

    % 应用灰度映射（索引+1以适配MATLAB）
    enhanced_img = Dm(double(input_img) + 1); % 关键修正：+1
    enhanced_img = uint8(enhanced_img);      % 保持8位输出
end
%% 对比评价
function C = Pingjia(img)
    img = double(img);
    mean_val = mean(img(:));
    C = mean(img(:).^2) - mean_val^2;
    C = 10 * log10(C);
end


