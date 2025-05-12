import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def single_scale_retinex(image, sigma):
    """
    单尺度 Retinex
    :param image: 输入图像 (应为浮点数类型，范围 0-1 或 0-255)
    :param sigma: 高斯模糊的标准差
    :return: Retinex 处理后的图像 (对数域)
    """
    # 为了避免 log(0) 的错误，给图像加上一个很小的正数
    image_plus_epsilon = image.astype(np.float32) + 1e-6
    # 对图像进行高斯模糊，作为光照分量的估计
    blurred = cv2.GaussianBlur(image_plus_epsilon, (0, 0), sigma)
    # 计算 Retinex：log(I) - log(L')
    retinex = np.log10(image_plus_epsilon) - np.log10(blurred)
    return retinex

def multi_scale_retinex(image, sigmas, weights=None):
    """
    多尺度 Retinex
    :param image: 输入图像 (应为浮点数类型)
    :param sigmas: 包含多个高斯模糊标准差的列表，如 [15, 80, 200]
    :param weights: 对应每个尺度的权重列表，如果为 None，则平均加权
    :return: MSR 处理后的图像 (对数域)
    """
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    elif len(weights) != len(sigmas):
        raise ValueError("Sigmas 和 weights 的长度必须一致")
    if sum(weights) != 1.0: # 可选：归一化权重
        weights = [w / sum(weights) for w in weights]

    msr_image = np.zeros_like(image, dtype=np.float32)
    for i, sigma in enumerate(sigmas):
        msr_image += weights[i] * single_scale_retinex(image, sigma)
    return msr_image

# --- 从 imread1.py 导入的 DPHE 相关函数 ---
def get_upper_threshold(hist):
    """
    计算自适应上阈值 T_UP (Calculate adaptive upper threshold T_UP)
    :param hist: 输入直方图
    :return: 上阈值 T_UP
    """
    # 1. 获取非零直方图 N (Get non-zero histogram N)
    non_zero_indices = hist > 0
    N = hist[non_zero_indices]
    L = len(N) # 非零统计值的数量 (Number of non-zero counts)

    # 2. 加窗，最大值滤波，然后将最大值求平均 (Windowing, max filtering, then average the maxima)
    #    This corresponds to finding local maxima in the non-zero histogram and averaging them.
    n_window = 5 # 滑动窗口大小 (Sliding window size) - can be adjusted
    half_n = n_window // 2
    local_maxima = []

    if L > 0: # 仅当存在非零计数时进行 (Only proceed if non-zero counts exist)
        for s in range(L):
            # Define the window boundaries
            win_start = max(0, s - half_n)
            win_end = min(L, s + half_n + 1) # Python slice excludes end
            window = N[win_start:win_end]
            
            # Get the center value of the window (current N[s])
            center_val = N[s]
            
            # Check if the center value is the unique maximum within the window
            if center_val == np.max(window) and np.sum(window == center_val) == 1:
                local_maxima.append(center_val)

        if not local_maxima: # 如果未找到局部最大值 (If no local maxima found)
            # Fallback: use the mean of all non-zero histogram counts
            Threshold_upper = np.mean(N) 
        else:
            # Calculate the average of the found local maxima
            Threshold_upper = np.mean(local_maxima) 
    else: # 如果直方图为空或所有像素值相同 (If histogram is empty or all pixels have the same value)
        Threshold_upper = 0 # Assign a default value (e.g., 0)

    print(f"Calculated Threshold_upper (T_UP): {Threshold_upper}")
    return Threshold_upper

def get_lower_threshold(hist, upper_threshold):
    """
    计算自适应下阈值 T_LOW (Calculate adaptive lower threshold T_LOW)
    :param hist: 输入直方图
    :param upper_threshold: 上阈值 T_UP
    :return: 下阈值 T_LOW
    """
    # 1. 获取非零直方图 N (Get non-zero histogram N)
    non_zero_indices = hist > 0
    N = hist[non_zero_indices]
    L = len(N) # 非零统计值的数量 (Number of non-zero counts)
    
    # 2. Calculate total number of pixels
    total_pixels = np.sum(hist)

    # 3. Calculate the lower threshold based on the formula:
    #    Threshold_lower = min(total_pixels, upper_threshold * L)
    #    where L is the number of non-zero histogram counts.
    if L > 0:
        sta = min(total_pixels, upper_threshold * L)
        threshold_lower = sta / 65536
    else: # If L is 0 (no non-zero counts, or only bin 0 has counts)
        threshold_lower = 0 # Assign a default value (e.g., 0) as min(total_pixels, 0)

    print(f"Calculated Threshold_lower (T_LOW): {threshold_lower}")
    return threshold_lower

def process_image_dphe(image, hist, upper_threshold, lower_threshold):
    """
    使用动态双平台直方图均衡化(DPHE)处理图像
    :param image: 输入图像
    :param hist: 图像的直方图
    :param upper_threshold: 上阈值 T_UP
    :param lower_threshold: 下阈值 T_LOW
    :return: 处理后的8位图像
    """
    # 1. 根据 DPHE 规则修改原始直方图
    modified_hist = np.copy(hist).astype(np.float64) # 使用 float 以便后续计算 CDF

    # 应用上阈值: 限制计数值过高的灰度级 (通常是背景)
    modified_hist[hist >= upper_threshold] = upper_threshold

    # 应用下阈值: 提升计数值过低的灰度级 (通常是细节)
    # 注意: 只修改那些原本像素数大于 0 且小于等于 lower_threshold 的项
    modified_hist[(hist > 0) & (hist <= lower_threshold)] = lower_threshold

    # 2. 计算修改后直方图的累积分布函数 (CDF)
    cdf = np.cumsum(modified_hist)
    cdf_max = cdf[-1] # CDF 的最大值 (等于修改后总像素计数)

    # 3. 归一化 CDF，创建查找表 (LUT)
    lut = np.floor((255 * cdf) / cdf_max) # 使用 floor 类似 MATLAB 代码
    lut = np.clip(lut, 0, 255)            # 确保输出在 [0, 255] 范围内
    lut = lut.astype(np.uint8)           # 最终 LUT 需要是 uint8 类型

    # 4. 应用查找表 (LUT) 到原始图像
    enhanced_image_8bit = lut[image]

    # 5. 返回结果
    return enhanced_image_8bit

def apply_optimized_msr_dphe(image_path, sigmas=[10, 60, 200], display_intermediate=True):
    """
    应用优化的 MSR 和 DPHE
    :param image_path: 16-bit TIR 图像路径 (如 .tiff, .png) 或已加载的 16-bit NumPy 数组
    :param sigmas: MSR 的尺度
    :param display_intermediate: 是否显示中间处理结果
    :return: 处理后的 8-bit 图像 (0-255, uint8)
    """
    # 1. 处理输入：可能是文件路径或直接是16位图像数组
    if isinstance(image_path, str):
        # 如果输入是字符串（文件路径），则读取图像
        img_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_16bit is None:
            print(f"错误: 无法读取图像 {image_path}")
            return None
    elif isinstance(image_path, np.ndarray):
        # 如果输入已经是NumPy数组，则直接使用
        img_16bit = image_path
    else:
        print("错误: 输入必须是图像路径或NumPy数组")
        return None

    # 假设 TIR 图像是单通道灰度图
    if len(img_16bit.shape) == 3:
        print("警告: 输入图像似乎是多通道的，将转换为灰度图。")
        img_16bit = cv2.cvtColor(img_16bit, cv2.COLOR_BGR2GRAY) # 或者其他转换方式

    # 将 16-bit 图像归一化到 0-1 的浮点数范围 (MSR 通常在浮点数上操作)
    img_float = img_16bit.astype(np.float32) / 65535.0 # 2^16 - 1

    # 2. 应用 MSR
    msr_log_image = multi_scale_retinex(img_float, sigmas)

    # 3. 将 MSR 输出转换回适合 DPHE 处理的范围
    # MSR 输出的是对数域的值，可能为负。需要映射到 0-65535 (16位).
    msr_output_normalized = cv2.normalize(msr_log_image, None, 0, 1, cv2.NORM_MINMAX)
    
    # 将归一化后的 MSR 输出转换为 16-bit uint16 图像 (0-65535)，以便进行DPHE处理
    msr_output_16bit = (msr_output_normalized * 65535.0).astype(np.uint16)

    if display_intermediate:
        # 显示MSR处理结果（为了显示转换为8位）
        msr_output_8bit = (msr_output_normalized * 255).astype(np.uint8)
        cv2.imshow("MSR Output", msr_output_8bit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 4. 应用 DPHE (动态双平台直方图均衡化)
    # 计算16位图像的直方图
    hist, _ = np.histogram(msr_output_16bit.flatten(), bins=65536, range=[0, 65536])
    
    # 计算自适应阈值
    upper_threshold = get_upper_threshold(hist)
    lower_threshold = get_lower_threshold(hist, upper_threshold)
    
    # 应用DPHE处理
    final_image_8bit = process_image_dphe(msr_output_16bit, hist, upper_threshold, lower_threshold)

    return final_image_8bit

# --- 使用示例 ---
if __name__ == "__main__":
    # 使用与 imread1.py 相同的参数读取原始 16-bit 图像
    file_path = r'C:\work_space\vscode\Task_Random_HDR\02_data\T10_200mk_3\000.raw'
    width = 1280
    height = 1024
    pixel_dtype = np.uint16

    bytes_per_pixel = np.dtype(pixel_dtype).itemsize
    frame_size_bytes = width * height * bytes_per_pixel

    with open(file_path, 'rb') as f:
        raw_data = f.read(frame_size_bytes)
        img_16bit = np.frombuffer(raw_data, dtype=pixel_dtype).reshape((height, width))

    # 应用优化的 MSR+DPHE 处理
    processed_img = apply_optimized_msr_dphe(img_16bit, sigmas=[20, 100, 240], display_intermediate=True)

    if processed_img is not None:
        # 仅用于显示：将16位图像归一化为8位，不影响处理
        img_8bit_for_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("Original 16-bit Image (Normalized)", img_8bit_for_display)
        cv2.imshow("Processed MSR+DPHE", processed_img)

        # --- 保存为 JPG --- 
        output_jpg_filename = file_path.replace('.raw', '_MSR_DPHE_enhanced.jpg')
        # --- 同时保存为 PNG ---
        output_png_filename = file_path.replace('.raw', '_MSR_DPHE_enhanced.png')
        try:
            # JPG保存 - 使用matplotlib (jpg格式不支持单通道，总是会转为RGB)
            plt.imsave(output_jpg_filename, processed_img, cmap='gray', format='jpg')
            print(f"Enhanced image saved as: {output_jpg_filename}")
            
            # PNG保存 - 确保保存为单通道灰度图像
            # 方法1：使用OpenCV直接保存单通道图像
            cv2.imwrite(output_png_filename, processed_img) # OpenCV默认保持通道数
            
            # 方法2：也可以用PIL/Pillow库确保单通道保存
            # from PIL import Image
            # Image.fromarray(processed_img).save(output_png_filename)
            
            print(f"Enhanced image also saved as single-channel PNG: {output_png_filename}")
            
            # 可选：检查并打印保存的PNG文件格式信息
            try:
                saved_img = cv2.imread(output_png_filename, cv2.IMREAD_UNCHANGED)
                print(f"保存的PNG图像形状: {saved_img.shape}, 数据类型: {saved_img.dtype}")
                if len(saved_img.shape) == 2:
                    print("成功保存为单通道灰度图像")
                else:
                    print(f"注意：保存的PNG有 {saved_img.shape[2]} 个通道")
            except Exception:
                pass  # 忽略检查过程中的错误
        except Exception as e:
            print(f"Error saving images: {e}")
        
        print("按任意键关闭窗口并退出程序...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 确保程序正常退出
        print("程序已完成")
    else:
        print("图像处理失败。")
