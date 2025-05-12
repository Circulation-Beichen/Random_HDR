import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 新增：伽马校正函数 ---
def apply_gamma_correction(image_16bit, gamma=1.0):
    """对 16 位图像应用伽马校正"""
    if gamma == 1.0:
        return image_16bit # 伽马为 1 则不改变图像
    
    # 归一化到 [0, 1]
    image_norm = image_16bit / 65535.0
    # 应用伽马校正
    corrected_norm = np.power(image_norm, 1.0 / gamma)
    # 反归一化回 [0, 65535] 并转换为 uint16
    corrected_16bit = np.clip(corrected_norm * 65535.0, 0, 65535).astype(np.uint16)
    return corrected_16bit
# --- 伽马校正函数结束 ---

def get_upper_threshold(hist):
    # 计算自适应上阈值 T_UP (Calculate adaptive upper threshold T_UP)
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

    print(f"Calculated Threshold_upper (T_UP): {Threshold_upper}") # Optional: print the calculated threshold
    return Threshold_upper

def get_lower_threshold(hist,upper_threshold):
    # 计算自适应下阈值 T_LOW (Calculate adaptive lower threshold T_LOW)
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

    print(f"Calculated Threshold_lower (T_LOW): {threshold_lower}") # Optional: print the calculated threshold
    return threshold_lower
    
def process_image(image,hist,upper_threshold,lower_threshold):
    # #测试
    # upper_threshold = 31212.22 
    # lower_threshold = 5120.00 
    # 1. 根据 DPHE 规则修改原始直方图
    # 我们在这里根据传入的阈值动态修改，这样函数更通用
    modified_hist = np.copy(hist).astype(np.float64) # 使用 float 以便后续计算 CDF

    # 应用上阈值: 限制计数值过高的灰度级 (通常是背景)
    modified_hist[hist >= upper_threshold] = upper_threshold

    # 应用下阈值: 提升计数值过低的灰度级 (通常是细节)
    # 注意: 只修改那些原本像素数大于 0 且小于等于 lower_threshold 的项
    modified_hist[(hist > 0) & (hist <= lower_threshold)] = lower_threshold

    # 2. 计算修改后直方图的累积分布函数 (CDF)
    #    CDF[i] 表示灰度值小于等于 i 的像素在修改后分布中占多少 (累积计数值)
    cdf = np.cumsum(modified_hist)
    cdf_max = cdf[-1] # CDF 的最大值 (等于修改后总像素计数)

    # 3. 归一化 CDF，创建查找表 (LUT)
    #    目标是将 CDF 的范围 [0, cdf_max] 映射到 8 位图像的目标范围 [0, 255]
    #    归一化公式: new_value = round(((cdf[v] - cdf_min) / (cdf_max - cdf_min)) * L_max)
    #    这里 cdf_min 通常认为是 cdf 中第一个非零值，但在 DPHE 修改后，cdf[0]可能非零
    #    且 L_max = 255 (8 位最大值)
    #    简化后 (假设 cdf_min 接近 0 或不需特别处理):
    #    lut[v] ≈ (cdf[v] / cdf_max) * 255
    lut = np.floor((255 * cdf) / cdf_max) # 使用 floor 类似 MATLAB 代码
    lut = np.clip(lut, 0, 255)            # 确保输出在 [0, 255] 范围内
    lut = lut.astype(np.uint8)           # 最终 LUT 需要是 uint8 类型

    # 4. 应用查找表 (LUT) 到原始图像
    #    这是"恢复"或应用变换的关键步骤。
    #    原始图像 `image` 中的每个像素值 (例如，范围 0-65535)
    #    被直接用作索引去查找 `lut` 数组中对应位置的值。
    #    `lut` 数组的长度应该与 `hist` 相同 (例如 65536)。
    #    查找到的值 (范围 0-255) 就是该像素在最终 8 位增强图像中的新灰度值。
    enhanced_image_8bit = lut[image]

    # 5. 返回结果
    return enhanced_image_8bit


def enhance_image(image):
    # 计算直方图
    hist, bins = np.histogram(image.flatten(), bins=65536, range=[0, 65536])
    
    # 显示原直方图
    plt.figure(figsize=(10, 6))
    plt.plot(bins[:-1], hist, lw=0.5)
    plt.title('Original 16-bit Image Histogram')
    plt.xlabel('Gray Level (0-65535)')
    plt.ylabel('Pixel Count')
    plt.grid(True)
    plt.show()

    # 动态双平台增强 (Dynamic Dual-Platform Enhancement)
    upper_threshold = get_upper_threshold(hist)
    lower_threshold = get_lower_threshold(hist,upper_threshold)
    #测试
    # upper_threshold = 31212.22 
    # lower_threshold = 5120.00 
    enhanced_image = process_image(image,hist,upper_threshold,lower_threshold)

    # 显示增强后的直方图 (Display enhanced histogram)
    hist_enhanced, bins_enhanced = np.histogram(enhanced_image.flatten(), bins=256, range=[0, 256])
    plt.figure(figsize=(10, 6))
    plt.plot(range(256), hist_enhanced, lw=0.5, color='r')
    plt.title('Enhanced 8-bit Image Histogram')
    plt.xlabel('Gray Level (0-255)')
    plt.ylabel('Pixel Count')
    plt.grid(True)
    plt.show()

    return enhanced_image

if __name__ == "__main__":
    # --- 请替换为实际参数 ---
    file_path = r'C:\work_space\vscode\Task_Random_HDR\02_data\T10_200mk_3\000.raw'
    # file_path = r'C:\work_space\vscode\Task_Random_HDR\02_data\T10_200mk_3\000.raw'
    width = 1280  # 示例宽度 (Example width)
    height = 1024  # 示例高度 (Example height)
    gamma_value = 0.5 # 示例伽马值，可以调整 (例如 0.45, 0.5 用于提亮暗部, 2.2 用于模拟显示器)

    pixel_dtype = np.uint16

    # 计算每帧的字节数
    bytes_per_pixel = np.dtype(pixel_dtype).itemsize
    frame_size_bytes = width * height * bytes_per_pixel

    with open(file_path, 'rb') as f:
        # 读取第一帧数据
        raw_data = f.read(frame_size_bytes)

        # 将字节流转换为 NumPy 数组
        image = np.frombuffer(raw_data, dtype=pixel_dtype)
        image = image.reshape((height, width))

        # --- 新增：应用伽马校正 ---
        print(f"Applying Gamma correction with gamma={gamma_value}...")
        gamma_corrected_image = apply_gamma_correction(image, gamma_value)
        # --- 伽马校正结束 ---

        # enhanced_image = enhance_image(image) # 原调用
        enhanced_image = enhance_image(gamma_corrected_image) # 使用伽马校正后的图像
        
        plt.imshow(enhanced_image, cmap='gray')
        plt.title('Final Enhanced Image') # 更新最终图像标题
        plt.show()

        # --- 新增：保存为 JPG --- 
        output_jpg_filename = file_path.replace('.raw', '_enhanced.jpg')
        # --- 同时保存为 PNG ---
        output_png_filename = file_path.replace('.raw', '_enhanced.png')
        try:
            # JPG保存 - 使用matplotlib
            plt.imsave(output_jpg_filename, enhanced_image, cmap='gray', format='jpg')
            print(f"Enhanced image saved as: {output_jpg_filename}")
            
            # PNG保存 - 使用OpenCV确保保存为单通道灰度图
            cv2.imwrite(output_png_filename, enhanced_image)
            print(f"Enhanced image also saved as single-channel PNG: {output_png_filename}")
        except Exception as e:
            print(f"Error saving JPG: {e}")
        # --- 保存结束 ---
