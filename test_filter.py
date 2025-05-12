import numpy as np
import matplotlib.pyplot as plt
import cv2 # 重新导入 OpenCV

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

    # print(f"Calculated Threshold_upper (T_UP): {Threshold_upper}") # 减少打印
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

    # print(f"Calculated Threshold_lower (T_LOW): {threshold_lower}") # 减少打印
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


def enhance_image_with_dphe(image_16bit_input, display_histograms=False, title_prefix=""):
    """对输入的16位图像应用DPHE并返回8位结果，可选显示直方图"""
    # print(f"DPHE input min: {np.min(image_16bit_input)}, max: {np.max(image_16bit_input)}")
    hist, bins = np.histogram(image_16bit_input.flatten(), bins=65536, range=[0, 65536])
    if display_histograms:
        plt.figure(figsize=(10,6))
    plt.plot(bins[:-1], hist, lw=0.5)
        plt.title(f'{title_prefix} Histogram for DPHE Input (16-bit scale)')
        plt.xlabel('Gray Level (0-65535)'); plt.ylabel('Pixel Count'); plt.grid(True); plt.show()

    upper_threshold = get_upper_threshold(hist)
    lower_threshold = get_lower_threshold(hist, upper_threshold)
    # print(f"{title_prefix} DPHE Thresholds: T_UP={upper_threshold:.2f}, T_LOW={lower_threshold:.2f}")

    enhanced_image_8bit = process_image(image_16bit_input, hist, upper_threshold, lower_threshold)

    if display_histograms:
        hist_enhanced, _ = np.histogram(enhanced_image_8bit.flatten(), bins=256, range=[0, 256])
        plt.figure(figsize=(10,6))
    plt.plot(range(256), hist_enhanced, lw=0.5, color='r')
        plt.title(f'{title_prefix} Histogram of DPHE Enhanced 8-bit Image'); plt.xlabel('Gray Level (0-255)'); 
        plt.ylabel('Pixel Count'); plt.grid(True); plt.show()
    return enhanced_image_8bit

# --- 修改后的滤波器函数 (返回16位或处理后的16位等效图像) ---
def apply_max_filter_16bit(image_16bit, kernel_size=3):
    """应用最大值滤波到16位图像，返回16位图像"""
    print(f"Applying Max filter ({kernel_size}x{kernel_size}) to 16-bit image...")
    return cv2.dilate(image_16bit, np.ones((kernel_size,kernel_size), np.uint8))

def apply_min_filter_16bit(image_16bit, kernel_size=3):
    """应用最小值滤波到16位图像，返回16位图像"""
    print(f"Applying Min filter ({kernel_size}x{kernel_size}) to 16-bit image...")
    return cv2.erode(image_16bit, np.ones((kernel_size,kernel_size), np.uint8))

def apply_sobel_to_16bit_and_scale(image_16bit):
    """应用Sobel到16位图像，取幅值，然后缩放到0-65535 uint16范围"""
    print("Applying Sobel filter to 16-bit image and scaling...")
    sobelx_64f = cv2.Sobel(image_16bit, cv2.CV_64F, 1, 0, ksize=3)
    sobely_64f = cv2.Sobel(image_16bit, cv2.CV_64F, 0, 1, ksize=3)
    magnitude_64f = cv2.magnitude(sobelx_64f, sobely_64f)
    # 将梯度幅值归一化到0-1，然后乘以65535
    cv2.normalize(magnitude_64f, magnitude_64f, 0, 65535, cv2.NORM_MINMAX)
    return magnitude_64f.astype(np.uint16)

def apply_laplacian_to_16bit_and_scale(image_16bit):
    """应用拉普拉斯到16位图像，取绝对值，然后缩放到0-65535 uint16范围"""
    print("Applying Laplacian filter to 16-bit image and scaling...")
    laplacian_64f = cv2.Laplacian(image_16bit, cv2.CV_64F, ksize=3)
    laplacian_abs_64f = np.abs(laplacian_64f)
    # 将绝对值归一化到0-1，然后乘以65535
    cv2.normalize(laplacian_abs_64f, laplacian_abs_64f, 0, 65535, cv2.NORM_MINMAX)
    return laplacian_abs_64f.astype(np.uint16)

# --- 新增：直接应用于8位图像的滤波器函数 (返回8位) ---
def apply_max_filter_8bit(image_8bit, kernel_size=3):
    print(f"Applying Max filter ({kernel_size}x{kernel_size}) to 8-bit image...")
    return cv2.dilate(image_8bit, np.ones((kernel_size,kernel_size), np.uint8))

def apply_min_filter_8bit(image_8bit, kernel_size=3):
    print(f"Applying Min filter ({kernel_size}x{kernel_size}) to 8-bit image...")
    return cv2.erode(image_8bit, np.ones((kernel_size,kernel_size), np.uint8))

def apply_sobel_filter_8bit(image_8bit):
    print("Applying Sobel filter to 8-bit image...")
    sobelx_64f = cv2.Sobel(image_8bit, cv2.CV_64F, 1, 0, ksize=3) # 计算时仍用更高精度
    sobely_64f = cv2.Sobel(image_8bit, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx_64f, sobely_64f)
    cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

def apply_laplacian_filter_8bit(image_8bit):
    print("Applying Laplacian filter to 8-bit image...")
    laplacian_64f = cv2.Laplacian(image_8bit, cv2.CV_64F, ksize=3) # 计算时仍用更高精度
    laplacian_abs = np.abs(laplacian_64f)
    cv2.normalize(laplacian_abs, laplacian_abs, 0, 255, cv2.NORM_MINMAX)
    return laplacian_abs.astype(np.uint8)
# --- 滤波器函数结束 ---

# --- 主执行块 --- 
if __name__ == "__main__":
    # --- 请替换为实际参数 ---
    file_path = r'C:\work_space\vscode\Task_Random_HDR\02_实验数据4的数据\T10-200mk-3\000.raw'
    # file_path = r'C:\work_space\vscode\Task_Random_HDR\02_实验数据4的数据\小信号\005.raw'
    width = 1280  # 示例宽度 (Example width)
    height = 1024  # 示例高度 (Example height)
    gamma_value = 0.5 # 示例伽马值，可以调整 (例如 0.45, 0.5 用于提亮暗部, 2.2 用于模拟显示器)
    kernel_filter_size = 3 # 用于最大/最小值滤波的核大小
    pixel_dtype = np.uint16

    # 计算每帧的字节数
    bytes_per_pixel = np.dtype(pixel_dtype).itemsize
    frame_size_bytes = width * height * bytes_per_pixel

    with open(file_path, 'rb') as f:
        # 读取第一帧数据
        raw_data = f.read(frame_size_bytes)

        # 将字节流转换为 NumPy 数组
        image_16bit_original = np.frombuffer(raw_data, dtype=pixel_dtype).reshape((height, width))

        # 准备存储DPHE结果的列表和标题
        dphe_results = []
        dphe_titles = []

        # --- 路径1: 仅伽马校正 -> DPHE ---
        print("\n--- Path 1: Gamma -> DPHE ---")
        processed_16bit_p1 = apply_gamma_correction(image_16bit_original, gamma_value)
        dphe_results.append(enhance_image_with_dphe(processed_16bit_p1, title_prefix="P1:Gamma"))
        dphe_titles.append("Gamma -> DPHE")

        # --- 路径2: 最大值滤波 -> 伽马 -> DPHE ---
        print("\n--- Path 2: Max Filter -> Gamma -> DPHE ---")
        max_filtered_16bit = apply_max_filter_16bit(image_16bit_original, kernel_filter_size)
        processed_16bit_p2 = apply_gamma_correction(max_filtered_16bit, gamma_value)
        dphe_results.append(enhance_image_with_dphe(processed_16bit_p2, title_prefix="P2:Max->Gamma"))
        dphe_titles.append("Max->Gamma->DPHE")

        # --- 路径3: 最小值滤波 -> 伽马 -> DPHE ---
        print("\n--- Path 3: Min Filter -> Gamma -> DPHE ---")
        min_filtered_16bit = apply_min_filter_16bit(image_16bit_original, kernel_filter_size)
        processed_16bit_p3 = apply_gamma_correction(min_filtered_16bit, gamma_value)
        dphe_results.append(enhance_image_with_dphe(processed_16bit_p3, title_prefix="P3:Min->Gamma"))
        dphe_titles.append("Min->Gamma->DPHE")

        # --- 路径4: Sobel滤波 -> 伽马 -> DPHE ---
        print("\n--- Path 4: Sobel -> Gamma -> DPHE ---")
        sobel_processed_16bit = apply_sobel_to_16bit_and_scale(image_16bit_original)
        processed_16bit_p4 = apply_gamma_correction(sobel_processed_16bit, gamma_value)
        dphe_results.append(enhance_image_with_dphe(processed_16bit_p4, title_prefix="P4:Sobel->Gamma"))
        dphe_titles.append("Sobel->Gamma->DPHE")

        # --- 路径5: Laplacian滤波 -> 伽马 -> DPHE ---
        print("\n--- Path 5: Laplacian -> Gamma -> DPHE ---")
        laplacian_processed_16bit = apply_laplacian_to_16bit_and_scale(image_16bit_original)
        processed_16bit_p5 = apply_gamma_correction(laplacian_processed_16bit, gamma_value)
        dphe_results.append(enhance_image_with_dphe(processed_16bit_p5, title_prefix="P5:Laplacian->Gamma"))
        dphe_titles.append("Laplacian->Gamma->DPHE")
        
        # --- 显示所有DPHE结果进行比较 ---
        num_results = len(dphe_results)
        fig_compare, axs_compare = plt.subplots(1, num_results, figsize=(5 * num_results, 5))
        if num_results == 1: # 如果只有一个结果，axs_compare不是数组
            axs_compare = [axs_compare]
        fig_compare.suptitle('Comparison of Final 8-bit DPHE Enhanced Images with Different Pre-processing', fontsize=16)
        for i, (img_8bit, title) in enumerate(zip(dphe_results, dphe_titles)):
            axs_compare[i].imshow(img_8bit, cmap='gray', vmin=0, vmax=255)
            axs_compare[i].set_title(title)
            axs_compare[i].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

        # --- 测试: DPHE增强后的图像再进行滤波 ---
        print("\n--- Path 6: (Gamma -> DPHE) then Filters ---")
        base_dphe_image_8bit = dphe_results[0] # 使用路径1的结果作为基础

        post_filtered_results = []
        post_filtered_titles = []

        # 1. DPHE (基准，已在 base_dphe_image_8bit)
        post_filtered_results.append(base_dphe_image_8bit)
        post_filtered_titles.append("DPHE Only (Input)")

        # 2. DPHE -> 最大值滤波
        dphe_then_max = apply_max_filter_8bit(base_dphe_image_8bit, kernel_filter_size)
        post_filtered_results.append(dphe_then_max)
        post_filtered_titles.append("DPHE -> Max Filter")

        # 3. DPHE -> 最小值滤波
        dphe_then_min = apply_min_filter_8bit(base_dphe_image_8bit, kernel_filter_size)
        post_filtered_results.append(dphe_then_min)
        post_filtered_titles.append("DPHE -> Min Filter")
        
        # 4. DPHE -> Sobel滤波
        dphe_then_sobel = apply_sobel_filter_8bit(base_dphe_image_8bit)
        post_filtered_results.append(dphe_then_sobel)
        post_filtered_titles.append("DPHE -> Sobel")

        # 5. DPHE -> Laplacian滤波
        dphe_then_laplacian = apply_laplacian_filter_8bit(base_dphe_image_8bit)
        post_filtered_results.append(dphe_then_laplacian)
        post_filtered_titles.append("DPHE -> Laplacian")

        # 显示 DPHE 后再滤波的结果
        num_post_results = len(post_filtered_results)
        fig_post_compare, axs_post_compare = plt.subplots(1, num_post_results, figsize=(5 * num_post_results, 5))
        if num_post_results == 1: 
            axs_post_compare = [axs_post_compare]
        fig_post_compare.suptitle('Comparison of Filters Applied After DPHE (on 8-bit Image)', fontsize=16)
        for i, (img_8bit, title) in enumerate(zip(post_filtered_results, post_filtered_titles)):
            axs_post_compare[i].imshow(img_8bit, cmap='gray', vmin=0, vmax=255)
            axs_post_compare[i].set_title(title)
            axs_post_compare[i].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

        # --- 可选：保存每个DPHE结果 (以第一个结果为例) ---
        if dphe_results:
            output_jpg_filename = file_path.replace('.raw', '_P1_gamma_dphe_enhanced.jpg')
        try:
                plt.imsave(output_jpg_filename, dphe_results[0], cmap='gray', format='jpg')
                print(f"Saved {dphe_titles[0]} result as: {output_jpg_filename}")
        except Exception as e:
                print(f"Error saving {dphe_titles[0]} result as JPG: {e}")
        
        # 你可以按需取消注释或添加代码来保存其他路径的结果
        #例如保存路径2的结果:
        # if len(dphe_results) > 1:
        #     output_jpg_filename_p2 = file_path.replace('.raw', '_P2_max_gamma_dphe_enhanced.jpg')
        #     try:
        #         plt.imsave(output_jpg_filename_p2, dphe_results[1], cmap='gray', format='jpg')
        #         print(f"Saved {dphe_titles[1]} result as: {output_jpg_filename_p2}")
        #     except Exception as e:
        #         print(f"Error saving {dphe_titles[1]} result as JPG: {e}")
