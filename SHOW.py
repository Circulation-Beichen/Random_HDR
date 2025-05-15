import cv2
import numpy as np
import os
from PIL import Image, ImageTk # 导入PIL库作为备选保存方式 和 Tkinter图像显示
import matplotlib.pyplot as plt
from scipy import stats # 导入scipy.stats用于计算SRCC和PLCC
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

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

# --- 新增：计算图像质量评价指标 ---
def calculate_image_metrics(original_img, enhanced_img):
    """
    计算SRCC和PLCC图像质量评价指标
    :param original_img: 原始图像
    :param enhanced_img: 增强后的图像
    :return: SRCC, PLCC值
    """
    # 确保图像尺寸相同
    if original_img.shape != enhanced_img.shape:
        # 如果增强后的图像是8位但原图是16位，需要将原图转换为8位进行比较
        if original_img.dtype == np.uint16 and enhanced_img.dtype == np.uint8:
            original_img = (original_img / 256).astype(np.uint8)
    
    # 将图像展平为1D数组
    original_flat = original_img.flatten()
    enhanced_flat = enhanced_img.flatten()
    
    # 计算SRCC (Spearman Rank Correlation Coefficient)
    # SRCC评估秩序一致性(单调性)
    srcc, _ = stats.spearmanr(original_flat, enhanced_flat)
    
    # 根据IEEE论文中的公式18应用非线性映射
    # 由于我们没有具体的公式，这里使用一个常见的非线性映射公式
    # 通常可以使用5参数logistic函数: 
    # L(x) = b1*(0.5 - 1/(1+exp(b2*(x-b3)))) + b4*x + b5
    # 但这里简化处理，直接计算线性相关系数
    plcc, _ = stats.pearsonr(original_flat, enhanced_flat)
    
    return srcc, plcc
# --- 计算图像质量评价指标结束 ---

def apply_optimized_msr_dphe(image_path, sigmas=[10, 60, 200], T_UP_factor=0.15, T_DOWN_factor=0.075, display_intermediate=False):
    """
    应用优化的 MSR 和 DPHE
    :param image_path: 16-bit TIR 图像路径 (如 .tiff, .png) 或已加载的 16-bit NumPy 数组
    :param sigmas: MSR 的尺度
    :param T_UP_factor: 计算上阈值的因子
    :param T_DOWN_factor: 计算下阈值的因子
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

    # 显示处理结果
    # 只有当display_intermediate为True时才显示中间结果
    if display_intermediate:
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

def apply_msr_clahe(image_16bit, sigmas=[10, 60, 200], clahe_clip_limit=2.0, clahe_tile_grid_size=(8,8)):
    """
    Applies Multi-Scale Retinex (MSR) followed by CLAHE.
    :param image_16bit: Input 16-bit NumPy array.
    :param sigmas: List of sigmas for MSR.
    :param clahe_clip_limit: Clip limit for CLAHE.
    :param clahe_tile_grid_size: Tile grid size for CLAHE.
    :return: Processed 8-bit image (0-255, uint8), or None if input is invalid.
    """
    if not isinstance(image_16bit, np.ndarray) or image_16bit.dtype != np.uint16:
        print("Error: apply_msr_clahe expects a 16-bit NumPy array.")
        if isinstance(image_16bit, np.ndarray):
            print(f"Received dtype: {image_16bit.dtype}")
        return None

    img_16bit_input = None
    if len(image_16bit.shape) == 3: # Check if it's a 3-dimensional array
        if image_16bit.shape[2] == 1: # (H, W, 1)
            img_16bit_input = image_16bit[:,:,0].copy()
        elif image_16bit.shape[2] == 3: # BGR
            print("Warning: Input image to MSR+CLAHE has 3 channels, converting to grayscale.")
            img_16bit_input = cv2.cvtColor(image_16bit, cv2.COLOR_BGR2GRAY)
        elif image_16bit.shape[2] == 4: # BGRA
            print("Warning: Input image to MSR+CLAHE has 4 channels, converting to grayscale.")
            img_16bit_input = cv2.cvtColor(image_16bit, cv2.COLOR_BGRA2GRAY)
        else: # Other multi-channel format
            print(f"Warning: Input image to MSR+CLAHE has {image_16bit.shape[2]} channels, using the first channel.")
            img_16bit_input = image_16bit[:,:,0].copy()
    elif len(image_16bit.shape) == 2: # (H, W)
        img_16bit_input = image_16bit.copy()
    else:
        print(f"Error: Unsupported image shape for MSR+CLAHE: {image_16bit.shape}")
        return None

    # 1. Apply MSR
    img_float = img_16bit_input.astype(np.float32) / 65535.0
    msr_log_image = multi_scale_retinex(img_float, sigmas) # multi_scale_retinex is already defined

    msr_output_normalized = cv2.normalize(msr_log_image, None, 0, 1, cv2.NORM_MINMAX)
    msr_output_16bit = (msr_output_normalized * 65535.0).astype(np.uint16)

    # 2. Apply CLAHE to the 16-bit MSR result
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    clahe_output_16bit = clahe.apply(msr_output_16bit)

    # 3. Normalize the 16-bit CLAHE output to an 8-bit image
    final_image_8bit = cv2.normalize(clahe_output_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return final_image_8bit

def process_directory_with_raw_images(directory_path, display_image_path=None):
    """
    处理指定目录中的所有 RAW 图像文件
    :param directory_path: 包含 RAW 图像的目录路径
    :param display_image_path: 要显示的特定图像路径
    """
    # 定义可能的尺寸组合
    possible_dimensions = [(640, 512), (1280, 1024)]
    
    # 查找所有.raw文件
    raw_files = []
    for dir_path, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name.lower().endswith('.raw'):
                raw_files.append(os.path.join(dir_path, file_name))
    
    
    print(f"找到 {len(raw_files)} 个 RAW 文件待处理")
    
    # 用于存储要显示的原始和处理后的图像
    display_original = None
    display_processed = None
    
    # 计算不同尺寸下每帧的字节数
    bytes_per_pixel = 2  # 16位 = 2字节/像素
    frame_size_bytes_list = [width * height * bytes_per_pixel for width, height in possible_dimensions]
    
    # 处理每个文件
    for file_path in raw_files:
        try:
            # 判断是否是要显示的图像
            is_display_image = (display_image_path is not None and 
                               file_path.lower() == display_image_path.lower())
            
            # 读取16位原始图像数据
            success = False
            with open(file_path, 'rb') as f:
                data = f.read()
                file_size = len(data)
                
                # 尝试不同的尺寸组合
                for dim_index, (width, height) in enumerate(possible_dimensions):
                    expected_size = width * height * bytes_per_pixel
                    
                    if file_size == expected_size:
                        print(f"使用尺寸: {width}x{height}")
                        img_16bit = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
                        success = True
                        break
                
                if not success:
                    print(f"警告: 文件 {file_path} 大小({file_size}字节)不符合任何预期尺寸，跳过")
                    print(f"预期尺寸: {frame_size_bytes_list}字节")
                    continue
            
            # 应用优化的 MSR+DPHE 处理
            # 传递display_intermediate参数控制是否显示中间结果
            processed_img = apply_optimized_msr_dphe(
                img_16bit, 
                sigmas=[20, 100, 240], 
                T_UP_factor=0.15,
                T_DOWN_factor=0.075,
                display_intermediate=is_display_image  # 只为要显示的图像显示中间处理结果
            )
            
            if processed_img is not None:
                # 如果是要显示的图像，保存原始和处理后的图像供显示
                if is_display_image:
                    display_original = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    display_processed = processed_img.copy()
                
                # 计算SRCC和PLCC值
                srcc, plcc = calculate_image_metrics(img_16bit, processed_img)
                # 将SRCC和PLCC值格式化为字符串(保留3位小数)
                metrics_str = f"_SRCC{srcc:.3f}_PLCC{plcc:.3f}"
                # 在文件名中加入SRCC和PLCC
                output_jpg_filename = file_path.replace('.raw', f'_MSR_DPHE_enhanced{metrics_str}.jpg')
                output_png_filename = file_path.replace('.raw', f'_MSR_DPHE_enhanced{metrics_str}.png')
                
                try:
                    # JPG保存 - 使用matplotlib
                    plt.imsave(output_jpg_filename, processed_img, cmap='gray', format='jpg')
                    
                    # PNG保存 - 使用OpenCV确保单通道
                    cv2.imwrite(output_png_filename, processed_img)
                    
                    print(f"保存图像: {output_jpg_filename} 和 {output_png_filename} (SRCC={srcc:.3f}, PLCC={plcc:.3f})")
                except Exception as e:
                    print(f"保存图像时出错: {e}")
            else:
                print(f"处理文件 {file_path} 失败")
        except Exception as e:
            print(f"处理文件 {file_path} 出错: {e}")
    
    # 显示指定图像的处理结果
    if display_original is not None and display_processed is not None:
        """
        cv2.imshow("Original 16-bit Image (Normalized)", display_original)
        cv2.imshow("Processed MSR+DPHE", display_processed)
        print("按任意键关闭窗口并继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
    
    # 完成
    print(f"所有 {len(raw_files)} 个文件处理完成")

# --- GUI Application ---
class ImageEnhancementGUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Enhancement Tool")
        master.geometry("700x550") # Adjusted size

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.file_path_var = tk.StringVar()
        self.algorithm_var = tk.StringVar(value="MSR+DPHE") # Default algorithm
        self.status_var = tk.StringVar(value="Please select a RAW file and an algorithm.")

        self.img_16bit_original = None
        self.img_processed_8bit = None
        
        # Possible dimensions
        self.possible_dimensions = [(640, 512), (1280, 1024)]
        self.pixel_dtype = np.uint16
        self.bytes_per_pixel = np.dtype(self.pixel_dtype).itemsize


        # --- Layout ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # File Selection Frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="Select RAW File", command=self.select_raw_file).pack(side=tk.LEFT, padx=5)
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=60, state="readonly")
        self.file_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Algorithm Selection Frame
        algo_frame = ttk.LabelFrame(main_frame, text="Algorithm Selection", padding="10")
        algo_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(algo_frame, text="TMO+MSR+DPHE", variable=self.algorithm_var, value="MSR+DPHE").pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text="TMO+MSR+CLAHE", variable=self.algorithm_var, value="MSR+CLAHE").pack(anchor=tk.W)

        # Parameter Frame (Optional - can be added if needed)
        # For now, using default parameters for algorithms

        # Image Display Frame
        display_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        display_frame.pack(expand=True, fill=tk.BOTH, pady=5)

        self.original_label = ttk.Label(display_frame, text="Original Image (16-bit Normalized)")
        self.original_label.grid(row=0, column=0, padx=5, pady=5)
        self.original_canvas = tk.Canvas(display_frame, width=300, height=250, bg="lightgrey")
        self.original_canvas.grid(row=1, column=0, padx=5, pady=5)

        self.processed_label = ttk.Label(display_frame, text="Processed Image (8-bit)")
        self.processed_label.grid(row=0, column=1, padx=5, pady=5)
        self.processed_canvas = tk.Canvas(display_frame, width=300, height=250, bg="lightgrey")
        self.processed_canvas.grid(row=1, column=1, padx=5, pady=5)
        
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)


        # Process Button
        self.process_button = ttk.Button(main_frame, text="Process Image", command=self.start_processing)
        self.process_button.pack(pady=10, fill=tk.X)
        # self.style.configure('Accent.TButton', font=('Helvetica', 10, 'bold')) # If using a custom style

        # Status Bar
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="5")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))

    def select_raw_file(self):
        filename = filedialog.askopenfilename(
            title="Select RAW File",
            filetypes=(("RAW files", "*.raw"), ("All files", "*.*"))
        )
        if filename:
            self.file_path_var.set(filename)
            self.status_var.set(f"Selected: {os.path.basename(filename)}")
            self.load_and_display_original_raw() # Try to display original on selection
        else:
            self.file_path_var.set("")
            self.status_var.set("File selection cancelled.")
            self.clear_canvases()

    def load_raw_image(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            file_size = len(data)
            
            img_16bit = None
            found_dim = None
            for width, height in self.possible_dimensions:
                expected_size = width * height * self.bytes_per_pixel
                if file_size == expected_size:
                    img_16bit = np.frombuffer(data, dtype=self.pixel_dtype).reshape((height, width))
                    found_dim = (width, height)
                    print(f"Using dimensions: {width}x{height}")
                    break
            
            if img_16bit is None:
                messagebox.showerror("Error", f"File size {file_size} does not match expected dimensions for {self.possible_dimensions}.")
                return None, None
            return img_16bit, found_dim

        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {file_path}")
            return None, None
        except Exception as e:
            messagebox.showerror("Error", f"Error reading RAW file: {e}")
            return None, None
            
    def load_and_display_original_raw(self):
        file_path = self.file_path_var.get()
        if not file_path:
            return

        self.img_16bit_original, dims = self.load_raw_image(file_path)
        if self.img_16bit_original is not None:
            # Normalize for display
            img_norm_8bit = cv2.normalize(self.img_16bit_original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.display_image_on_canvas(img_norm_8bit, self.original_canvas)
            self.status_var.set(f"Loaded: {os.path.basename(file_path)} ({dims[0]}x{dims[1]})")
        else:
            self.clear_canvases()

    def display_image_on_canvas(self, img_array_8bit, canvas):
        canvas.delete("all")
        if img_array_8bit is None:
            return
        
        # Rotate the image by 180 degrees
        img_array_8bit = cv2.rotate(img_array_8bit, cv2.ROTATE_180)
        
        img_pil = Image.fromarray(img_array_8bit)
        
        # Resize to fit canvas while maintaining aspect ratio
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 2 or canvas_height < 2: # Canvas not yet realized
            # Try to force update to get dimensions, or use default
            canvas.update_idletasks() 
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            if canvas_width < 2 or canvas_height < 2:
                 canvas_width, canvas_height = 300, 250 # Fallback Default

        img_aspect = img_pil.width / img_pil.height
        canvas_aspect = canvas_width / canvas_height

        if img_aspect > canvas_aspect: # Image is wider than canvas
            new_width = canvas_width
            new_height = int(new_width / img_aspect)
        else: # Image is taller or same aspect
            new_height = canvas_height
            new_width = int(new_height * img_aspect)
        
        # Ensure new_width and new_height are at least 1
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        img_pil_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)
        
        try:
            # Keep a reference to the PhotoImage object to prevent it from being garbage collected
            # Store it as an attribute of the canvas or the class instance
            if canvas == self.original_canvas:
                self.original_photo_image = ImageTk.PhotoImage(image=img_pil_resized)
                canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.original_photo_image)
            elif canvas == self.processed_canvas:
                self.processed_photo_image = ImageTk.PhotoImage(image=img_pil_resized)
                canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.processed_photo_image)

        except Exception as e:
            print(f"Error creating PhotoImage: {e}")
            messagebox.showerror("Display Error", f"Could not display image: {e}")


    def clear_canvases(self):
        self.original_canvas.delete("all")
        self.processed_canvas.delete("all")
        self.img_16bit_original = None
        self.img_processed_8bit = None
        # Clear PhotoImage references too
        self.original_photo_image = None 
        self.processed_photo_image = None


    def start_processing(self):
        if self.img_16bit_original is None:
            messagebox.showwarning("Warning", "Please select and load a RAW file first.")
            return

        selected_algorithm = self.algorithm_var.get()
        self.status_var.set(f"Processing with {selected_algorithm}...")
        self.master.update_idletasks()

        try:
            if selected_algorithm == "MSR+DPHE":
                self.img_processed_8bit = apply_optimized_msr_dphe(self.img_16bit_original.copy())
            elif selected_algorithm == "MSR+CLAHE":
                self.img_processed_8bit = apply_msr_clahe(self.img_16bit_original.copy())
            else:
                messagebox.showerror("Error", "Invalid algorithm selected.")
                self.status_var.set("Algorithm error.")
                return

            if self.img_processed_8bit is not None:
                self.display_image_on_canvas(self.img_processed_8bit, self.processed_canvas)
                
                # Calculate metrics
                srcc, plcc = calculate_image_metrics(self.img_16bit_original, self.img_processed_8bit)
                self.status_var.set(f"Processed with {selected_algorithm}. SRCC: {srcc:.3f}, PLCC: {plcc:.3f}")
                
                # Show Matplotlib plots for histograms (optional, can be annoying with GUI)
                self.show_histograms_matplotlib()
            else:
                messagebox.showerror("Error", "Processing failed to produce an image.")
                self.status_var.set("Processing failed.")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            self.status_var.set(f"Error: {e}")
            print(f"Full traceback for processing error: {traceback.format_exc()}") # For more detailed debugging
            
    def show_histograms_matplotlib(self):
        if self.img_16bit_original is None or self.img_processed_8bit is None:
            return

        # Conditional import of traceback if needed for error reporting
        import traceback 

        plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for better plot appearance
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original Image Histogram
        try:
            min_val_orig = np.min(self.img_16bit_original)
            max_val_orig = np.max(self.img_16bit_original)
            hist_orig, bins_orig = np.histogram(self.img_16bit_original.flatten(), bins=256, range=[min_val_orig, max_val_orig + (1 if max_val_orig == min_val_orig else 0)])
            axes[0].plot(bins_orig[:-1], hist_orig, lw=1, color='blue')
            axes[0].set_title('Original 16-bit Histogram (256 bins)')
            axes[0].set_xlabel(f'Gray Level (Range: {min_val_orig}-{max_val_orig})')
            axes[0].set_ylabel('Pixel Count')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            axes[0].set_yscale('log')
        except Exception as e_hist_orig:
            print(f"Error plotting original histogram: {e_hist_orig}")
            axes[0].set_title('Original Histogram Error')

        # Processed Image Histogram
        try:
            hist_processed, bins_processed = np.histogram(self.img_processed_8bit.flatten(), bins=256, range=[0, 256])
            axes[1].plot(bins_processed[:-1], hist_processed, lw=1, color='red')
            axes[1].set_title('Processed 8-bit Histogram')
            axes[1].set_xlabel('Gray Level (0-255)')
            axes[1].set_ylabel('Pixel Count')
            axes[1].grid(True, linestyle='--', alpha=0.7)
        except Exception as e_hist_proc:
            print(f"Error plotting processed histogram: {e_hist_proc}")
            axes[1].set_title('Processed Histogram Error')
            
        fig.suptitle("Image Histograms", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plt.show(block=False) # Non-blocking display

if __name__ == "__main__":
    root = tk.Tk()
    # It's generally better to set font globally if needed for specific languages
    # For English, default fonts are usually fine.
    # Example for a specific font (if available on system):
    # import tkinter.font as tkFont # Moved import here
    # default_font = tkFont.nametofont("TkDefaultFont")
    # default_font.configure(family="Arial", size=10)
    # root.option_add("*Font", default_font)
    
    app = ImageEnhancementGUI(root)
    root.mainloop()
    # Removed any residual calls to process_directory_with_raw_images or similar batch processing logic.
    # The script should now only start the GUI and wait for user interaction.
