import cv2
import numpy as np
import os
from PIL import Image  # 导入PIL库作为备选保存方式
import matplotlib.pyplot as plt
from scipy import stats # 导入scipy.stats用于计算SRCC和PLCC

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

def apply_optimized_msr_clahe(image_path, sigmas=[10, 60, 200], clip_limit=20.0, tile_grid_size=(8, 8), display_intermediate=False):
    """
    应用优化的 MSR 和 CLAHE
    :param image_path: 16-bit TIR 图像路径 (如 .tiff, .png) 或已加载的 16-bit NumPy 数组
    :param sigmas: MSR 的尺度
    :param clip_limit: CLAHE 的对比度限制因子
    :param tile_grid_size: CLAHE 的网格大小
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
        quit()
        return None

    # 假设 TIR 图像是单通道灰度图
    if len(img_16bit.shape) == 3:
        print("警告: 输入图像似乎是多通道的，将转换为灰度图。")
        img_16bit = cv2.cvtColor(img_16bit, cv2.COLOR_BGR2GRAY) # 或者其他转换方式

    # 将 16-bit 图像归一化到 0-1 的浮点数范围 (MSR 通常在浮点数上操作)
    # Retinex 的输入通常期望是强度值，而不是原始传感器读数。
    # 如果你的 16-bit 数据已经是线性强度，可以直接归一化。
    # 如果不是，可能需要一些预处理，但这超出了简单 MSR 的范畴。
    # 这里我们假设可以直接归一化。
    img_float = img_16bit.astype(np.float32) / 65535.0 # 2^16 - 1

    # 2. 应用 MSR
    msr_log_image = multi_scale_retinex(img_float, sigmas)

    # 3. 将 MSR 输出转换回适合显示的范围 (例如，通过动态范围调整或简单的归一化)
    # MSR 输出的是对数域的值，可能为负。需要映射到 0-255。
    # 一种简单的方法是归一化，但更好的方法是做动态范围调整，如论文中可能隐含的
    # 这里我们先做简单的归一化到 0-1
    msr_output_normalized = cv2.normalize(msr_log_image, None, 0, 1, cv2.NORM_MINMAX)

    # 将归一化后的 MSR 输出转换为 8-bit uint8 图像 (0-255)
    msr_output_8bit = (msr_output_normalized * 255).astype(np.uint8)

    # 显示处理结果
    if display_intermediate:
        """
        cv2.imshow("MSR Output", msr_output_8bit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    # 4. 应用全局 CLAHE
    # OpenCV 的 CLAHE 需要 uint8 输入
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    final_image_8bit = clahe.apply(msr_output_8bit)

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

    if not raw_files:
        print(f"目录 {directory_path} 中没有找到 .raw 文件")
        return
    
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
            
            # 应用优化的 MSR+CLAHE 处理
            # 传递display_intermediate参数控制是否显示中间结果
            processed_img = apply_optimized_msr_clahe(
                img_16bit, 
                sigmas=[20, 100, 240], 
                clip_limit=30.0,
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
                output_jpg_filename = file_path.replace('.raw', f'_MSR_CLAHE_enhanced{metrics_str}.jpg')
                output_png_filename = file_path.replace('.raw', f'_MSR_CLAHE_enhanced{metrics_str}.png')
                
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
        cv2.imshow("Original 16-bit Image (Normalized)", display_original)
        cv2.imshow("Processed MSR+CLAHE", display_processed)
        print("按任意键关闭窗口并继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 完成
    print(f"所有 {len(raw_files)} 个文件处理完成")

# --- 使用示例 ---
if __name__ == "__main__":
    # 要处理的根目录
    root_dir = r'C:\work_space\vscode\Task_Random_HDR\02_data'
    # 用于显示的特定图像
    display_image_path = r'C:\work_space\vscode\Task_Random_HDR\02_data\T10_200mk_3\000.raw'
    
    process_directory_with_raw_images(root_dir, display_image_path)