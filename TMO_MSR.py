import cv2
import numpy as np
import os
from PIL import Image  # 导入PIL库作为备选保存方式
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

def apply_optimized_msr_clahe(image_path, sigmas=[10, 60, 200], clip_limit=20.0, tile_grid_size=(8, 8)):
    """
    应用优化的 MSR 和 CLAHE
    :param image_path: 16-bit TIR 图像路径 (如 .tiff, .png) 或已加载的 16-bit NumPy 数组
    :param sigmas: MSR 的尺度
    :param clip_limit: CLAHE 的对比度限制因子
    :param tile_grid_size: CLAHE 的网格大小
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
    cv2.imshow("MSR Output", msr_output_8bit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 4. 应用全局 CLAHE
    # OpenCV 的 CLAHE 需要 uint8 输入
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    final_image_8bit = clahe.apply(msr_output_8bit)

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

    # 应用优化的 MSR+CLAHE 处理
    processed_img = apply_optimized_msr_clahe(img_16bit, sigmas=[20, 100, 240], clip_limit=30.0)

    if processed_img is not None:
        # 仅用于显示：将16位图像归一化为8位，不影响处理
        img_8bit_for_display = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("Original 16-bit Image (Normalized)", img_8bit_for_display)
        cv2.imshow("Processed MSR+CLAHE", processed_img)

        # --- 保存为 JPG --- 
        output_jpg_filename = file_path.replace('.raw', '_MSR_CLAHE_enhanced.jpg')
        # --- 同时保存为 PNG ---
        output_png_filename = file_path.replace('.raw', '_MSR_CLAHE_enhanced.png')
        try:
            plt.imsave(output_jpg_filename, processed_img, cmap='gray', format='jpg')
            print(f"Enhanced image saved as: {output_jpg_filename}")
            # 保存PNG版本
            plt.imsave(output_png_filename, processed_img, cmap='gray', format='png')
            print(f"Enhanced image also saved as PNG: {output_png_filename}")
        except Exception as e:
            print(f"Error saving JPG: {e}")
        
        print("按任意键关闭窗口并退出程序...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 确保程序正常退出
        print("程序已完成")
    else:
        print("图像处理失败。")