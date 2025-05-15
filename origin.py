import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

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
    
    # 计算不同尺寸下每帧的字节数
    bytes_per_pixel = 2  # 16位 = 2字节/像素
    frame_size_bytes_list = [width * height * bytes_per_pixel for width, height in possible_dimensions]
    
    # 处理每个文件
    for file_path in raw_files:
        try:
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

            # 直接16位转8位，不做额外处理
            processed_img = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            if processed_img is not None:
                # 保存处理后的图像
                output_filename = file_path.replace('.raw', '_origin.png')
                
                try:
                    # 保存为PNG
                    cv2.imwrite(output_filename, processed_img)
                    print(f"保存图像: {output_filename}")
                except Exception as e:
                    print(f"保存图像时出错: {e}")
            else:
                print(f"处理文件 {file_path} 失败")
        except Exception as e:
            print(f"处理文件 {file_path} 出错: {e}")
    
    # 完成
    print(f"所有 {len(raw_files)} 个文件处理完成")

# --- 使用示例 ---
if __name__ == "__main__":
    # 要处理的根目录
    root_dir = r'C:\work_space\vscode\Task_Random_HDR\02_data'
    # 用于显示的特定图像
    display_image_path = r'C:\work_space\vscode\Task_Random_HDR\02_data\T10_200mk_3\000.raw'
    
    # 处理指定目录中的RAW文件
    process_directory_with_raw_images(root_dir, display_image_path)
