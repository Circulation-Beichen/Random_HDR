import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from scipy.stats import spearmanr, pearsonr

def calculate_metrics(img1, img2):
    """计算两张图片之间的SRCC和PLCC指标"""
    # 确保图像大小相同
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # 将图像转换为一维数组
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    # 计算SRCC (Spearman Rank Correlation Coefficient)
    srcc, _ = spearmanr(img1_flat, img2_flat)
    
    # 计算PLCC (Pearson Linear Correlation Coefficient) 
    plcc, _ = pearsonr(img1_flat, img2_flat)
    
    return srcc, plcc

class ImageQualityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像质量评估工具")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 参考图像选择
        ttk.Label(self.main_frame, text="参考图像:").grid(row=0, column=0, sticky=tk.W)
        self.ref_path = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.ref_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(self.main_frame, text="浏览", command=self.select_ref_image).grid(row=0, column=2)
        
        # 待评估图像选择
        ttk.Label(self.main_frame, text="待评估图像:").grid(row=1, column=0, sticky=tk.W)
        self.test_path = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.test_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(self.main_frame, text="浏览", command=self.select_test_image).grid(row=1, column=2)
        
        # 评估按钮
        ttk.Button(self.main_frame, text="计算评估指标", command=self.evaluate_quality).grid(row=2, column=1, pady=10)
        
        # 结果显示区域
        self.result_text = tk.Text(self.main_frame, height=5, width=50)
        self.result_text.grid(row=3, column=0, columnspan=3, pady=5)

    def select_ref_image(self):
        filename = filedialog.askopenfilename(
            title="选择参考图像",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.raw")]
        )
        self.ref_path.set(filename)

    def select_test_image(self):
        filename = filedialog.askopenfilename(
            title="选择待评估图像",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.raw")]
        )
        self.test_path.set(filename)

    def evaluate_quality(self):
        ref_path = self.ref_path.get()
        test_path = self.test_path.get()
        
        if not ref_path or not test_path:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "错误: 请选择两张图像进行比较\n")
            return
            
        try:
            # 读取图像
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
            
            if ref_img is None or test_img is None:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "错误: 图像读取失败\n")
                return
                
            # 计算指标
            srcc, plcc = calculate_metrics(ref_img, test_img)
            
            # 显示结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"评估结果:\n")
            self.result_text.insert(tk.END, f"SRCC (Spearman Rank Correlation Coefficient): {srcc:.4f}\n")
            self.result_text.insert(tk.END, f"PLCC (Pearson Linear Correlation Coefficient): {plcc:.4f}\n")
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"错误: {str(e)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageQualityGUI(root)
    root.mainloop()
