# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本用于对动漫风格参考图像进行预处理，主要应用一种边缘平滑的滤波技术。
该技术源自 CartoonGAN 项目 (taki0112/CartoonGAN-Tensorflow)。
处理流程包括：
1.  使用 Canny 算子检测图像边缘。
2.  对检测到的边缘进行扩张 (dilation) 操作，使边缘区域变粗。
3.  在原始彩色图像中，仅对扩张后边缘区域内的像素应用高斯模糊。
    这意味着图像的非边缘区域保持不变，而边缘区域则被平滑处理。
处理后的图像会被保存到 `dataset/<dataset_name>/smooth` 目录下，
供 AnimeGANv2 模型训练时使用，以帮助生成器学习生成具有清晰且平滑边缘的动漫效果。

原始脚本来源:
The edge_smooth.py is from taki0112/CartoonGAN-Tensorflow
https://github.com/taki0112/CartoonGAN-Tensorflow#2-do-edge_smooth

使用方式:
通过命令行运行此脚本，需要指定数据集名称和目标图像尺寸。
例如:
`python tools/edge_smooth.py --dataset Hayao --img_size 256`
"""
from tools.utils import check_folder # 导入用于检查和创建文件夹的工具函数
import numpy as np
import cv2, os, argparse
from glob import glob # 用于查找匹配特定模式的文件路径
from tqdm import tqdm # 用于显示进度条

def parse_args():
    """
    定义并解析命令行参数。

    返回:
    - args (argparse.Namespace): 一个包含所有已解析命令行参数的对象。
    """
    desc = "图像边缘平滑处理脚本 (Edge smoothed)" # argparse 解析器的描述信息
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Shinkai',
                        help='数据集名称 (例如: Hayao, Paprika, Shinkai)。脚本会处理 `dataset/<dataset_name>/style/` 目录下的图像。')
    parser.add_argument('--img_size', type=int, default=256,
                        help='图像在处理前将被调整到的目标尺寸 (正方形)。默认: 256。')

    return parser.parse_args()

def make_edge_smooth(dataset_name, img_size) :
    """
    对指定数据集的风格图像进行边缘平滑处理。

    处理流程:
    1.  确定输入 (style) 和输出 (smooth) 目录。
    2.  定义用于边缘扩张和高斯模糊的核。
    3.  遍历输入目录中的每张图像:
        a.  读取彩色图像和灰度图像，并调整到 `img_size`。
        b.  对彩色图像进行反射填充 (padding)，为后续高斯模糊做准备。
        c.  使用 Canny 算子在灰度图像上检测边缘。
        d.  使用定义的核扩张检测到的边缘。
        e.  创建一个原始彩色图像的副本。
        f.  遍历扩张后边缘图中的所有边缘点。
        g.  对于每个边缘点，在原始彩色图像副本的对应位置，使用高斯核对其邻域进行加权平均 (高斯模糊)。
            模糊操作仅应用于边缘点，非边缘点保持不变。
        h.  保存处理后的边缘平滑图像到输出目录。

    参数:
    - dataset_name (str): 数据集名称。
    - img_size (int): 图像的目标处理尺寸。
    """
    # 构建项目根目录路径 (假设此脚本在 tools/ 子目录下)
    project_root = os.path.dirname(os.path.dirname(__file__)) # 即 /app

    # 检查并创建保存平滑后图像的输出目录
    # 路径: dataset/<dataset_name>/smooth
    smooth_output_dir = os.path.join(project_root, 'dataset', dataset_name, 'smooth')
    check_folder(smooth_output_dir)

    # 获取所有待处理的风格图像文件路径列表
    # 路径: dataset/<dataset_name>/style/*.*
    style_image_files = glob(os.path.join(project_root, 'dataset', dataset_name, 'style', '*.*'))

    # 定义边缘扩张核的大小和结构
    kernel_size = 5 # 高斯核和扩张核的尺寸
    # 创建一个 5x5 的全1矩阵作为形态学操作 (如扩张) 的核
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 创建高斯模糊核
    # cv2.getGaussianKernel 生成一维高斯核
    gauss_1d = cv2.getGaussianKernel(kernel_size, 0) # sigma=0 表示根据 kernel_size 自动计算
    # 通过一维高斯核的外积得到二维高斯核
    gauss_2d = gauss_1d * gauss_1d.transpose(1, 0)

    # 使用 tqdm 显示处理进度
    for f_path in tqdm(style_image_files, desc=f"正在处理 {dataset_name} 数据集"):
        file_name = os.path.basename(f_path) # 获取文件名

        # 读取彩色图像 (BGR格式) 和灰度图像
        bgr_img = cv2.imread(f_path)
        gray_img = cv2.imread(f_path, 0) # 0 表示以灰度模式读取

        if bgr_img is None or gray_img is None:
            print(f"\n警告: 无法读取图像 {f_path}，已跳过。")
            continue

        # 将彩色图像和灰度图像都调整到指定尺寸
        bgr_img_resized = cv2.resize(bgr_img, (img_size, img_size))
        gray_img_resized = cv2.resize(gray_img, (img_size, img_size))

        # 对调整尺寸后的彩色图像进行反射填充 (padding)
        # 上下左右各填充 kernel_size//2 (即2个) 像素，以处理边缘像素的高斯模糊
        # mode='reflect' 使用反射模式填充，有助于减少边缘效应
        pad_amount = kernel_size // 2
        pad_img_resized = np.pad(bgr_img_resized, ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), mode='reflect')

        # 使用 Canny 算子检测灰度图像的边缘
        # 参数 100 和 200 分别是低阈值和高阈值
        edges = cv2.Canny(gray_img_resized, 100, 200)

        # 使用定义的 `kernel` 扩张 (dilate) 检测到的边缘。使边缘更粗，以便后续模糊。
        dilation = cv2.dilate(edges, kernel)

        # 创建一个调整尺寸后的彩色图像的副本，用于在其上进行选择性高斯模糊
        gauss_img_output = np.copy(bgr_img_resized)

        # 找到扩张后边缘图中所有非零像素点 (即边缘点) 的坐标
        idx_edges = np.where(dilation != 0)

        # 遍历所有检测到的边缘点
        for i in range(np.sum(dilation != 0)): # np.sum(dilation != 0) 是边缘点的总数
            # 获取当前边缘点的 (行, 列) 坐标
            edge_row, edge_col = idx_edges[0][i], idx_edges[1][i]

            # 对该边缘点在原始彩色图像 (副本 gauss_img_output) 中的对应位置应用高斯模糊
            # 模糊操作是逐通道进行的 (B, G, R)
            # 使用之前填充过的图像 (pad_img_resized) 来提取邻域像素，以正确处理靠近图像边界的边缘点
            # 提取以 (edge_row, edge_col) 为中心，大小为 kernel_size x kernel_size 的邻域
            # 注意：pad_img_resized 的坐标系相对于 bgr_img_resized 有 pad_amount 的偏移

            # B 通道
            neighborhood_b = pad_img_resized[edge_row : edge_row + kernel_size, edge_col : edge_col + kernel_size, 0]
            gauss_img_output[edge_row, edge_col, 0] = np.sum(np.multiply(neighborhood_b, gauss_2d))

            # G 通道
            neighborhood_g = pad_img_resized[edge_row : edge_row + kernel_size, edge_col : edge_col + kernel_size, 1]
            gauss_img_output[edge_row, edge_col, 1] = np.sum(np.multiply(neighborhood_g, gauss_2d))

            # R 通道
            neighborhood_r = pad_img_resized[edge_row : edge_row + kernel_size, edge_col : edge_col + kernel_size, 2]
            gauss_img_output[edge_row, edge_col, 2] = np.sum(np.multiply(neighborhood_r, gauss_2d))

        # 将处理后的边缘平滑图像保存到输出目录
        cv2.imwrite(os.path.join(smooth_output_dir, file_name), gauss_img_output)

def main():
    """
    主执行函数。
    解析命令行参数并调用 `make_edge_smooth` 函数开始处理。
    """
    # 解析命令行参数
    args = parse_args()
    if args is None: # 如果参数解析失败 (例如，提供了无效参数)
        exit() # 退出程序

    # 调用核心处理函数
    make_edge_smooth(args.dataset, args.img_size)
    print(f"\n[*] 数据集 '{args.dataset}' 的边缘平滑处理完成。")

if __name__ == '__main__':
    # 当脚本作为主程序执行时，调用 main() 函数
    main()
