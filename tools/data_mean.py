# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本用于计算指定数据集中 'style' 子目录下所有图像的蓝色 (B)、绿色 (G) 和红色 (R)
颜色通道的平均值。然后，它会计算一个整体的平均强度（即 B_mean, G_mean, R_mean 的平均值），
并输出这个整体平均强度与每个单独颜色通道平均值之间的差异。

这些输出的差异值 (mean-B_mean, mean-G_mean, mean-R_mean) 可能旨在用于某种形式的
颜色分析或颜色增强预处理步骤，尽管它们在 AnimeGANv2 训练流程中的直接应用
从其他脚本来看并不明显。一种可能的用途是了解风格图像集的整体色偏，
或者作为颜色标准化/增强的参数。

使用方式:
此脚本通过命令行运行，需要指定数据集名称。
例如:
`python tools/data_mean.py --dataset Paprika`

脚本执行后，会打印出计算得到的三个差异值。
"""
import cv2, argparse, os
from glob import glob # 用于查找匹配特定模式的文件路径
from tqdm import tqdm # 用于显示进度条

def parse_args():
    """
    定义并解析命令行参数。

    返回:
    - args (argparse.Namespace): 一个包含所有已解析命令行参数的对象。
    """
    desc = "计算整个数据集中图像 B, G, R 通道均值，并分析其与总体均值的差异。"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Paprika',
                        help='数据集名称 (例如: Paprika, Hayao, Shinkai)。脚本将处理 `dataset/<dataset_name>/style/` 目录下的图像。')

    return parser.parse_args()

def read_img(image_path):
    """
    读取指定路径的图像文件，并计算其 B, G, R 各颜色通道的平均值。
    注意：OpenCV 默认以 BGR 顺序读取图像。

    参数:
    - image_path (str): 图像文件的完整路径。

    返回:
    - tuple: `(B_mean, G_mean, R_mean)`，分别表示该图像蓝色、绿色和红色通道的像素平均值。
             如果图像读取失败或不是3通道图像，则行为未定义（可能出错）。
    """
    img = cv2.imread(image_path) # 以 BGR 格式读取图像
    if img is None: # 检查图像是否成功加载
        raise IOError(f"无法读取图像文件: {image_path}")
    assert len(img.shape) == 3, f"图像 {image_path} 必须是3通道的彩色图像。" # 确保是3通道图像

    # 计算各通道的平均值
    # img[..., 0] 是 B 通道, img[..., 1] 是 G 通道, img[..., 2] 是 R 通道
    B_mean = img[..., 0].mean()
    G_mean = img[..., 1].mean()
    R_mean = img[..., 2].mean()
    return B_mean, G_mean, R_mean

def get_mean(dataset_name):
    """
    计算指定数据集中所有风格图像的 B, G, R 通道平均值，
    然后计算一个整体平均强度，并返回该整体平均强度与各通道平均值之间的差异。

    算法步骤:
    1.  获取 `dataset/<dataset_name>/style/` 目录下所有图像文件的列表。
    2.  初始化 B, G, R 各通道的总和为 0。
    3.  遍历图像列表：
        a.  对每张图像调用 `read_img` 函数，获取其 B, G, R 通道均值。
        b.  将这些均值累加到对应的通道总和中。
    4.  计算整个数据集中 B, G, R 通道的平均值 (B_mean_dataset, G_mean_dataset, R_mean_dataset)。
    5.  计算一个整体的平均强度 `overall_mean = (B_mean_dataset + G_mean_dataset + R_mean_dataset) / 3`。
    6.  返回三个差异值：`(overall_mean - B_mean_dataset, overall_mean - G_mean_dataset, overall_mean - R_mean_dataset)`。

    参数:
    - dataset_name (str): 数据集名称。

    返回:
    - tuple: `(diff_B, diff_G, diff_R)`，表示整体平均强度与各通道平均值的差异。
    """
    # 构建风格图像所在目录的路径
    # os.path.dirname(__file__) 获取当前脚本 (data_mean.py) 所在的目录 (tools/)
    # os.path.dirname(os.path.dirname(__file__)) 获取上级目录 (项目根目录 /app)
    project_root = os.path.dirname(os.path.dirname(__file__))
    style_image_dir = os.path.join(project_root, 'dataset', dataset_name, 'style')

    # 获取目录下所有图像文件的路径列表 (支持多种常见图像格式)
    file_list = glob(os.path.join(style_image_dir, '*.*')) # '*.*' 匹配所有文件，read_img 中会校验

    image_num = len(file_list) # 获取图像总数
    if image_num == 0:
        raise FileNotFoundError(f"在目录 '{style_image_dir}' 中未找到图像文件。")
    print(f'找到图像数量 (image_num): {image_num}')

    # 初始化各通道总值
    B_total = 0.0
    G_total = 0.0
    R_total = 0.0

    processed_count = 0 # 记录成功处理的图像数量
    # 遍历所有图像文件，累加各通道的均值
    for f_path in tqdm(file_list, desc=f"正在处理 {dataset_name} 数据集"):
        try:
            bgr_means_single_image = read_img(f_path) # 获取单张图像的 (B_mean, G_mean, R_mean)
            B_total += bgr_means_single_image[0]
            G_total += bgr_means_single_image[1]
            R_total += bgr_means_single_image[2]
            processed_count += 1
        except (IOError, AssertionError) as e: # 捕获 read_img 中可能抛出的异常
            print(f"\n警告: 处理图像 {f_path} 时出错: {e}，已跳过。")
            # image_num -= 1 # 不再在此处修改 image_num，而是使用 processed_count

    if processed_count == 0: # 如果所有图像都处理失败
        raise ValueError(f"未能成功处理 '{style_image_dir}' 目录中的任何图像。")

    # 计算整个数据集各通道的平均值，基于成功处理的图像数量
    B_mean_dataset = B_total / processed_count
    G_mean_dataset = G_total / processed_count
    R_mean_dataset = R_total / processed_count

    # 计算整体平均强度 (所有通道平均值的平均值)
    overall_intensity_mean = (B_mean_dataset + G_mean_dataset + R_mean_dataset) / 3.0

    # 返回整体平均强度与各通道平均值之间的差异
    return (overall_intensity_mean - B_mean_dataset,
            overall_intensity_mean - G_mean_dataset,
            overall_intensity_mean - R_mean_dataset)

"""main"""
def main():
    """
    主执行函数。
    解析命令行参数并调用 `get_mean` 函数获取结果。

    返回:
    - tuple: 从 `get_mean` 函数返回的颜色通道均值差异。
    """
    # parse arguments
    args = parse_args()
    if args is None: # 如果参数解析失败
        exit() # 退出程序

    return get_mean(args.dataset) # 调用核心函数并返回结果
if __name__ == '__main__':
    """
    脚本主入口点。
    调用 `main()` 函数执行主要逻辑，并打印返回的结果。
    """
    result_differences = main() # 执行计算
    # 打印结果，格式为 (B通道差异, G通道差异, R通道差异)
    print(f'风格数据颜色均值与总体均值的差异 (B, G, R): {result_differences}')