# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本用于将两张图像（通常是一张原始输入图像和其对应的动漫风格转换结果）横向拼接在一起，
以便于对比查看。脚本在拼接前会对原始图像进行尺寸调整预处理，并在两张图像之间添加一条白色的垂直分隔线。
处理后的拼接图像会保存到指定的输出目录。

根据脚本中当前的硬编码路径，它被设置为处理 `../dataset/test/HR_photo` 目录下的图像，
并将其与 `../results/Hayao/HR_photo` 目录下的对应风格化图像进行拼接，
最终结果保存到 `../results/Hayao/concat/` 目录中。

使用方式:
此脚本设计为直接通过命令行运行：
`python tools/concat.py`

注意:
脚本中的输入和输出路径是硬编码的。如果需要处理不同的数据集、风格或目录结构，
用户需要直接修改脚本中的路径变量。
例如，修改 `dirpath` 的初始值和 `style` (输出目录) 以及 `img_path2` 的构造逻辑。
"""
import os
import cv2
import numpy as np
from tqdm import tqdm # 用于显示处理进度条

def check_folder(log_dir):
    """
    检查指定的目录是否存在，如果不存在，则创建该目录。
    (此函数与 `tools.utils.py` 中的 `check_folder` 功能相同。)

    参数:
    - log_dir (str): 需要检查或创建的目录路径。

    返回:
    - str: 传入的目录路径。
    """
    if not os.path.exists(log_dir): # 如果目录不存在
        os.makedirs(log_dir) # 则创建该目录 (包括任何必需的父目录)
        print(f"[*] 目录 '{log_dir}' 已创建。")
    return log_dir

def preprocessing(img, size=[256,256]):
    """
    对输入图像进行尺寸调整预处理。
    确保调整后的高度和宽度都是32的倍数，并且不小于指定的 `size`。
    (注意: 此版本的 `preprocessing` 与 `tools.utils.py` 中的版本类似，
     但不包含像素值归一化到 [-1,1] 的步骤，图像仍保持在 [0,255] 范围。)

    参数:
    - img (numpy.ndarray): 输入图像 (BGR 格式，来自 cv2.imread)。
    - size (list or tuple, 可选): 期望的最小输出尺寸 [高度, 宽度]。默认: [256, 256]。

    返回:
    - numpy.ndarray: 经过尺寸调整的图像。
    """
    h, w = img.shape[:2] # 获取图像的原始高度和宽度

    # 调整高度 (h_new)
    if h <= size[0]: # 如果原始高度小于等于期望最小高度
        h_new = size[0]
    else: # 否则，调整为不大于原始高度的32的最大倍数
        x = h % 32
        h_new = h - x

    # 调整宽度 (w_new)
    if w < size[1]: # 如果原始宽度小于期望最小宽度
        w_new = size[1]
    else: # 否则，调整为不大于原始宽度的32的最大倍数
        y = w % 32
        w_new = w - y

    # 使用 OpenCV调整图像尺寸。注意：cv2.resize 的 dsize 参数格式是 (宽度, 高度)。
    img_resized = cv2.resize(img, (w_new, h_new))
    return img_resized

if __name__ == '__main__':
    """
    脚本主执行块。

    逻辑说明:
    1.  硬编码输入目录 (`root_input_dir`) 和输出目录 (`output_style_dir`)。
        -   `root_input_dir`: 指向原始图像的根目录，当前为 `../dataset/test/HR_photo`。
        -   `output_style_dir`: 指向拼接后图像的保存目录，当前为 `../results/Hayao/concat/`。
    2.  使用 `os.walk` 遍历 `root_input_dir` 下的所有文件。
    3.  对于每个找到的图像文件 (`filepath`):
        a.  构造原始图像的完整路径 (`img_path1`)。
        b.  通过替换路径中的部分字符串，构造对应的风格化图像的路径 (`img_path2`)。
            例如，将 `dataset/test/` 替换为 `results/Hayao/` 来定位风格化后的图像。
            这部分逻辑高度依赖于项目固定的目录结构。
        c.  使用 `cv2.imread` 读取原始图像 (`img1`) 和风格化图像 (`img2`)。
        d.  对原始图像 `img1` 调用本地的 `preprocessing` 函数进行尺寸调整。
        e.  断言 (`assert`) 确保调整后的 `img1` 和读取的 `img2` 具有相同的形状。
            如果形状不匹配，意味着预期的对应图像未找到或尺寸不一致，程序会中断。
        f.  创建一条宽度为7像素的白色垂直分隔线 (`separator_strip`)。
        g.  使用 `np.concatenate` 将 `img1`、白色分隔线和 `img2` 沿水平方向 (axis=1) 拼接起来。
        h.  使用 `cv2.imwrite` 将拼接后的图像保存到 `output_style_dir` 目录中，
            文件名按处理顺序编号 (例如 `1.jpg`, `2.jpg`, ...)。
    4.  使用 `tqdm` 显示处理进度。
    """
    # 硬编码的输入图像根目录 (原始照片)
    # 注意：路径是相对于当前脚本 (tools目录) 的相对路径。
    # `os.path.abspath` 可用于转换为绝对路径，以增加稳健性。
    root_input_dir = '../dataset/test/HR_photo'
    # 硬编码的输出目录 (拼接后的图像) 和风格名称 (用于构造风格化图像路径)
    style_name_for_results = 'Hayao' # 例如 'Hayao', 'Paprika', 'Shinkai'
    output_style_dir = f'../results/{style_name_for_results}/concat/'

    # 检查并创建输出目录
    check_folder(output_style_dir)

    # 使用 os.walk 遍历输入目录
    # dirpath: 当前目录的路径
    # dirnames: 当前目录下的子目录列表 (在此脚本中未使用)
    # filenames: 当前目录下的文件列表
    for dirpath, dirnames, filenames in os.walk(root_input_dir):

        file_count_in_dir = len(filenames) # 当前目录下文件数量
        print(f"在目录 '{dirpath}' 中找到 {file_count_in_dir} 个文件。")

        # 使用 tqdm 包装 filenames 列表以显示进度条
        for i, filename in enumerate(tqdm(filenames, desc=f"处理 {os.path.basename(dirpath)} 中的图像")):
            # 构造原始图像的完整路径
            img_path1 = os.path.join(dirpath, filename)

            # 构造对应的风格化图像的路径
            # 这是一个关键的路径替换，高度依赖于项目的目录结构。
            # 它假设原始图像在 'dataset/test/' 下，而风格化结果在 'results/<style_name_for_results>/' 下，
            # 且子目录结构保持一致。
            # 例如: ../dataset/test/HR_photo/image.png -> ../results/Hayao/HR_photo/image.png
            styled_image_dirpath = dirpath.replace('dataset/test/', f'results/{style_name_for_results}/')
            img_path2 = os.path.join(styled_image_dirpath, filename)

            # 读取原始图像 (img1) 和风格化图像 (img2)
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)

            # 检查图像是否成功加载
            if img1 is None:
                print(f"\n警告: 无法读取原始图像 {img_path1}，已跳过。")
                continue
            if img2 is None:
                print(f"\n警告: 无法读取风格化图像 {img_path2}，已跳过。")
                continue

            # 对原始图像 img1 进行预处理 (尺寸调整)
            img1_processed = preprocessing(img1) # 注意：img2 期望已是目标尺寸

            # 断言：确保两张图像具有相同的形状，以便拼接
            # 如果此断言失败，通常意味着 img_path2 未找到对应的已处理图像，
            # 或者 img1 的预处理逻辑与 img2 的生成尺寸不符。
            if img1_processed.shape != img2.shape:
                print(f"\n警告: 图像形状不匹配，跳过拼接。")
                print(f"  原始图像 (处理后) {os.path.basename(img_path1)}: {img1_processed.shape}")
                print(f"  风格图像 {os.path.basename(img_path2)}: {img2.shape}")
                continue

            h, w, c = img1_processed.shape # 获取处理后图像的高度、宽度和通道数

            # 创建一条白色的垂直分隔线
            # 高度与图像相同，宽度为7像素，3个颜色通道，数据类型为 uint8，像素值为255 (白色)
            separator_strip = np.ones((h, 7, 3), dtype='uint8') * 255

            # 第一次拼接：将处理后的原始图像 (img1_processed) 和白色分隔线水平拼接
            concatenated_img_part1 = np.concatenate([img1_processed, separator_strip], axis=1) # axis=1 表示水平拼接
            # 第二次拼接：将上面的结果和风格化图像 (img2) 水平拼接
            final_concatenated_img = np.concatenate([concatenated_img_part1, img2], axis=1)

            # 构建输出文件名并保存拼接后的图像
            # 使用 enumerate 的索引 i (从0开始) + 1 作为文件名编号
            output_filename = f"{i+1}.jpg" # 保存为 jpg 格式
            cv2.imwrite(os.path.join(output_style_dir, output_filename), final_concatenated_img)

    print(f"\n[*] 图像拼接完成，结果已保存到 '{output_style_dir}' 目录。")
