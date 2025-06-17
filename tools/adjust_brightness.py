# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本提供了一系列函数，用于将目标图像的亮度调整到与源图像的平均亮度相匹配的水平。
它包含了读取图像、计算图像平均亮度的辅助函数。
这个功能主要被项目中的其他工具（例如 `tools.utils.save_images`）调用，
以便在保存生成的动漫风格图像时，可以选择性地将其亮度与原始输入照片的亮度进行匹配，
从而使得风格转换后的图像在整体亮度上更接近原始照片。

使用方式:
主要的亮度调整功能由 `adjust_brightness_from_src_to_dst` 函数提供，
该函数可以被其他脚本导入并调用。
脚本末尾的 `if __name__ == '__main__':` 代码块提供了一个简单的使用示例，
演示了如何使用此模块中的函数来读取两张图像并调整其中一张的亮度以匹配另一张。
（注意：示例中使用的图像路径如 `../Brightness_tool/A.png` 是相对路径，
可能需要根据实际的文件结构进行调整才能成功运行该示例。）
"""
import numpy as np
import cv2

def read_img(image_path):
    """
    从指定路径读取图像文件。

    处理步骤:
    1. 使用 OpenCV (`cv2.imread`) 读取图像。OpenCV 默认以 BGR 格式加载。
    2. 将图像从 BGR 色彩空间转换为 RGB 色彩空间。
    3. 断言确保图像是3通道的 (即彩色图像)。

    参数:
    - image_path (str): 图像文件的完整路径。

    返回:
    - numpy.ndarray: RGB 格式的图像数据。数据类型通常是 `uint8` (0-255 范围)，
                     但如果 OpenCV 读取失败，则返回 None。
    """
    img = cv2.imread(image_path) # 以 BGR 格式读取图像
    if img is None: # 检查图像是否成功加载
        raise IOError(f"无法读取图像文件: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 将 BGR 转换为 RGB
    assert len(img.shape) == 3, "图像必须是3通道的彩色图像" # 确保是3通道图像
    return img


def calculate_average_brightness(img):
    """
    计算图像的平均感知亮度和各颜色通道的平均值。

    平均感知亮度根据标准的人眼亮度感知公式计算：
    Brightness = 0.299 * R + 0.587 * G + 0.114 * B

    参数:
    - img (numpy.ndarray): 输入图像 (假定为 RGB 格式，像素值范围 0-255)。

    返回:
    - tuple: `(brightness, B_mean, G_mean, R_mean)`
        - `brightness` (float): 图像的平均感知亮度。
        - `B_mean` (float): 蓝色通道的平均值。
        - `G_mean` (float): 绿色通道的平均值。
        - `R_mean` (float): 红色通道的平均值。
    """
    # 计算 R, G, B 各通道的平均值
    R_mean = img[..., 0].mean() # 红色通道 (索引0)
    G_mean = img[..., 1].mean() # 绿色通道 (索引1)
    B_mean = img[..., 2].mean() # 蓝色通道 (索引2)

    # 根据标准亮度公式计算平均感知亮度
    brightness = 0.299 * R_mean + 0.587 * G_mean + 0.114 * B_mean
    return brightness, B_mean, G_mean, R_mean # 注意返回顺序与内部计算变量名顺序可能不同

def adjust_brightness_from_src_to_dst(dst, src, path=None, if_show=None, if_info=None):
    """
    调整目标图像 (`dst`) 的亮度，使其平均感知亮度与源图像 (`src`) 的平均感知亮度相匹配。

    参数:
    - dst (numpy.ndarray): 目标图像 (RGB格式, 0-255范围)，其亮度将被调整。
    - src (numpy.ndarray): 源图像 (RGB格式, 0-255范围)，作为亮度调整的参考。
    - path (str, 可选): 如果提供，则将拼接后的对比图像 (原始dst, src, 调整后的dstf) 保存到此路径。
                         默认: None (不保存对比图像)。
    - if_show (bool, 可选): 如果为 True，则使用 `cv2.imshow` 显示拼接后的对比图像。
                            默认: None (不显示)。
    - if_info (bool, 可选): 如果为 True，则打印源图像、目标图像以及计算出的亮度差异信息。
                             默认: None (不打印信息)。

    算法步骤:
    1. 计算源图像 (`src`) 和目标图像 (`dst`) 的平均感知亮度。
    2. 计算亮度差异比率 (`brightness_difference = brightness_src / brightness_dst`)。
    3. 将目标图像 (`dst`) 的每个像素值乘以该差异比率，以调整其整体亮度。
       (脚本中还注释掉了另一种逐通道调整亮度的方法。)
    4. 使用 `np.clip` 将调整后的像素值限制在 [0, 255] 范围内，以防止溢出。
    5. 将结果转换为 `uint8` 数据类型。
    6. (可选) 创建一个包含原始目标图像、源图像和调整后目标图像的拼接图像，用于显示或保存。

    返回:
    - numpy.ndarray: 经过亮度调整的目标图像 (`dstf`)，数据类型为 `uint8`。
    """
    # 计算源图像和目标图像的平均亮度及各通道均值
    brightness1, B1, G1, R1 = calculate_average_brightness(src)  # 源图像的亮度信息
    brightness2, B2, G2, R2 = calculate_average_brightness(dst)  # 目标图像的亮度信息

    # 计算亮度差异比率
    if brightness2 == 0: # 防止除以零错误
        brightness_difference = 1.0 # 如果目标图像全黑，则不调整
    else:
        brightness_difference = brightness1 / brightness2

    if if_info: # 如果需要打印信息
        print(f'源图像平均亮度 (Average brightness of source image): {brightness1:.2f}')
        print(f'目标图像平均亮度 (Average brightness of target): {brightness2:.2f}')
        print(f'亮度差异比率 (Brightness Difference Ratio): {brightness_difference:.4f}')

    # --- 主要亮度调整方法：根据平均感知亮度进行整体缩放 ---
    # 将目标图像的每个像素值乘以亮度差异比率
    # 注意：这里操作的是浮点数类型的图像副本，以避免精度损失
    dstf = dst.astype(np.float32) * brightness_difference

    # --- 备选方法 (已注释掉)：根据各颜色通道的平均值进行逐通道缩放 ---
    # dstf = dst.copy().astype(np.float32)
    # if B2 != 0: dstf[..., 2] = dst[..., 2] * (B1 / B2) # Blue channel (BGR顺序的索引2) - 假设dst是RGB,则索引应为2
    # if G2 != 0: dstf[..., 1] = dst[..., 1] * (G1 / G2) # Green channel
    # if R2 != 0: dstf[..., 0] = dst[..., 0] * (R1 / R2) # Red channel - 假设dst是RGB,则索引应为0
    # (注意：原注释中 dst[...,0] 对应 B1/B2，可能基于 BGR 假设，但函数输入通常是 RGB)
    # (如果 dst 是 RGB 格式，则 dst[...,0] 是 R, dst[...,1] 是 G, dst[...,2] 是 B)

    # 将调整后的像素值限制在 [0, 255] 范围内，并转换为 uint8 类型
    # np.clip 对于超出范围的值，会将其设为边界值。
    dstf = np.clip(dstf, 0, 255)
    dstf = np.uint8(dstf) # 转换为无符号8位整型

    # --- 创建并处理用于显示或保存的对比图像 ---
    # 获取源图像和目标图像的尺寸
    ma, na, _ = src.shape  # (height_src, width_src, channels)
    mb, nb, _ = dst.shape  # (height_dst, width_dst, channels)

    # 创建一个足够大的画布以容纳三张图像并排显示
    # 高度取两者最大值，宽度为三者最大宽度之和（这里假设目标图和调整后目标图宽度相同）
    result_show_img = np.zeros((max(ma, mb), nb + na + nb, 3), dtype=np.uint8) # 初始化为黑色背景

    # 将原始目标图像、源图像、调整后的目标图像依次放入画布
    result_show_img[:mb, :nb, :] = dst # 左: 原始目标图像
    result_show_img[:ma, nb:nb + na, :] = src # 中: 源图像 (亮度参考)
    result_show_img[:mb, nb + na:nb + na + nb, :] = dstf # 右: 调整亮度后的目标图像

    # result_show_img = result_show_img.astype(np.uint8) # 已在初始化时设为 uint8

    if if_show: # 如果需要显示图像
        # 注意：cv2.imshow 显示的是 BGR 格式，而 result_show_img 是 RGB 格式
        cv2.imshow('Comparison (Original_Dst | Source_Ref | Adjusted_Dst)', cv2.cvtColor(result_show_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0) # 等待按键
        cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口

    if path is not None: # 如果提供了保存路径
        # 注意：cv2.imwrite 保存的是 BGR 格式
        cv2.imwrite(path, cv2.cvtColor(result_show_img, cv2.COLOR_RGB2BGR))
        print(f"对比图像已保存到: {path}")

    return dstf # 返回调整亮度后的目标图像

if __name__ == '__main__':
    """
    主执行块，用于演示 `adjust_brightness_from_src_to_dst` 函数的功能。
    它读取两张示例图像 A.png 和 B.png，然后将图像 A 的亮度调整为与图像 B 的平均亮度相匹配。
    注意：示例中的路径 `../Brightness_tool/A.png` 和 `../Brightness_tool/B.png`
    是相对路径，可能需要根据实际的文件存放位置进行修改才能正确运行。
    """
    # 定义示例图像的路径 (需要确保这些文件存在于指定位置)
    # 假设此脚本位于 tools/ 目录，那么 ../Brightness_tool/ 指向项目根目录下的 Brightness_tool/ 文件夹
    path_A = '../Brightness_tool/A.png'
    path_B = '../Brightness_tool/B.png'

    try:
        # 读取图像 A (目标图像，将被调整)
        image_A = read_img(path_A)
        # 读取图像 B (源图像，作为亮度参考)
        image_B = read_img(path_B)

        print("正在调整图像 A 的亮度以匹配图像 B...")
        # 调用亮度调整函数
        # adjusted_A = adjust_brightness_from_src_to_dst(image_A, image_B, path='resA_adjusted_to_B.png', if_show=True, if_info=True)
        # 简化调用，不保存对比图，不显示，不打印信息
        adjusted_A = adjust_brightness_from_src_to_dst(image_A, image_B, if_info=True)
        print(f"亮度调整完成。调整后的图像 A (未保存) 的数据类型: {adjusted_A.dtype}, 形状: {adjusted_A.shape}")

        # 可以选择性地保存调整后的图像 A
        # cv2.imwrite('../Brightness_tool/A_adjusted.png', cv2.cvtColor(adjusted_A, cv2.COLOR_RGB2BGR))
        # print("调整后的图像 A 已保存为 'A_adjusted.png'")

    except IOError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")
