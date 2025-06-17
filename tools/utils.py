# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本提供了一系列工具函数，用于支持 AnimeGANv2 项目中的各种常见任务。
这些任务包括但不限于：加载和预处理测试数据、保存图像、图像变换（如归一化、反归一化、裁剪）、
显示 TensorFlow 模型中的可训练变量信息，以及目录管理。

使用方式:
此文件中的函数通常不直接运行，而是由项目中的其他脚本（例如 `AnimeGANv2.py` 进行模型定义和训练、
`test.py` 进行推理、`train.py` 驱动训练流程、`edge_smooth.py` 进行数据预处理等）导入并按需调用。
"""
import tensorflow as tf
from tensorflow.contrib import slim # 用于模型分析 (slim.model_analyzer)
from tools.adjust_brightness import adjust_brightness_from_src_to_dst, read_img # 导入亮度调整相关函数
import os,cv2
import numpy as np


def load_test_data(image_path, size):
    """
    从指定路径加载一张测试图像，并进行预处理。

    处理步骤:
    1. 读取图像文件 (OpenCV 默认 BGR 格式)。
    2. 将图像从 BGR 转换为 RGB 色彩空间。
    3. 调用 `preprocessing` 函数对图像进行尺寸调整和归一化。
    4. 增加一个批处理维度 (batch dimension)，以符合模型输入的期望格式。

    参数:
    - image_path (str): 图像文件的完整路径。
    - size (list or tuple): 目标图像尺寸 [高度, 宽度]，用于 `preprocessing` 函数。

    返回:
    - numpy.ndarray: 预处理后的图像，形状为 (1, height, width, channels)，数据类型为 float32。
    """
    img = cv2.imread(image_path).astype(np.float32) # 读取图像并转换为 float32 类型
    if img is None: # 检查图像是否成功加载
        raise IOError(f"无法读取图像文件: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    img = preprocessing(img,size) # 应用预处理：尺寸调整和归一化
    img = np.expand_dims(img, axis=0) # 增加批处理维度，例如 (H,W,C) -> (1,H,W,C)
    return img

def preprocessing(img, size):
    """
    对输入图像进行预处理，包括尺寸调整和像素值归一化。

    尺寸调整逻辑:
    - 确保调整后的高度 (h_new) 和宽度 (w_new) 都是 32 的倍数。
    - 如果原始高度/宽度小于指定的 `size` 中的对应值，则使用 `size` 中的值作为基准。
    - 如果原始高度/宽度大于等于 `size` 中的对应值，则将其调整为不大于原始尺寸的32的最大倍数。
    像素值归一化:
    - 将像素值从 [0, 255] 范围归一化到 [-1.0, 1.0] 范围。

    参数:
    - img (numpy.ndarray): 输入图像 (RGB 格式，float32 类型)。
    - size (list or tuple): 期望的最小输出尺寸 [高度, 宽度]。

    返回:
    - numpy.ndarray: 经过尺寸调整和归一化处理的图像。
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

    # 像素值归一化到 [-1.0, 1.0]
    return img_resized / 127.5 - 1.0 # (image / 127.5) -> [0, 2], then -1.0 -> [-1, 1]

def save_images(images, image_path, photo_path = None):
    """
    保存图像。首先对图像进行反归一化处理，然后可以选择性地根据源照片调整亮度，
    最后将图像保存到指定路径。

    参数:
    - images (numpy.ndarray): 要保存的图像数据。通常是模型输出的图像，
                              期望在调用 `squeeze()` 后为 (height, width, channels) 形状。
    - image_path (str): 图像保存的完整路径。
    - photo_path (str, 可选): 源照片的路径。如果提供此参数，
                               将使用此照片作为参考来调整输出图像的亮度。
                               默认: None (不调整亮度)。

    返回:
    - bool: `cv2.imwrite` 的返回值，通常表示保存是否成功。
    """
    # 反归一化图像 (从 [-1,1] 到 [0,255]) 并去除批处理维度 (如果存在)
    fake_image = inverse_transform(images.squeeze())

    if photo_path: # 如果提供了源照片路径，则进行亮度调整
        # adjust_brightness_from_src_to_dst: 从 src (photo_path) 调整 dst (fake_image) 的亮度
        # read_img: 可能是 tools.adjust_brightness 中的一个函数，用于读取参考图像
        adjusted_image = adjust_brightness_from_src_to_dst(fake_image, read_img(photo_path))
        return imsave(adjusted_image, image_path) # 保存调整亮度后的图像
    else: # 否则，直接保存反归一化后的图像
        return imsave(fake_image, image_path)

def inverse_transform(images):
    """
    反归一化图像，将像素值从 [-1.0, 1.0] 范围转换回 [0, 255] 范围，
    并转换为 uint8 数据类型。

    参数:
    - images (numpy.ndarray): 归一化后的图像数据。

    返回:
    - numpy.ndarray: uint8 类型的图像数据，像素值范围为 [0, 255]。
    """
    # (images + 1.0) -> [0, 2.0]
    # / 2.0         -> [0, 1.0]
    # * 255         -> [0, 255]
    images = (images + 1.) / 2 * 255

    # 重要步骤: 裁剪像素值到 [0, 255] 范围。
    # 由于浮点数计算可能存在微小误差，导致某些像素值略微超出此范围 (例如 -0.0001 或 255.0001)。
    # np.clip 操作确保所有像素值都在有效范围内，避免在转换为 uint8 或显示时产生伪影或失真。
    images = np.clip(images, 0, 255)

    return images.astype(np.uint8) # 转换为无符号8位整型


def imsave(images, path):
    """
    使用 OpenCV (`cv2.imwrite`) 保存图像。
    在保存前，将图像从 RGB 色彩空间转换为 BGR 色彩空间，因为 OpenCV 默认使用 BGR 格式。

    参数:
    - images (numpy.ndarray): RGB 格式的图像数据。
    - path (str): 图像保存的完整路径。

    返回:
    - bool: `cv2.imwrite` 的返回值，指示保存是否成功。
    """
    # 将图像从 RGB 转换为 BGR，以适应 OpenCV 的 imwrite 函数
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))

# 定义一个 lambda 函数用于图像裁剪
crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]
"""
图像裁剪 lambda 函数。

参数:
- img (numpy.ndarray): 输入图像。
- x0 (int): 裁剪区域左上角的 x 坐标。
- y0 (int): 裁剪区域左上角的 y 坐标。
- w (int): 裁剪区域的宽度。
- h (int): 裁剪区域的高度。

返回:
- numpy.ndarray: 裁剪后的图像区域。
"""

def random_crop(img1, img2, crop_H, crop_W):
    """
    对两张输入图像（假定它们形状相同且内容对齐）进行相同的随机裁剪。

    参数:
    - img1 (numpy.ndarray): 第一张输入图像。
    - img2 (numpy.ndarray): 第二张输入图像。
    - crop_H (int): 期望裁剪区域的高度。
    - crop_W (int): 期望裁剪区域的宽度。

    返回:
    - tuple: `(crop_1, crop_2)`，包含两张裁剪后的图像。
             如果无法进行有效裁剪（例如，裁剪尺寸大于原图），行为未明确定义，
             但断言确保了 img1 和 img2 形状相同。
    """
    # 确保两张输入图像的形状一致
    assert  img1.shape ==  img2.shape
    h, w = img1.shape[:2] # 获取图像的原始高度和宽度

    # 确保裁剪宽度不超过原始图像宽度
    if crop_W > w:
        crop_W = w
    
    # 确保裁剪高度不超过原始图像高度
    if crop_H > h:
        crop_H = h

    # 随机生成裁剪区域左上角的坐标 (x0, y0)
    # 确保裁剪区域完全在原图内部
    x0 = np.random.randint(0, w - crop_W + 1) # x 坐标范围: [0, w - crop_W]
    y0 = np.random.randint(0, h - crop_H + 1) # y 坐标范围: [0, h - crop_H]

    # 使用定义的 crop_image lambda 函数进行裁剪
    crop_1 = crop_image(img1, x0, y0, crop_W, crop_H)
    crop_2 = crop_image(img2, x0, y0, crop_W, crop_H)
    return crop_1,crop_2


def show_all_variables():
    """
    在 TensorFlow 计算图中显示所有可训练变量的信息。
    主要使用 `tf.contrib.slim.model_analyzer.analyze_vars` 来分析和打印变量的统计信息，
    如名称、形状、参数数量等。
    此函数特别关注 'generator' 作用域下的变量。
    (注意: 判别器 'discriminator' 的分析部分在原代码中被注释掉了。)
    """
    model_vars = tf.trainable_variables() # 获取图中所有可训练的变量

    # slim.model_analyzer.analyze_vars(model_vars, print_info=True) # 原代码中此行被注释，用于分析所有可训练变量

    print('生成器 (Generator) 变量信息:')
    # 筛选出名称以 'generator' 开头的可训练变量，并分析它们
    slim.model_analyzer.analyze_vars([var for var in tf.trainable_variables() if var.name.startswith('generator')], print_info=True)

    # print('判别器 (Discriminator) 变量信息:') # 原代码中判别器分析部分被注释
    # slim.model_analyzer.analyze_vars([var for var in tf.trainable_variables() if var.name.startswith('discriminator')], print_info=True)

def check_folder(log_dir):
    """
    检查指定的目录是否存在，如果不存在，则创建该目录。

    参数:
    - log_dir (str): 需要检查或创建的目录路径。

    返回:
    - str: 传入的目录路径。
    """
    if not os.path.exists(log_dir): # 如果目录不存在
        os.makedirs(log_dir) # 则创建该目录 (包括任何必需的父目录)
    return log_dir

def str2bool(x):
    """
    将字符串转换为布尔值。
    用于处理命令行参数中布尔类型的参数 (例如，argparse 中的 type=str2bool)。
    转换是大小写不敏感的。

    参数:
    - x (str): 输入的字符串，期望是 'true' 或 'false' 的某种变体。

    返回:
    - bool: 如果输入字符串是 'true' (不区分大小写)，则返回 `True`，否则返回 `False`。
    """
    return x.lower() in ('true') # 检查小写后的字符串是否为 'true'
