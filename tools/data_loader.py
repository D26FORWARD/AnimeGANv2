# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本定义了 `ImageGenerator` 类，该类负责为 AnimeGANv2 模型加载、预处理和批量化图像数据。
它利用 `tf.data.Dataset` API 来创建一个高效的数据输入管道，支持多线程处理，
从而在模型训练期间加速数据供给。

使用方式:
此文件不直接运行。`ImageGenerator` 类在 `AnimeGANv2.py` 脚本中被实例化，
用于为真实照片、动漫风格图像以及边缘平滑后的动漫图像创建各自的数据迭代器。
这些迭代器随后在训练循环中被用来获取成批的图像数据。
"""
import os
import tensorflow as tf
import cv2,random # random 模块在此文件中未被使用
import numpy as np

class ImageGenerator(object):
    """
    图像数据生成器类。
    管理从指定目录加载图像，并将其组织成批次以供模型训练。
    使用 TensorFlow 的 `tf.data.Dataset` API 实现高效的数据流水线。
    """
    def __init__(self, image_dir, size, batch_size, num_cpus = 16):
        """
        初始化 ImageGenerator。

        参数:
        - image_dir (str): 包含图像文件的目录路径。
        - size (list/tuple): 目标图像尺寸 [高度, 宽度]。
                             注意: 在当前 `load_image` 和 `read_image` 方法的实现中，
                             这个 `size` 参数似乎并没有被直接用于调整图像大小。
                             图像的缩放可能在其他地方处理，或者此参数是为未来扩展或
                             与其他工具函数 (如 `tools.utils.load_test_data`) 配合使用而保留的。
                             在此类内部，主要进行的是色彩空间转换和归一化。
        - batch_size (int): 每个批次中的图像数量。
        - num_cpus (int, 可选): 用于并行处理数据的 CPU 核心数量。
                                 通过 `tf.data.Dataset.map` 的 `num_parallel_calls` 参数使用。
                                 默认: 16。
        """
        # 获取指定目录下所有有效图像文件的路径列表
        self.paths = self.get_image_paths_train(image_dir)
        # 数据集中图像的总数
        self.num_images = len(self.paths)
        # 用于并行数据处理的 CPU 核心数
        self.num_cpus = num_cpus
        # 目标图像尺寸 (当前类中未直接用于缩放)
        self.size = size
        # 每批图像数量
        self.batch_size = batch_size

    def get_image_paths_train(self, image_dir):
        """
        扫描指定的图像目录，收集所有有效图像文件的完整路径。
        有效的图像文件扩展名包括 'jpg', 'jpeg', 'png', 'gif'。

        参数:
        - image_dir (str): 要扫描的图像目录路径。

        返回:
        - list: 包含所有有效图像文件完整路径的列表。
        """
        paths = [] # 初始化路径列表
        for path in os.listdir(image_dir): # 遍历目录中的所有文件和文件夹
            # 检查文件扩展名是否为支持的图像格式
            if path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'gif']: # 转为小写以兼容大写扩展名
                continue # 如果不是，则跳过此文件

            # 构建图像文件的完整路径
            path_full = os.path.join(image_dir, path)

            # 确认路径指向的是一个文件 (而不是目录)
            if not os.path.isfile(path_full):
                continue # 如果不是文件，则跳过

            paths.append(path_full) # 将有效图像路径添加到列表中
        return paths

    def read_image(self, img_path1_tensor):
        """
        读取并初步处理单个图像文件。
        此方法被 `tf.py_func` 调用，因此接收的是一个 TensorFlow 张量。

        根据图像路径中是否包含 'style' 或 'smooth' 字符串，进行不同的处理：
        - 如果是 'style' 或 'smooth' 图像：
            - `image1`: 读取彩色图像，并转换为 RGB 格式。
            - `image2`: 读取同一图像的灰度版本，并复制为3通道图像 (模拟 RGB)。
        - 如果是其他类型图像 (通常是真实照片 `train_photo`):
            - `image1`: 读取彩色图像，并转换为 RGB 格式。
            - `image2`: 创建一个与 `image1` 形状相同但内容全为零的数组。
                        这可能用作占位符，或在某些损失计算中被特定方式处理。

        参数:
        - img_path1_tensor (tf.Tensor): 包含图像文件路径的 TensorFlow 字符串张量。

        返回:
        - tuple: `(image1, image2)`
            - `image1` (numpy.ndarray): float32 类型的 RGB 彩色图像。
            - `image2` (numpy.ndarray): float32 类型的图像 (灰度三通道或全零数组)。
        """
        # tf.py_func 会将字符串作为 bytes 传递，需要解码
        img_path_str = img_path1_tensor.decode()

        # 判断图像类型 ('style', 'smooth', 或其他)
        if 'style' in img_path_str or 'smooth' in img_path_str:
            # --- 处理 'style' 或 'smooth' 图像 ---
            # image1: 读取彩色图像
            image1 = cv2.imread(img_path_str).astype(np.float32)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) # OpenCV 默认 BGR, 转换为 RGB

            # image2: 读取灰度图像，并扩展为3通道
            image2_gray = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            # 将单通道灰度图复制三份，形成 (H, W, 3) 的形状
            image2 = np.asarray([image2_gray, image2_gray, image2_gray])
            image2 = np.transpose(image2, (1,2,0)) # 从 (3, H, W) 转换为 (H, W, 3)

        else:
            # --- 处理其他类型图像 (如真实照片) ---
            # image1: 读取彩色图像
            image1 = cv2.imread(img_path_str).astype(np.float32)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

            # image2: 创建一个与 image1 形状相同的全零数组
            # 构思说明: 对于真实照片，image2 可能不直接使用，或用于特定的损失计算，
            # 例如，风格损失计算时，真实照片没有对应的“风格灰度图”，因此用零值代替。
            image2 = np.zeros(image1.shape).astype(np.float32)

        return image1, image2

    def load_image(self, img_path_tensor):
        """
        加载并归一化图像。
        此方法作为 `tf.data.Dataset.map` 的处理函数，通过 `tf.py_func` 包装 `read_image`。

        参数:
        - img_path_tensor (tf.Tensor): 包含图像文件路径的 TensorFlow 字符串张量。

        返回:
        - tuple: `(processing_image1, processing_image2)`
            - `processing_image1` (numpy.ndarray): 归一化到 [-1.0, 1.0] 的图像1。
            - `processing_image2` (numpy.ndarray): 归一化到 [-1.0, 1.0] 的图像2。
        """
        # 调用 read_image 读取原始图像数据 (RGB, float32, 0-255 范围)
        image1, image2 = self.read_image(img_path_tensor)

        # 归一化处理：将像素值从 [0, 255] 映射到 [-1.0, 1.0]
        # 构思说明: (image / 127.5) 将范围变为 [0, 2], 然后减 1 得到 [-1, 1]。
        # 这是生成对抗网络中常见的图像预处理步骤。
        processing_image1 = image1 / 127.5 - 1.0
        processing_image2 = image2 / 127.5 - 1.0

        return processing_image1, processing_image2

    def load_images(self):
        """
        构建并返回一个 `tf.data.Dataset` 迭代器，用于加载和预处理图像批次。

        数据管道的构建步骤:
        1. 从图像路径列表创建数据集 (`from_tensor_slices`)。
        2. 无限重复数据集 (`repeat`)，以支持多轮训练。
        3. 随机打乱数据集 (`shuffle`)，以增加训练数据的随机性。
        4. 并行地将 `load_image` 函数（通过 `tf.py_func` 包装）映射到每个图像路径 (`map`)，
           以异步加载和预处理图像。
        5. 将处理后的图像组织成批次 (`batch`)。
        6. 创建一个一次性的迭代器 (`make_one_shot_iterator`) 并获取下一批数据的操作 (`get_next`)。

        返回:
        - tuple: `(img1_batch, img2_batch)`
            - `img1_batch` (tf.Tensor): 一个批次的图像1 (通常是彩色图像)。
            - `img2_batch` (tf.Tensor): 一个批次的图像2 (灰度三通道图或全零图)。
        """
        # 步骤 1: 从图像路径列表创建 TensorFlow 数据集
        dataset = tf.data.Dataset.from_tensor_slices(self.paths)

        # 步骤 2: 无限重复数据集，以支持任意数量的训练轮次
        dataset = dataset.repeat()

        # 步骤 3: 随机打乱数据集
        # buffer_size 通常设置为数据集的大小，以确保充分打乱。
        dataset = dataset.shuffle(buffer_size=len(self.paths))

        # 步骤 4: 并行加载和预处理图像
        # 使用 dataset.map() 将 load_image 函数应用于数据集中的每个元素 (图像路径)。
        # tf.py_func 用于将普通的 Python 函数 (self.load_image) 包装成 TensorFlow 操作。
        # [tf.float32, tf.float32] 指定了 self.load_image 函数返回的两个值的 TensorFlow 数据类型。
        # self.num_cpus 指定了并行处理的线程数。
        dataset = dataset.map(lambda img_path_tensor: tf.py_func(
            self.load_image, # 要调用的 Python 函数
            [img_path_tensor],  # 函数的输入参数 (来自数据集的单个元素)
            [tf.float32,tf.float32]), # 函数返回值的 TensorFlow 数据类型
                              num_parallel_calls=self.num_cpus) # 并行处理的调用数量

        # 步骤 5: 将图像组织成批次
        dataset = dataset.batch(self.batch_size)

        # 步骤 6: 创建迭代器并获取下一批数据的操作
        # make_one_shot_iterator 创建一个简单的迭代器，只能从头到尾迭代一次（但由于前面有 .repeat()，所以可以持续获取数据）
        img1_batch, img2_batch = dataset.make_one_shot_iterator().get_next()

        return img1_batch, img2_batch
