# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本定义了 `Vgg19` 类，用于构建 VGG19 卷积神经网络模型，并从一个 `.npy` 文件加载预训练的权重。
VGG19 网络在计算机视觉领域被广泛应用，特别是在风格迁移和生成对抗网络 (GAN)（如 AnimeGANv2）中，
它常被用来从图像中提取特征，以计算感知损失 (包括内容损失和风格损失)。
感知损失通过比较原始图像、生成图像以及风格参考图像在 VGG19 网络不同层级的激活输出来衡量它们之间的相似性。

`VGG_MEAN` 常量:
这是一个包含三个值的列表 `[103.939, 116.779, 123.68]`。这些值是 VGG 模型在训练时所使用的数据集的
平均像素值，对应 BGR (蓝、绿、红) 通道。在将图像输入 VGG19 网络之前，需要从图像的 BGR 各通道中减去这些均值，
这是 VGG 模型标准的预处理步骤之一。

使用方式:
`Vgg19` 类在使用时首先被实例化 (例如，在 `AnimeGANv2.py` 中)，
然后调用其实例的 `build` 方法，并传入一个输入图像张量。
`build` 方法会根据加载的权重构建 VGG19 的计算图，并将网络各层的激活输出存储在实例的属性中，
以便其他部分代码（如损失函数计算模块）可以访问这些特征图。
"""
import tensorflow as tf

import numpy as np
import time
import sys # 用于在加载权重失败时退出程序

# VGG19 网络在 ImageNet 数据集上训练时使用的 BGR 通道均值。
# 顺序: Blue, Green, Red
VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    """
    Vgg19 类，用于构建 VGG19 网络并加载预训练权重。

    该类能够:
    1. 从指定的 `.npy` 文件加载预训练的 VGG19 权重。
    2. 根据加载的权重构建 VGG19 网络的前向传播计算图。
    3. 提供对网络各卷积层和池化层输出激活的访问。

    实例属性 (在调用 `build()` 后填充):
    - `self.conv1_1`, `self.conv1_2`, `self.pool1`
    - `self.conv2_1`, `self.conv2_2`, `self.pool2`
    - `self.conv3_1`, `self.conv3_2`, `self.conv3_3`, `self.conv3_4`, `self.pool3`
    - `self.conv4_1`, `self.conv4_2`, `self.conv4_3`, `self.conv4_4`
    - `self.conv4_4_no_activation` (conv4_4 层在 ReLU 激活之前的输出)
    - `self.pool4`
    - `self.conv5_1`, `self.conv5_2`, `self.conv5_3`, `self.conv5_4`, `self.pool5`
    - (如果 `include_fc=True`): `self.fc6`, `self.relu6`, `self.fc7`, `self.relu7`, `self.fc8`, `self.prob`
    """
    def __init__(self, vgg19_npy_path='vgg19_weight/vgg19.npy'):
        """
        初始化 Vgg19 类实例。
        主要工作是加载预训练的 VGG19 权重。

        参数:
        - vgg19_npy_path (str, 可选): 包含 VGG19 预训练权重的 `.npy` 文件路径。
                                     默认路径为 'vgg19_weight/vgg19.npy'。
                                     这个 `.npy` 文件通常是一个字典，其中键是层名 (如 'conv1_1')，
                                     值是包含权重和偏置的列表 (例如 `[weights, biases]`)。
        """
        if vgg19_npy_path is not None:
            try:
                # 加载 .npy 文件。encoding='latin1' 和 allow_pickle=True 通常用于加载用 Python 2 保存的 NumPy 数据。
                self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
                print(f"VGG19 .npy 权重文件已加载: {vgg19_npy_path}")
            except FileNotFoundError:
                print(f"[错误] VGG19 .npy 权重文件未找到: {vgg19_npy_path}")
                self.data_dict = None
                sys.exit(1) # 权重加载失败则退出程序
            except Exception as e:
                print(f"[错误] 加载 VGG19 .npy 权重文件时出错: {e}")
                self.data_dict = None
                sys.exit(1)
        else:
            self.data_dict = None
            print("[错误] 未提供 VGG19 .npy 权重文件路径!")
            sys.exit(1) # 权重路径未提供则退出

    def build(self, rgb, include_fc=False):
        """
        构建 VGG19 网络计算图，并加载预训练权重到各层。

        参数:
        - rgb (tf.Tensor): 输入图像张量。期望格式为 RGB，像素值范围为 [-1.0, 1.0]。
                           形状: [batch_size, height, width, 3]。
        - include_fc (bool, 可选): 是否在网络中包含全连接层 (fc6, fc7, fc8)。
                                  对于计算感知损失，通常不需要全连接层，设为 `False`。
                                  如果需要完整的 VGG19 分类网络，则设为 `True`。
                                  默认: `False`。

        预处理步骤:
        1. 将输入 `rgb` 张量的像素值从 [-1.0, 1.0] 范围反归一化到 [0, 255.0] 范围。
        2. 将 RGB 图像数据转换为 BGR 格式 (因为 VGG19 预训练模型期望 BGR 输入)。
        3. 从 BGR 各通道中减去 `VGG_MEAN` (预计算的均值)。

        层构建:
        - 依次调用 `conv_layer` (卷积层 + ReLU) 和 `max_pool` (最大池化层)
          来构建 VGG19 的卷积基。
        - 如果 `include_fc` 为 `True`，则继续构建全连接层。
        - 各层的权重和偏置从 `self.data_dict` 中加载。
        - 将网络中特定层的激活输出保存为实例属性 (如 `self.conv1_1`, `self.pool1`, 等)。
          特别地，`self.conv4_4_no_activation` 保存了 `conv4_4` 层在 ReLU 激活之前的输出，
          这在某些风格损失计算中可能会用到。

        注意:
        - 此方法会修改实例的状态，填充 `self.convX_Y` 等属性。
        - 如果 `include_fc` 为 `True`，在模型构建完成后，`self.data_dict` 会被设为 `None` 以释放内存。
        """
        print("正在构建 VGG19 模型...")
        start_time = time.time() # 记录开始时间

        # 预处理步骤 1: 反归一化 RGB 图像从 [-1, 1] 到 [0, 255]
        # 输入 rgb 的范围是 [-1, 1], ((rgb + 1) / 2) 将其映射到 [0, 1], 再乘以 255 得到 [0, 255]
        rgb_scaled = ((rgb + 1) / 2) * 255.0

        # 预处理步骤 2 & 3: RGB -> BGR 并减去均值
        # 将 rgb_scaled 张量按通道分割 (red, green, blue)
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        # 验证通道分割是否正确 (可选)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1] # 假设输入是 224x224
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]

        # 重新组合为 BGR 顺序，并减去 VGG_MEAN 各通道的均值
        # VGG_MEAN 的顺序是 [B, G, R]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],  # Blue channel
            green - VGG_MEAN[1], # Green channel
            red - VGG_MEAN[2],   # Red channel
        ])
        # 验证 BGR 张量形状 (可选)
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # --- 构建卷积层 (Convolutional Layers) ---
        # Block 1: 对应 VGG19 的 conv1_1, conv1_2, pool1
        self.conv1_1 = self.conv_layer(bgr, "conv1_1") # 第一个卷积层
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2") # 第二个卷积层
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2") # 第二个卷积层
        self.pool1 = self.max_pool(self.conv1_2, 'pool1') # 第一个最大池化层

        # Block 2: 对应 VGG19 的 conv2_1, conv2_2, pool2
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        # Block 3: 对应 VGG19 的 conv3_1 到 conv3_4, pool3
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        # Block 4: 对应 VGG19 的 conv4_1 到 conv4_4, pool4
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        # conv4_4 层在 ReLU 激活之前的输出，某些损失函数可能会使用这个特征
        self.conv4_4_no_activation = self.no_activation_conv_layer(self.conv4_3, "conv4_4")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4") # 标准的 conv4_4 (带 ReLU)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        # Block 5: 对应 VGG19 的 conv5_1 到 conv5_4, pool5
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        # --- 构建全连接层 (Fully Connected Layers)，如果需要 ---
        if include_fc:
            self.fc6 = self.fc_layer(self.pool5, "fc6") # 第一个全连接层
            # 验证 fc6 输出形状是否为 [batch_size, 4096]
            assert self.fc6.get_shape().as_list()[1:] == [4096]
            self.relu6 = tf.nn.relu(self.fc6) # ReLU 激活

            self.fc7 = self.fc_layer(self.relu6, "fc7") # 第二个全连接层
            self.relu7 = tf.nn.relu(self.fc7) # ReLU 激活

            self.fc8 = self.fc_layer(self.relu7, "fc8") # 第三个全连接层 (输出层)

            # Softmax 激活，得到类别概率分布
            self.prob = tf.nn.softmax(self.fc8, name="prob")

            # 清除已加载的权重数据，以释放内存
            # 这样做是因为权重已经作为 tf.constant 加载到计算图中，
            # self.data_dict 不再需要在内存中保留。
            self.data_dict = None

        print(f"VGG19 模型构建完成，耗时: {time.time() - start_time:.4f} 秒")

    def avg_pool(self, bottom, name):
        """
        平均池化层。

        参数:
        - bottom (tf.Tensor): 输入张量。
        - name (str): 层的名称。

        返回:
        - tf.Tensor: 经过平均池化后的张量。
        """
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        """
        最大池化层。

        参数:
        - bottom (tf.Tensor): 输入张量。
        - name (str): 层的名称。

        返回:
        - tf.Tensor: 经过最大池化后的张量。
        """
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        """
        构建一个卷积层，包含 ReLU 激活。
        权重和偏置从 `self.data_dict` 中加载。

        参数:
        - bottom (tf.Tensor): 输入张量。
        - name (str): 层的名称 (例如 "conv1_1")，用于从 `self.data_dict` 查找权重和偏置。

        返回:
        - tf.Tensor: 经过卷积和 ReLU 激活后的张量。
        """
        with tf.variable_scope(name): # 定义变量作用域
            filt = self.get_conv_filter(name) # 获取该层的卷积核权重

            # 执行2D卷积:
            # bottom: 输入张量
            # filt: 卷积核 (权重)
            # strides: 步长 [batch, height, width, channels], 通常设为 [1, 1, 1, 1]
            # padding='SAME': 表示输出特征图与输入特征图在高度和宽度上相同 (当 stride=1 时)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name) # 获取该层的偏置项
            bias = tf.nn.bias_add(conv, conv_biases) # 添加偏置

            relu = tf.nn.relu(bias) # 应用 ReLU 激活函数
            return relu

    def no_activation_conv_layer(self, bottom, name):
        """
        构建一个卷积层，但不包含 ReLU 激活。
        权重和偏置从 `self.data_dict` 中加载。
        主要用于获取特定层 (如 conv4_4) 在激活之前的输出。

        参数:
        - bottom (tf.Tensor): 输入张量。
        - name (str): 层的名称。

        返回:
        - tf.Tensor: 经过卷积（无激活）后的张量。
        """
        # 使用不同的作用域名称以避免与 `conv_layer` 中的同名层产生冲突，
        # 尽管 TensorFlow 的 reuse 机制可以处理，但显式区分更清晰。
        with tf.variable_scope(name + "_no_activation"):
            filt = self.get_conv_filter(name) # 获取卷积核权重

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME') # 执行卷积

            conv_biases = self.get_bias(name) # 获取偏置项
            x = tf.nn.bias_add(conv, conv_biases) # 添加偏置

            return x

    def fc_layer(self, bottom, name):
        """
        构建一个全连接层。
        权重和偏置从 `self.data_dict` 中加载。

        参数:
        - bottom (tf.Tensor): 输入张量。通常是前一个池化层或全连接层的输出。
        - name (str): 层的名称 (例如 "fc6")。

        返回:
        - tf.Tensor: 经过全连接层处理后的张量 (在 ReLU 激活之前)。
        """
        with tf.variable_scope(name): # 定义变量作用域
            shape = bottom.get_shape().as_list() # 获取输入张量的形状
            # 将输入张量展平 (flatten) 为二维 [batch_size, num_features]
            # 例如，如果输入是 [batch, 7, 7, 512] (pool5的输出)，则展平为 [batch, 7*7*512]
            dim = 1
            for d in shape[1:]: # 从第二个维度开始累乘 (跳过批处理维度)，计算特征总数
                dim *= d
            x = tf.reshape(bottom, [-1, dim]) # -1 表示自动推断批大小

            weights = self.get_fc_weight(name) # 获取全连接层的权重
            biases = self.get_bias(name)       # 获取全连接层的偏置

            # 执行全连接操作: output = x * weights + biases
            # tf.matmul(x, weights) 执行矩阵乘法
            # tf.nn.bias_add 会自动广播偏置项 biases 到正确形状。
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        """
        从 `self.data_dict` 中获取指定卷积层的权重 (卷积核)。
        权重被加载为 TensorFlow 常量。

        参数:
        - name (str): 层的名称 (例如 'conv1_1')。

        返回:
        - tf.Tensor (tf.constant): 包含权重的 TensorFlow 常量。
        """
        # self.data_dict[name] 是一个列表，通常 [权重, 偏置]
        # self.data_dict[name][0] 存储的是权重 (卷积核)
        return tf.constant(self.data_dict[name][0], name="filter_" + name)

    def get_bias(self, name):
        """
        从 `self.data_dict` 中获取指定层的偏置项。
        偏置被加载为 TensorFlow 常量。

        参数:
        - name (str): 层的名称。

        返回:
        - tf.Tensor (tf.constant): 包含偏置的 TensorFlow 常量。
        """
        # self.data_dict[name][1] 存储的是偏置项
        return tf.constant(self.data_dict[name][1], name="biases_" + name)

    def get_fc_weight(self, name):
        """
        从 `self.data_dict` 中获取指定全连接层的权重。
        权重被加载为 TensorFlow 常量。

        参数:
        - name (str): 层的名称 (例如 'fc6')。

        返回:
        - tf.Tensor (tf.constant): 包含权重的 TensorFlow 常量。
        """
        # 对于全连接层，权重也通常存储在 self.data_dict[name][0]
        return tf.constant(self.data_dict[name][0], name="weights_" + name)