# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本提供了一系列在 AnimeGANv2 模型中广泛使用的 TensorFlow 操作和自定义层。
它包括了卷积层 (`conv`)、反卷积层 (`deconv`)、残差块 (`resblock`) 等网络构建模块，
以及多种激活函数 (如 `lrelu`, `relu`)、归一化技术 (实例归一化 `instance_norm`,
层归一化 `layer_norm`, 谱归一化 `spectral_norm`)。

此外，该文件还定义了多种重要的损失函数，这些损失函数是训练 GAN 和实现特定图像风格转换效果的基石，
例如：
- 基本损失: L1 损失 (`L1_loss`), L2 损失 (`L2_loss`), Huber 损失 (`Huber_loss`)。
- GAN 对抗损失: 判别器损失 (`discriminator_loss`) 和生成器损失 (`generator_loss`)，支持多种 GAN 变体 (LSGAN, WGAN-GP, HingeGAN 等)。
- 感知损失: 内容损失 (`con_loss`, `con_sty_loss`) 和风格损失 (`style_loss`, `con_sty_loss`)，通常利用 VGG19 网络提取特征。
- 其他特定损失: 色彩损失 (`color_loss`) 用于保持颜色一致性，全变分损失 (`total_variation_loss`) 作为平滑正则项。
- 颜色空间转换: `rgb2yuv` 用于将图像从 RGB 转换到 YUV 色彩空间，常用于色彩损失的计算。

这些函数和操作被项目中的其他核心脚本（如 `AnimeGANv2.py` 进行模型训练逻辑的定义，
以及 `net/generator.py` 和 `net/discriminator.py` 进行网络架构的构建）大量导入和调用。

全局变量说明:
- `weight_init`: 定义了网络层权重的默认初始化器，此处使用均值为0、标准差为0.02的正态分布 (`tf.random_normal_initializer`)。
- `weight_regularizer`: 定义了权重的默认正则化器，此处为 `None`，表示默认不使用权重正则化。
  (注释中提到了 Xavier, He 初始化器和 L2 正则化器作为可选项。)
"""
import tensorflow as tf
import tensorflow.contrib as tf_contrib


# Xavier 初始化器: tf_contrib.layers.xavier_initializer()
# He 初始化器: tf_contrib.layers.variance_scaling_initializer() (TensorFlow 核心 API 中也有 tf.keras.initializers.VarianceScaling)
# 普通正态分布初始化器: tf.random_normal_initializer(mean=0.0, stddev=0.02)
# L2 正则化器: tf_contrib.layers.l2_regularizer(0.0001)

# 默认权重初始化策略：均值为0.0，标准差为0.02的正态分布。
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
# 默认权重正则化策略：无正则化。
weight_regularizer = None

##################################################################################
# 层定义 (Layers)
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    """
    自定义二维卷积层。

    此函数封装了卷积操作，提供了灵活的填充选项 (包括 'zero' 和 'reflect')，
    并支持可选的谱归一化 (Spectral Normalization, SN)。
    当使用谱归一化时，卷积权重会经过谱归一化处理，这有助于稳定 GAN 的训练。

    参数:
    - x (tf.Tensor): 输入张量，形状通常为 [batch, height, width, in_channels]。
    - channels (int): 输出通道数 (即卷积核的数量)。
    - kernel (int): 卷积核的大小 (正方形核，例如 kernel=4 表示 4x4)。默认: 4。
    - stride (int): 卷积步长。默认: 2 (通常用于下采样)。
    - pad (int): 在进行 'VALID' 卷积之前，手动在图像边界添加的填充量 (通常用于控制输出尺寸)。默认: 0。
    - pad_type (str): 手动填充的类型。可选 'zero' (零填充) 或 'reflect' (反射填充)。默认: 'zero'。
    - use_bias (bool): 是否使用偏置项。默认: True。
    - sn (bool): 是否对卷积核权重应用谱归一化。默认: False。
    - scope (str): TensorFlow 变量作用域的名称。默认: 'conv_0'。

    返回:
    - tf.Tensor: 经过卷积操作（可能还包括谱归一化和偏置添加）后的张量。

    实现说明:
    - 填充逻辑 (`pad_top`, `pad_bottom`, `pad_left`, `pad_right`) 是为了在 `padding='VALID'` 的卷积操作之前，
      通过 `tf.pad` 手动控制有效的输入区域，从而达到类似 `padding='SAME'` 但具有特定填充方式的效果。
    - 当 `sn=True` 时，权重 `w` 通过 `spectral_norm(w)` 进行归一化，然后使用底层的 `tf.nn.conv2d`。
      偏置项是单独创建和添加的。
    - 当 `sn=False` 时，使用 `tf.layers.conv2d`，这是一个更高级的API，它内部处理权重和偏置的创建。
    """
    with tf.variable_scope(scope):
        # 计算手动填充量，确保卷积核在滑动时能覆盖期望的区域
        # 这个填充逻辑比较特殊，似乎是为了在 stride > 1 时，通过 'VALID' 卷积达到特定的输出尺寸
        if (kernel - stride) % 2 == 0 : # 如果 (核大小 - 步长) 是偶数
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad
        else : # 如果是奇数
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        # 根据 pad_type 应用填充
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect' : # 反射填充
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn : # 如果使用谱归一化
            # 手动创建卷积核权重 w
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            # 对权重应用谱归一化，然后进行卷积 (padding='VALID' 因为手动填充已完成)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias : # 如果使用偏置
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias) # 添加偏置
        else : # 如果不使用谱归一化
            # 使用 tf.layers.conv2d 高级 API (注意：此 API 在 TensorFlow 2.x 中已整合到 tf.keras.layers.Conv2D)
            # 由于前面已经手动 padding，这里的 padding 可能需要根据具体情况调整，
            # 或者依赖于 tf.layers.conv2d 自身的 padding='SAME'/'VALID' 逻辑（如果手动 pad 为0）。
            # 鉴于手动 pad 的存在，这里如果 stride > 1 且希望保持尺寸，可能仍需 careful consideration。
            # 然而，原代码中 `tf.layers.conv2d` 没有指定 padding 参数，默认为 'valid'。
            # 这意味着输出尺寸会是 ((W - K + 2P_manual)/S) + 1。
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias) # 默认 padding='VALID'

        return x

def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='deconv_0'):
    """
    自定义二维反卷积层 (Transposed Convolution)。
    用于上采样特征图。支持可选的谱归一化。

    参数:
    - x (tf.Tensor): 输入张量。
    - channels (int): 输出通道数。
    - kernel (int): 卷积核大小。默认: 4。
    - stride (int): 步长，通常用于控制上采样的倍数 (例如 stride=2 表示尺寸加倍)。默认: 2。
    - use_bias (bool): 是否使用偏置项。默认: True。
    - sn (bool): 是否对卷积核权重应用谱归一化。默认: False。
    - scope (str): TensorFlow 变量作用域的名称。默认: 'deconv_0'。

    返回:
    - tf.Tensor: 经过反卷积操作后的张量。

    实现说明:
    - 输出形状 `output_shape` 被计算为输入尺寸乘以步长。
    - 当 `sn=True` 时，权重 `w` 经过谱归一化处理，然后使用底层的 `tf.nn.conv2d_transpose`。
    - 当 `sn=False` 时，使用 `tf.layers.conv2d_transpose` 高级 API。
    """
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list() # 获取输入张量的静态形状
        # 计算输出形状：[batch, height*stride, width*stride, out_channels]
        # tf.shape(x)[1] 和 tf.shape(x)[2] 获取动态的高度和宽度
        output_shape = [x_shape[0], tf.shape(x)[1]*stride, tf.shape(x)[2]*stride, channels]

        if sn : # 如果使用谱归一化
            # 手动创建反卷积核权重 w
            # 注意：反卷积的权重形状是 [kernel, kernel, output_channels, input_channels]
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            # 对权重应用谱归一化，然后进行反卷积 (padding='SAME' 以尝试匹配 output_shape)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')

            if use_bias : # 如果使用偏置
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias) # 添加偏置
        else : # 如果不使用谱归一化
            # 使用 tf.layers.conv2d_transpose 高级 API
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding='SAME', use_bias=use_bias) # padding='SAME' 确保输出尺寸大致为输入尺寸的 stride 倍

        return x


##################################################################################
# 残差块 (Residual-block)
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock_0'):
    """
    定义一个残差块 (Residual Block)。
    残差块包含两个卷积层序列，每个序列是 Conv -> InstanceNorm -> ReLU。
    最后，将块的输入 `x_init` 与第二个卷积序列的输出相加（跳跃连接）。

    参数:
    - x_init (tf.Tensor): 残差块的输入张量。
    - channels (int): 块内卷积层的通道数。
    - use_bias (bool): 卷积层是否使用偏置项。默认: True。
    - scope (str): TensorFlow 变量作用域的名称。默认: 'resblock_0'。

    返回:
    - tf.Tensor: 经过残差块处理后的张量。
    """
    with tf.variable_scope(scope): # 定义残差块的主作用域
        # 第一个卷积序列
        with tf.variable_scope('res1'):
            # 卷积 (kernel=3, stride=1, pad=1, pad_type='reflect')
            # pad=1 和 pad_type='reflect' 意味着使用反射填充使 3x3 卷积保持输入尺寸
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x) # 实例归一化
            x = relu(x) # ReLU 激活

        # 第二个卷积序列
        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            # 注意：第二个卷积序列后没有 ReLU 激活，直接与输入相加

        return x + x_init # 残差连接：将输入 x_init 加到处理后的 x 上

##################################################################################
# Sampling (此类别名可能不太准确，flatten 更像是塑形操作)
##################################################################################

def flatten(x) :
    """
    将输入张量展平 (flatten)。
    除了批处理维度 (通常是第一个维度) 外，将所有其他维度合并为一个维度。
    例如，输入形状 [N, H, W, C] 会被展平为 [N, H*W*C]。
    使用了 `tf.layers.flatten`。

    参数:
    - x (tf.Tensor): 输入张量。

    返回:
    - tf.Tensor: 展平后的张量。
    """
    return tf.layers.flatten(x)

##################################################################################
# 激活函数 (Activation function)
##################################################################################

def lrelu(x, alpha=0.2):
    """
    Leaky ReLU (带泄露修正线性单元) 激活函数。
    f(x) = alpha * x  if x < 0
    f(x) = x         if x >= 0
    有助于防止神经元"死亡"。

    参数:
    - x (tf.Tensor): 输入张量。
    - alpha (float): 负值部分的斜率。默认: 0.2。

    返回:
    - tf.Tensor: 激活后的张量。
    """
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    """
    ReLU (修正线性单元) 激活函数。
    f(x) = max(0, x)

    参数:
    - x (tf.Tensor): 输入张量。

    返回:
    - tf.Tensor: 激活后的张量。
    """
    return tf.nn.relu(x)


def tanh(x):
    """
    Tanh (双曲正切) 激活函数。
    将输入值映射到 [-1, 1] 范围。

    参数:
    - x (tf.Tensor): 输入张量。

    返回:
    - tf.Tensor: 激活后的张量。
    """
    return tf.tanh(x)

def sigmoid(x) :
    """
    Sigmoid 激活函数。
    将输入值映射到 (0, 1) 范围。

    参数:
    - x (tf.Tensor): 输入张量。

    返回:
    - tf.Tensor: 激活后的张量。
    """
    return tf.sigmoid(x)

##################################################################################
# 归一化函数 (Normalization function)
##################################################################################

def instance_norm(x, scope='instance_norm'):
    """
    实例归一化 (Instance Normalization)。
    对特征图的每个通道在每个样本内独立进行归一化。
    常用于风格迁移任务，因为它能消除特定图像的对比度信息。

    参数:
    - x (tf.Tensor): 输入张量，通常形状为 [batch, height, width, channels]。
    - scope (str): TensorFlow 变量作用域的名称。默认: 'instance_norm'。

    返回:
    - tf.Tensor: 经过实例归一化后的张量。
    """
    # 使用 TensorFlow Contrib 中的 instance_norm 实现
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05, # 防止除以零的小常数
                                           center=True, scale=True, # 是否学习偏置 (beta) 和缩放 (gamma) 参数
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    """
    层归一化 (Layer Normalization)。
    对单个样本的所有特征进行归一化。

    参数:
    - x (tf.Tensor): 输入张量。
    - scope (str): TensorFlow 变量作用域的名称。默认: 'layer_norm'。

    返回:
    - tf.Tensor: 经过层归一化后的张量。
    """
    # 使用 TensorFlow Contrib 中的 layer_norm 实现
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True, # 是否学习偏置 (beta) 和缩放 (gamma) 参数
                                        scope=scope)

def batch_norm(x, is_training=True, scope='batch_norm'):
    """
    批量归一化 (Batch Normalization)。
    在一个批次 (mini-batch) 的数据上对每个特征通道进行归一化。
    依赖于 `is_training` 参数来决定是使用当前批次的统计量还是使用训练阶段累积的移动平均统计量。

    参数:
    - x (tf.Tensor): 输入张量。
    - is_training (bool): 是否处于训练模式。在训练时为 True，在测试/推理时为 False。默认: True。
    - scope (str): TensorFlow 变量作用域的名称。默认: 'batch_norm'。

    返回:
    - tf.Tensor: 经过批量归一化后的张量。
    """
    # 使用 TensorFlow Contrib 中的 batch_norm 实现
    # updates_collections=None 表示批归一化的更新操作 (移动均值和方差的更新) 会被添加到默认的 UPDATE_OPS 集合中，
    # 需要在训练时手动执行这些操作，或者依赖于某些训练框架的自动处理。
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, # 移动均值和方差的衰减率
                                        epsilon=1e-05, # 防止除以零的小常数
                                        center=True, scale=True, # 是否学习 beta 和 gamma 参数
                                        updates_collections=None,
                                        is_training=is_training, # 关键参数，控制行为
                                        scope=scope)


def spectral_norm(w, iteration=1):
    """
    对权重矩阵 `w` 应用谱归一化。
    谱归一化通过将权重矩阵除以其最大奇异值 (谱范数) 来限制其 Lipschitz 常数。
    这有助于稳定 GAN 的训练。最大奇异值通过幂迭代法 (power iteration) 近似计算。

    参数:
    - w (tf.Variable): 需要进行谱归一化的权重张量。通常是卷积核或全连接层的权重。
                       期望形状为 [..., output_channels]。
    - iteration (int): 幂迭代的次数。通常 1 次迭代就足够。默认: 1。

    返回:
    - tf.Tensor: 经过谱归一化后的权重张量。
    """
    w_shape = w.shape.as_list() # 获取权重的静态形状
    w = tf.reshape(w, [-1, w_shape[-1]]) # 将权重 reshape 为二维矩阵 [N, output_channels]

    # u 是幂迭代中使用的向量，形状为 [1, output_channels]
    # 初始化为截断正态分布，且不可训练
    u = tf.get_variable("u_vector_for_sn", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u # 当前迭代的 u
    v_hat = None # 当前迭代的 v

    # 幂迭代过程
    for i in range(iteration):
        # v' = u_hat * W^T
        v_ = tf.matmul(u_hat, tf.transpose(w))
        # v_hat = v' / ||v'||_2 (L2 归一化)
        v_hat = l2_norm(v_)

        # u' = v_hat * W
        u_ = tf.matmul(v_hat, w)
        # u_hat = u' / ||u'||_2 (L2 归一化)
        u_hat = l2_norm(u_)

    # sigma (最大奇异值) ≈ u_hat * W * v_hat^T
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    # 归一化权重: W_sn = W / sigma
    w_norm = w / sigma

    # 使用 tf.control_dependencies 确保在计算 w_norm 之前，u 的值被更新为最新的 u_hat。
    # 这对于下一次调用 spectral_norm (如果在同一个作用域内且权重共享) 很重要。
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape) # 将归一化后的权重 reshape回原始形状

    return w_norm

def l2_norm(v, eps=1e-12):
    """
    计算向量 `v` 的 L2 范数，并用其对向量进行归一化。
    即: v / ||v||_2

    参数:
    - v (tf.Tensor): 输入向量。
    - eps (float): 一个非常小的正数，用于防止除以零。默认: 1e-12。

    返回:
    - tf.Tensor: L2 归一化后的向量。
    """
    # tf.reduce_sum(v ** 2) 计算向量元素平方和
    # (...) ** 0.5 计算平方根，即 L2 范数
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

##################################################################################
# 损失函数 (Loss function)
##################################################################################

def L1_loss(x, y):
    """
    计算 L1 损失 (Mean Absolute Error, MAE)。
    L1_loss = mean(|x - y|)

    参数:
    - x (tf.Tensor): 第一个输入张量 (例如，预测值)。
    - y (tf.Tensor): 第二个输入张量 (例如，真实值)。

    返回:
    - tf.Tensor: 一个标量，表示计算得到的 L1 损失。
    """
    loss = tf.reduce_mean(tf.abs(x - y)) # 计算差的绝对值的均值
    return loss

def L2_loss(x,y):
    """
    计算 L2 损失 (Mean Squared Error, MSE)，但这里的实现是 (1/N) * sum( (x_i - y_i)^2 )，
    或者说是 (2/N) * (1/2 * sum( (x_i - y_i)^2 ))。
    其中 N 是张量中元素的总数。
    注意：`tf.nn.l2_loss(t)` 计算 `sum(t**2) / 2`。

    参数:
    - x (tf.Tensor): 第一个输入张量。
    - y (tf.Tensor): 第二个输入张量。

    返回:
    - tf.Tensor: 一个标量，表示计算得到的 L2 损失。
    """
    size = tf.size(x) # 获取张量 x 中元素的总数
    # tf.nn.l2_loss(x-y) 计算 sum((x-y)^2) / 2
    # 乘以 2 得到 sum((x-y)^2)
    # 除以 tf.to_float(size) 得到均方误差。
    # 所以，此函数计算的是 MSE。
    return tf.nn.l2_loss(x-y)* 2 / tf.to_float(size)

def Huber_loss(x,y):
    """
    计算 Huber 损失。
    Huber 损失是 L1 损失和 L2 损失的一种结合，它在误差较小时表现得像 L2 损失（平滑），
    在误差较大时表现得像 L1 损失（对异常值不那么敏感）。

    参数:
    - x (tf.Tensor): 第一个输入张量。
    - y (tf.Tensor): 第二个输入张量。

    返回:
    - tf.Tensor: 一个标量，表示计算得到的 Huber 损失。
    """
    # 使用 TensorFlow 内置的 Huber 损失函数
    return tf.losses.huber_loss(labels=y, predictions=x) # 注意参数顺序，通常是 (labels, predictions)

def discriminator_loss(loss_func, real, gray, fake, real_blur):
    """
    计算判别器的总损失。
    根据指定的 `loss_func` (GAN 类型)，损失的计算方式不同。
    判别器的目标是：
    - 对真实的动漫图像 (`real`) 输出高分 (或接近1)。
    - 对灰度的动漫图像 (`gray`) 输出低分 (或接近0)。
    - 对生成器生成的伪造图像 (`fake`) 输出低分 (或接近0)。
    - 对边缘平滑的真实动漫图像 (`real_blur`) 输出低分 (或接近0)。

    参数:
    - loss_func (str): GAN 损失函数的类型。可选: 'wgan-gp', 'wgan-lp', 'lsgan', 'gan', 'dragan', 'hinge'。
    - real (tf.Tensor): 判别器对真实动漫图像的输出 logit。
    - gray (tf.Tensor): 判别器对灰度动漫图像的输出 logit。
    - fake (tf.Tensor): 判别器对生成器生成的伪造图像的输出 logit。
    - real_blur (tf.Tensor): 判别器对边缘平滑的真实动漫图像的输出 logit。

    返回:
    - tf.Tensor: 一个标量，表示判别器的总损失。

    权重说明:
    - 最终损失是各项损失的加权和。代码中硬编码了权重 (例如 `1.7 * real_loss + ...`)。
      这些权重可能针对特定数据集 (如 Hayao, Paprika, Shinkai) 进行了调整。
      当前代码中的权重 (1.7, 1.7, 1.7, 1.0) 可能对应 Shinkai 数据集，如注释所示。
    """
    real_loss = 0
    gray_loss = 0
    fake_loss = 0
    real_blur_loss = 0

    # --- 根据 GAN 类型计算各项基础损失 ---
    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp': # Wasserstein GAN 变体
        # 判别器目标：最大化 E[D(real)] - E[D(fake)]
        # 因此，损失为：-E[D(real)] + E[D(fake)] (+ E[D(gray)] + E[D(real_blur)])
        real_loss = -tf.reduce_mean(real)
        gray_loss = tf.reduce_mean(gray) # 目标是让 D(gray) 更小 (接近负无穷)
        fake_loss = tf.reduce_mean(fake) # 目标是让 D(fake) 更小
        real_blur_loss = tf.reduce_mean(real_blur) # 目标是让 D(real_blur) 更小

    if loss_func == 'lsgan' : # Least Squares GAN
        # 判别器目标：D(real) -> 1, D(other) -> 0
        # 损失: 0.5 * ( (D(real)-1)^2 + (D(fake)-0)^2 + (D(gray)-0)^2 + (D(real_blur)-0)^2 )
        real_loss = tf.reduce_mean(tf.square(real - 1.0))
        gray_loss = tf.reduce_mean(tf.square(gray)) # (gray - 0)^2
        fake_loss = tf.reduce_mean(tf.square(fake)) # (fake - 0)^2
        real_blur_loss = tf.reduce_mean(tf.square(real_blur)) # (real_blur - 0)^2

    if loss_func == 'gan' or loss_func == 'dragan' : # Standard GAN (with sigmoid cross-entropy) or DRAGAN
        # 判别器目标：D(real) -> 1 (logits -> +inf), D(other) -> 0 (logits -> -inf)
        # 损失: H(1, D(real)) + H(0, D(fake)) + H(0, D(gray)) + H(0, D(real_blur))
        # H(p,q) = -p*log(q) - (1-p)*log(1-q)
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        gray_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(gray), logits=gray))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
        real_blur_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_blur), logits=real_blur))

    if loss_func == 'hinge': # Hinge Loss GAN
        # 判别器目标: D(real) >= 1, D(other) <= -1
        # 损失: E[max(0, 1-D(real))] + E[max(0, 1+D(fake))] + ...
        real_loss = tf.reduce_mean(relu(1.0 - real))
        gray_loss = tf.reduce_mean(relu(1.0 + gray))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))
        real_blur_loss = tf.reduce_mean(relu(1.0 + real_blur))

    # 不同数据集使用的权重，注释中提供了参考值，当前代码使用的是 Shinkai 的权重。
    # Hayao: 1.2, 1.2, 1.2, 0.8
    # Paprika: 1.0, 1.0, 1.0, 0.005
    # Shinkai (当前使用): 1.7, 1.7, 1.7, 1.0
    loss = 1.7 * real_loss +  1.7 * fake_loss + 1.7 * gray_loss  +  1.0 * real_blur_loss

    return loss

def generator_loss(loss_func, fake):
    """
    计算生成器的对抗性损失。
    生成器的目标是欺骗判别器，使其对生成的伪造图像 (`fake`) 输出高分 (或接近1)。

    参数:
    - loss_func (str): GAN 损失函数的类型。与 `discriminator_loss` 中的选项一致。
    - fake (tf.Tensor): 判别器对生成器生成的伪造图像的输出 logit。

    返回:
    - tf.Tensor: 一个标量，表示生成器的对抗性损失。
    """
    fake_loss = 0 # 初始化生成器损失

    # --- 根据 GAN 类型计算损失 ---
    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp': # Wasserstein GAN 变体
        # 生成器目标：最大化 E[D(fake)]
        # 因此，损失为：-E[D(fake)]
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' : # Least Squares GAN
        # 生成器目标：D(fake) -> 1
        # 损失: 0.5 * (D(fake)-1)^2
        fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

    if loss_func == 'gan' or loss_func == 'dragan': # Standard GAN (with sigmoid cross-entropy) or DRAGAN
        # 生成器目标：D(fake) -> 1 (logits -> +inf)
        # 损失: H(1, D(fake))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge': # Hinge Loss GAN
        # 生成器目标: D(fake) -> (趋向于 >= 1)
        # 损失: -E[D(fake)]
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss # 生成器的总对抗损失

    return loss

def gram(x):
    """
    计算输入张量 `x` 的 Gram 矩阵。
    Gram 矩阵用于衡量特征图中不同特征之间的相关性，常用于风格损失的计算。
    计算方式: G(x)_{ij} = sum_k (F_{ik} * F_{jk})，其中 F 是将特征图 reshape 后的矩阵。

    参数:
    - x (tf.Tensor): 输入特征图张量，形状为 [batch, height, width, channels]。

    返回:
    - tf.Tensor: 计算得到的 Gram 矩阵，形状为 [batch, channels, channels]。
    """
    shape_x = tf.shape(x) # 获取输入张量的动态形状
    b = shape_x[0] # batch_size
    # h = shape_x[1] # height (未使用)
    # w = shape_x[2] # width (未使用)
    c = shape_x[3] # number of channels

    # Reshape: [batch, height, width, channels] -> [batch, height*width, channels]
    x = tf.reshape(x, [b, -1, c])

    # 计算 Gram 矩阵: (x^T * x) / (num_elements_in_feature_map_per_batch)
    # tf.transpose(x, [0, 2, 1]) 交换最后两个维度: [b, c, h*w]
    # tf.matmul(...) 执行批量矩阵乘法: [b, c, h*w] * [b, h*w, c] -> [b, c, c]
    # tf.cast((tf.size(x) // b), tf.float32) 计算每个样本的特征图元素数量 (h*w*c)，并转换为 float32 用于归一化。
    # 这里的归一化因子是 (h*w*c)，有些实现可能会用 (h*w) 或 (h*w*c*c)。
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)

def con_loss(vgg, real, fake):
    """
    计算内容损失 (Content Loss)。
    内容损失通过比较真实图像和生成图像在 VGG19 网络某一中间层的特征表示来衡量它们在内容上的相似度。
    通常使用 L1 或 L2 距离。

    参数:
    - vgg (Vgg19): 一个预训练的 Vgg19 网络实例。
    - real (tf.Tensor): 真实图像张量 (通常是原始照片)。
    - fake (tf.Tensor): 生成器生成的图像张量。

    返回:
    - tf.Tensor: 一个标量，表示内容损失。
    """
    # 使用 VGG19 提取真实图像的特征图
    vgg.build(real) # 将真实图像输入 VGG 网络
    real_feature_map = vgg.conv4_4_no_activation # 获取 conv4_4 层在 ReLU 激活前的输出作为内容特征

    # 使用 VGG19 提取生成图像的特征图 (复用 VGG 网络实例)
    vgg.build(fake) # 将生成图像输入 VGG 网络
    fake_feature_map = vgg.conv4_4_no_activation # 获取对应的特征图

    # 计算真实特征图和伪造特征图之间的 L1 损失
    loss = L1_loss(real_feature_map, fake_feature_map)

    return loss


def style_loss(style_vgg_features, fake_vgg_features):
    """
    计算风格损失 (Style Loss)。
    风格损失通过比较风格参考图像和生成图像在 VGG19 网络多个中间层的特征图的 Gram 矩阵来实现。
    这里简化为只比较传入的某一层特征的 Gram 矩阵。
    (注意：原函数名是 style_loss(style, fake)，但参数实际是特征图，修改为更清晰的名称)

    参数:
    - style_vgg_features (tf.Tensor): 从风格参考图像中提取的 VGG 特征图。
    - fake_vgg_features (tf.Tensor): 从生成图像中提取的对应 VGG 特征图。

    返回:
    - tf.Tensor: 一个标量，表示风格损失 (基于单层特征的 Gram 矩阵)。
    """
    # 计算风格特征图的 Gram 矩阵和生成图像特征图的 Gram 矩阵之间的 L1 损失
    return L1_loss(gram(style_vgg_features), gram(fake_vgg_features))

def con_sty_loss(vgg, real, anime_style_ref, fake):
    """
    计算组合的内容损失和风格损失。

    参数:
    - vgg (Vgg19): 预训练的 Vgg19 网络实例。
    - real (tf.Tensor): 真实图像张量 (用于内容重建的目标)。
    - anime_style_ref (tf.Tensor): 动漫风格参考图像张量 (用于风格模仿的目标)。
    - fake (tf.Tensor): 生成器生成的图像张量。

    返回:
    - tuple: `(c_loss, s_loss)`
        - `c_loss` (tf.Tensor): 内容损失。
        - `s_loss` (tf.Tensor): 风格损失。
    """
    # 提取真实图像 (内容目标) 的 VGG 特征 (conv4_4)
    vgg.build(real)
    real_feature_map = vgg.conv4_4_no_activation

    # 提取生成图像的 VGG 特征 (conv4_4)
    vgg.build(fake)
    fake_feature_map = vgg.conv4_4_no_activation

    # 提取动漫风格参考图像的 VGG 特征 (conv4_4)
    # 注意：anime_style_ref 可能与 fake_feature_map 的批大小不同，
    # 如果 VGG build 方法或后续的 gram/L1_loss 不支持广播或不同批大小，这里可能需要调整。
    # 通常，风格参考图像可以是一个，然后其风格特征与批次中每个伪造图像的特征进行比较。
    # 此处假设 anime_style_ref 的批大小与 fake_feature_map 兼容（例如，通过切片或广播）。
    # anime[:fake_feature_map.shape[0]] 确保了批大小一致。
    vgg.build(anime_style_ref[:tf.shape(fake_feature_map)[0]]) # 截取 anime_style_ref 以匹配 fake 的批大小
    anime_feature_map_for_style = vgg.conv4_4_no_activation

    # 计算内容损失: L1( VGG(real), VGG(fake) )
    c_loss = L1_loss(real_feature_map, fake_feature_map)
    # 计算风格损失: L1( Gram(VGG(anime_style_ref)), Gram(VGG(fake)) )
    s_loss = style_loss(anime_feature_map_for_style, fake_feature_map)

    return c_loss, s_loss

def color_loss(con_image, fake_image):
    """
    计算色彩损失 (Color Loss)。
    该损失旨在使生成图像的颜色分布与原始内容图像的颜色分布相似。
    它在 YUV 色彩空间中计算：
    - Y (亮度) 通道使用 L1 损失。
    - U (色度蓝) 和 V (色度红) 通道使用 Huber 损失。

    参数:
    - con_image (tf.Tensor): 内容参考图像 (通常是原始照片)，RGB 格式，范围 [-1, 1]。
    - fake_image (tf.Tensor): 生成器生成的图像，RGB 格式，范围 [-1, 1]。

    返回:
    - tf.Tensor: 一个标量，表示色彩损失。
    """
    # 将 RGB 图像转换为 YUV 色彩空间
    con_yuv = rgb2yuv(con_image)  # 范围可能变为 Y:[0,1], U:[-0.436,0.436], V:[-0.615,0.615] 或类似
    fake_yuv = rgb2yuv(fake_image)

    # Y 通道 (亮度) 使用 L1 损失
    # con_yuv[:,:,:,0] 和 fake_yuv[:,:,:,0] 分别提取 Y 通道
    y_loss = L1_loss(con_yuv[:,:,:,0], fake_yuv[:,:,:,0])
    # U 通道 (色度) 使用 Huber 损失
    u_loss = Huber_loss(con_yuv[:,:,:,1],fake_yuv[:,:,:,1])
    # V 通道 (色度) 使用 Huber 损失
    v_loss = Huber_loss(con_yuv[:,:,:,2],fake_yuv[:,:,:,2])

    return  y_loss + u_loss + v_loss # 总色彩损失是各通道损失之和

def total_variation_loss(inputs):
    """
    计算全变分损失 (Total Variation Loss, TV Loss)。
    TV 损失是一种正则化项，用于鼓励生成图像在空间上更平滑，减少噪点。
    它通过惩罚图像中相邻像素之间的梯度差异来实现。
    V(y) = sum_ij ( (y_{i+1,j} - y_{ij})^2 + (y_{i,j+1} - y_{ij})^2 )^{1/2} (L1 TV)
    或 V(y) = sum_ij ( (y_{i+1,j} - y_{ij})^2 + (y_{i,j+1} - y_{ij})^2 ) (L2 TV, 此处使用)

    参数:
    - inputs (tf.Tensor): 输入图像张量，形状为 [batch, height, width, channels]。

    返回:
    - tf.Tensor: 一个标量，表示全变分损失。
    """
    # 计算高度方向上的差分 (y_{i+1,j} - y_{ij})
    # inputs[:, :-1, ...] 选择除最后一行外的所有行
    # inputs[:, 1:, ...]  选择除第一行外的所有行
    dh = inputs[:, :-1, :, :] - inputs[:, 1:, :, :]
    # 计算宽度方向上的差分 (y_{i,j+1} - y_{ij})
    dw = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]

    # 获取差分张量中元素的总数，用于归一化
    size_dh = tf.cast(tf.size(dh), tf.float32) # tf.size 返回 int32, 转为 float32
    size_dw = tf.cast(tf.size(dw), tf.float32)

    # 计算 L2 损失 (sum of squares / 2)，然后乘以2并除以元素数量，得到均方差
    # 此处实际上是计算 dh 和 dw 的均方值之和 (MSE of differences)
    # tf.nn.l2_loss(t) = sum(t**2) / 2
    return tf.nn.l2_loss(dh) / size_dh + tf.nn.l2_loss(dw) / size_dw

def rgb2yuv(rgb):
    """
    将 RGB 图像转换为 YUV 色彩空间。
    首先将 RGB 像素值从 [-1, 1] 范围反归一化到 [0, 1] 范围，
    然后使用 TensorFlow 内置的 `tf.image.rgb_to_yuv` 函数进行转换。

    参数:
    - rgb (tf.Tensor): 输入的 RGB 图像张量，像素值范围为 [-1, 1]。

    返回:
    - tf.Tensor: YUV 格式的图像张量。
                 Y 通道范围 [0, 1]。
                 U 通道范围 approx [-0.436, 0.436]。
                 V 通道范围 approx [-0.615, 0.615]。
    """
    # 将 RGB 图像从 [-1, 1] 范围转换到 [0, 1] 范围，这是 tf.image.rgb_to_yuv 的期望输入范围
    rgb_scaled = (rgb + 1.0) / 2.0

    # # 原代码中手动计算 YUV 的部分被注释掉了，改用 tf.image.rgb_to_yuv
    # rgb2yuv_filter = tf.constant([[[[0.299, -0.169, 0.499],
    #                                 [0.587, -0.331, -0.418],
    #                                 [0.114, 0.499, -0.0813]]]]) # RGB 到 YUV 的转换矩阵 (BT.601 标准)
    # rgb2yuv_bias = tf.constant([0., 0.5, 0.5]) # YUV 的偏置
    # temp = tf.nn.conv2d(rgb_scaled, rgb2yuv_filter, [1, 1, 1, 1], 'SAME') # 使用卷积实现矩阵乘法
    # temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    # return temp

    return tf.image.rgb_to_yuv(rgb_scaled)
