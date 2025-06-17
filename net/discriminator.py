# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本定义了 AnimeGANv2 项目的判别器网络 (`D_net`)。
判别器的主要职责是区分真实的动漫图像（以及经过边缘平滑处理的动漫图像）
与由生成器生成的伪造动漫风格图像。它通过学习真实动漫图像的特征分布，
来判断输入图像是否属于这个分布。

使用方式:
此文件不直接运行。`D_net` 函数由其他脚本导入和使用，主要是在 `AnimeGANv2.py`
中用于构建 GAN 模型的判别器部分。
"""

from tools.ops import * # 导入自定义的操作函数，例如 conv, lrelu, layer_norm 等

def D_net(x_init, ch, n_dis, sn, scope, reuse):
    """
    定义判别器网络的架构。

    判别器通过一系列卷积层、激活函数和归一化层来处理输入图像，
    最终输出一个标量值 (logit)，表示输入图像被判断为 "真实" (属于目标动漫风格) 的程度。

    参数:
    - x_init (tf.Tensor): 输入张量，代表需要判别的图像。
                          它可以是真实的动漫图像、边缘平滑的动漫图像或生成器生成的图像。
    - ch (int): 卷积层通道数的基础值。实际通道数会基于此值进行调整。
    - n_dis (int): 判别器中主要下采样/特征提取块的数量。该值控制判别器的深度和感受野大小。
                   例如，如果 n_dis=3，则会有 (n_dis - 1) = 2 个主要的下采样块。
    - sn (bool): 是否在卷积层中使用谱归一化 (Spectral Normalization)。
                 谱归一化有助于稳定 GAN 的训练，通过限制判别器权重矩阵的谱范数来强制执行 Lipschitz 约束。
    - scope (str): 此判别器网络的 TensorFlow 变量作用域名称。
    - reuse (bool or tf.AUTO_REUSE): 是否复用在此作用域下已定义的变量。
                                     在构建对抗网络时，判别器会被多次调用（例如，一次用于真实图像，一次用于生成图像），
                                     此时应将 reuse 设置为 True 或 tf.AUTO_REUSE 以共享权重。

    返回:
    - tf.Tensor: 判别器的输出 logit。这是一个原始的、未经激活的标量值，
                 在不同的 GAN 损失函数 (如 LSGAN, WGAN-GP) 中有不同的解释方式，
                 但通常表示输入图像的“真实性”得分。

    架构说明:
    1.  初始卷积层: 对输入图像进行初步的特征提取。
    2.  循环构建 `n_dis-1` 个主要的判别器块:
        -   每个块开始于一个步长为2的卷积层 (strided convolution)，用于图像下采样，同时增加通道数，扩大感受野。
        -   接着是一个步长为1的卷积层，用于进一步的特征提取。
        -   然后是层归一化 (Layer Normalization) 和 Leaky ReLU 激活函数。
        -   每个块处理后，基础通道数 `channel` 会翻倍，为下一个块的卷积层提供更多的通道。
    3.  最终卷积层: 在循环之后，还有若干卷积层用于进一步处理特征，并将特征图的通道数逐步减少到1，
        从而得到最终的 logit 输出。
    """
    # 初始化基础通道数，这里是传入的 ch 的一半
    # 构思说明: 初始通道数设置得相对较小，然后在后续层中逐渐增加。
    channel = ch // 2

    # 使用指定的变量作用域 (scope) 和复用设置 (reuse)
    with tf.variable_scope(scope, reuse=reuse):
        # 初始卷积层
        # 输入图像 -> Conv(kernel=3, stride=1, channels=channel) -> Leaky ReLU
        # tools.ops.conv 是一个自定义的卷积操作封装，可能包含了特定的初始化或 padding 策略。
        # sn=sn 表示根据参数决定是否对该卷积层使用谱归一化。
        x = conv(x_init, channel, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='conv_0')
        x = lrelu(x, 0.2) # Leaky ReLU 激活，alpha=0.2

        # 循环构建主要的判别器块 (n_dis-1 次循环，因为 i 从 1 开始)
        # 结构划分: 下采样与特征提取块
        for i in range(1, n_dis): # 例如，如果 n_dis=3, 则循环执行 i=1, i=2 两次
            # 块内的第一个卷积: 下采样层
            # Conv(kernel=3, stride=2, channels=channel*2) -> Leaky ReLU
            # 步长为2，使特征图尺寸减半，感受野扩大。通道数增加。
            x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=False, sn=sn, scope='conv_s2_' + str(i))
            x = lrelu(x, 0.2)

            # 块内的第二个卷积: 特征提取层
            # Conv(kernel=3, stride=1, channels=channel*4) -> LayerNorm -> Leaky ReLU
            # 步长为1，保持特征图尺寸不变。通道数进一步增加。
            x = conv(x, channel * 4, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='conv_s1_' + str(i))
            # tools.ops.layer_norm 是自定义的层归一化操作。
            x = layer_norm(x, scope='1_norm_' + str(i))
            x = lrelu(x, 0.2)

            # 为下一个块更新基础通道数 (翻倍)
            channel = channel * 2
            # 构思说明: 随着网络的加深和特征图尺寸的减小，通常会增加通道数以保持信息的容量。

        # 循环结束后的附加卷积层
        # 结构划分: 最终特征处理与 Logit 输出
        # Conv(kernel=3, stride=1, channels=channel*2) -> LayerNorm -> Leaky ReLU
        # 这里的 channel 是最后一次循环更新后的值。
        x = conv(x, channel * 2, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='last_conv')
        x = layer_norm(x, scope='2_ins_norm') # 注意: scope 名称中包含 'ins_norm'，但实际调用的是 layer_norm。
                                             # 这可能是笔误或历史遗留，实际行为是层归一化。
        x = lrelu(x, 0.2)

        # 最终卷积层，输出 Logit
        # Conv(kernel=3, stride=1, channels=1)
        # 将特征图的通道数降为1，得到一个二维的 logit 图 (每个空间位置一个值)。
        # 在某些 GAN 实现中，这里可能会接一个全局平均池化或 Flatten 层，以得到单个标量 logit。
        # 但在此模型中，似乎是直接输出一个与输入空间维度相关的 logit 图。
        # 具体的损失函数计算方式会决定如何处理这个 logit 图。
        x = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, sn=sn, scope='D_logit')

        return x # 返回最终的 logit 输出
