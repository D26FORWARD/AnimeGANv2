# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本定义了 AnimeGANv2 的核心类 `AnimeGANv2`。该类封装了整个生成对抗网络 (GAN) 的结构，
包括生成器 (Generator) 和判别器 (Discriminator)，以及训练过程、损失函数定义、模型保存与加载等关键逻辑。
它是 AnimeGANv2 实现的核心部分。

使用方式:
此文件不应直接运行。`AnimeGANv2` 类主要由其他驱动脚本 (如 `train.py` 用于模型训练，
`test.py` 或 `video2anime.py` 用于图像或视频的风格转换) 实例化并调用其方法。
"""
from tools.ops import *
from tools.utils import *
from glob import glob
import time
import numpy as np
from net import generator
from net.discriminator import D_net
from tools.data_loader import ImageGenerator
from tools.vgg19 import Vgg19

class AnimeGANv2(object) :
    """
    AnimeGANv2 类:
    管理 AnimeGANv2 模型的所有方面，包括其架构定义、训练流程和推理组件。

    主要功能:
    - 初始化模型超参数。
    - 构建生成器和判别器网络。
    - 定义各种损失函数 (内容损失, 风格损失, 对抗性损失等)。
    - 实现训练循环，包括数据加载、模型优化、检查点保存和样本生成。
    - 提供模型加载和保存的功能。
    """
    def __init__(self, sess, args):
        """
        初始化 AnimeGANv2 模型实例。

        参数:
        - sess: TensorFlow `Session` 对象，用于执行计算图。
        - args:  一个包含各种配置参数的命名空间或字典对象，通常由 `argparse` 解析命令行参数得到。
                 应包含以下关键参数:
                 - dataset (str): 数据集名称 (例如 'Hayao', 'Paprika')。
                 - epoch (int): 总训练轮数。
                 - init_epoch (int): 初始生成器预训练轮数。
                 - batch_size (int): 每批训练的图像数量。
                 - save_freq (int): 保存模型检查点的频率 (以轮为单位)。
                 - init_lr (float): 初始生成器训练阶段的学习率。
                 - d_lr (float): 判别器学习率。
                 - g_lr (float): 生成器学习率 (对抗训练阶段)。
                 - g_adv_weight (float): 生成器对抗性损失的权重。
                 - d_adv_weight (float): 判别器对抗性损失的权重。
                 - con_weight (float): 内容损失权重。
                 - sty_weight (float): 风格损失权重。
                 - color_weight (float): 色彩损失权重。
                 - tv_weight (float): 全变分损失权重。
                 - training_rate (int): 生成器相对于判别器的训练频率 (例如，训练 G `training_rate` 次，再训练 D 1 次)。
                 - ld (float): 梯度惩罚 (GP) 或 Lipschitz 惩罚 (LP) 的系数。
                 - img_size (list/tuple): 输入图像的尺寸 [高度, 宽度]。
                 - img_ch (int): 输入图像的通道数 (例如 3 对于 RGB 图像)。
                 - n_dis (int): 判别器中残差块或类似结构的数量。
                 - ch (int): 网络中卷积层的基础通道数。
                 - sn (bool): 是否在判别器中使用谱归一化 (Spectral Normalization)。
                 - checkpoint_dir (str): 保存模型检查点的目录。
                 - log_dir (str): 保存 TensorBoard 日志的目录。
                 - sample_dir (str): 保存训练过程中生成的样本图像的目录。
                 - gan_type (str): GAN 的类型 (例如 'lsgan', 'wgan-gp', 'dragan')。
        """
        self.model_name = 'AnimeGANv2' # 模型名称
        self.sess = sess # TensorFlow 会话
        self.checkpoint_dir = args.checkpoint_dir # 检查点保存目录
        self.log_dir = args.log_dir # TensorBoard 日志目录
        self.dataset_name = args.dataset # 数据集名称

        self.epoch = args.epoch # 总训练轮数
        self.init_epoch = args.init_epoch # 初始预训练轮数 (例如 args.epoch // 20)

        self.gan_type = args.gan_type # GAN 类型 (如 'lsgan', 'wgan-gp')
        self.batch_size = args.batch_size # 每批图像数量
        self.save_freq = args.save_freq # 模型保存频率

        self.init_lr = args.init_lr # 初始学习率 (用于生成器预训练)
        self.d_lr = args.d_lr # 判别器学习率
        self.g_lr = args.g_lr # 生成器学习率 (用于对抗训练)

        """ 损失权重 (Weight) """
        self.g_adv_weight = args.g_adv_weight # 生成器对抗损失权重
        self.d_adv_weight = args.d_adv_weight # 判别器对抗损失权重
        self.con_weight = args.con_weight # 内容损失权重
        self.sty_weight = args.sty_weight # 风格损失权重
        self.color_weight = args.color_weight # 色彩损失权重
        self.tv_weight = args.tv_weight # 全变分损失权重

        self.training_rate = args.training_rate # G 和 D 的训练频率比 (G:D)
        self.ld = args.ld # 梯度惩罚系数 (lambda)

        self.img_size = args.img_size # 图像尺寸 [H, W]
        self.img_ch = args.img_ch # 图像通道数

        """ 判别器 (Discriminator) 相关参数 """
        self.n_dis = args.n_dis # 判别器中的层数或块数
        self.ch = args.ch # 卷积层基础通道数
        self.sn = args.sn # 是否使用谱归一化

        # 构造样本保存目录路径
        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir) # 确保目录存在

        # 定义 TensorFlow 占位符 (Placeholders)，用于在运行时送入数据
        # 真实照片输入
        self.real = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='real_A')
        # 动漫风格参考图像
        self.anime = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='anime_A')
        # 边缘平滑后的动漫风格图像 (用于判别器)
        self.anime_smooth = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='anime_smooth_A')
        # 测试时输入的真实照片 (批大小为1, 尺寸不固定)
        self.test_real = tf.placeholder(tf.float32, [1, None, None, self.img_ch], name='test_input')
        # 灰度化的动漫风格参考图像 (用于内容和风格损失计算)
        self.anime_gray = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch],name='anime_B')

        # 初始化数据加载器 (ImageGenerator)
        # 真实照片数据加载器
        self.real_image_generator = ImageGenerator('./dataset/train_photo', self.img_size, self.batch_size)
        # 动漫风格图像数据加载器
        self.anime_image_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/style'), self.img_size, self.batch_size)
        # 动漫风格平滑图像数据加载器
        self.anime_smooth_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/smooth'), self.img_size, self.batch_size)
        # 数据集中的最大图像数量，用于确定每轮的步数
        self.dataset_num = max(self.real_image_generator.num_images, self.anime_image_generator.num_images)

        # 初始化 VGG19 网络，用于计算感知损失 (内容和风格损失)
        self.vgg = Vgg19()

        # 打印模型配置信息
        print()
        print("##### 模型配置信息 (Information) #####")
        print("# GAN 类型 (gan type) : ", self.gan_type)
        print("# 数据集 (dataset) : ", self.dataset_name)
        print("# 最大数据集数量 (max dataset number) : ", self.dataset_num)
        print("# 批大小 (batch_size) : ", self.batch_size)
        print("# 训练总轮数 (epoch) : ", self.epoch)
        print("# 初始预训练轮数 (init_epoch) : ", self.init_epoch)
        print("# 训练图像尺寸 (training image size [H, W]) : ", self.img_size)
        print("# 损失权重 (g_adv, d_adv, con, sty, color, tv) : ", self.g_adv_weight,self.d_adv_weight,self.con_weight,self.sty_weight,self.color_weight,self.tv_weight)
        print("# 学习率 (init_lr, g_lr, d_lr) : ", self.init_lr,self.g_lr,self.d_lr)
        print(f"# 训练频率 G:D (training_rate G -- D): {self.training_rate} : 1" )
        print()

    ##################################################################################
    # 生成器 (Generator)
    ##################################################################################

    def generator(self, x_init, reuse=False, scope="generator"):
        """
        定义生成器网络。

        参数:
        - x_init: 输入的真实图像张量 (tf.Tensor)。
        - reuse (bool): 是否复用已存在的变量作用域 (variable scope)。
                       在构建模型用于测试时，应设为 True 以加载已训练的权重。
        - scope (str): TensorFlow 变量作用域的名称。

        返回:
        - G.fake: 生成器输出的动漫风格图像张量。
        """
        # 使用 tf.variable_scope 来组织生成器的变量，并允许复用
        with tf.variable_scope(scope, reuse=reuse):
            # 调用在 net/generator.py 中定义的 G_net
            G = generator.G_net(x_init)
            return G.fake # 返回生成的伪图像

    ##################################################################################
    # 判别器 (Discriminator)
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        """
        定义判别器网络。

        参数:
        - x_init: 输入的图像张量 (可以是真实动漫图、生成的动漫图或平滑动漫图)。
        - reuse (bool): 是否复用已存在的变量作用域。
        - scope (str): TensorFlow 变量作用域的名称。

        返回:
        - D: 判别器网络的实例 (来自 net/discriminator.py 中的 D_net)。
             其输出通常包含一个 logit 值 (原始预测分数) 和可能的其他中间特征。
        """
        # 使用 tf.variable_scope 来组织判别器的变量，并允许复用
        # 调用在 net/discriminator.py 中定义的 D_net
        D = D_net(x_init, self.ch, self.n_dis, self.sn, reuse=reuse, scope=scope)
        return D # 返回判别器对象

    ##################################################################################
    # 模型构建与损失定义 (Model Building & Loss Definition)
    ##################################################################################
    def gradient_panalty(self, real, fake, scope="discriminator"):
        """
        计算梯度惩罚 (Gradient Penalty, GP)。
        用于 WGAN-GP 或类似类型的 GAN，以强制执行 Lipschitz 约束，稳定训练。

        参数:
        - real: 真实的动漫图像张量。
        - fake: 生成器生成的动漫图像张量。
        - scope (str): 判别器变量作用域的名称。

        返回:
        - GP: 计算得到的梯度惩罚项 (一个标量 tf.Tensor)。
        """
        # DRAGAN 类型的梯度惩罚变体
        if self.gan_type.__contains__('dragan') :
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.) # 均匀分布的噪声
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3]) # 计算真实数据的方差
            x_std = tf.sqrt(x_var)  # 标准差，噪声的大小决定了局部区域的大小

            # 为真实样本添加扰动，构造新的样本点
            fake = real + 0.5 * x_std * eps

        # 计算真实样本和生成样本之间的插值点
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real) # (公式: x_hat = alpha * x_real + (1-alpha) * x_fake)

        # 获取插值点在判别器中的输出 logit
        logit, _ = self.discriminator(interpolated, reuse=True, scope=scope)

        # 计算判别器输出相对于插值点的梯度
        grad = tf.gradients(logit, interpolated)[0]
        grad_norm = tf.norm(flatten(grad), axis=1) # 计算梯度的 L2 范数

        GP = 0.0 # 初始化梯度惩罚项
        # WGAN-LP (Lipschitz Penalty) 类型的梯度惩罚
        if self.gan_type.__contains__('lp'):
            # ld 是惩罚系数, max(0, ||grad|| - 1)^2
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))
        # WGAN-GP 或 DRAGAN 类型的梯度惩罚
        elif self.gan_type.__contains__('gp') or self.gan_type == 'dragan' :
            # ld 是惩罚系数, (||grad|| - 1)^2
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def build_model(self):
        """
        构建完整的 AnimeGANv2 模型计算图。
        包括定义生成器、判别器实例，计算所有损失函数，以及定义优化器。
        """

        """ 步骤 1: 定义生成器和判别器实例 (Define Generator, Discriminator) """
        # 生成器处理真实照片，得到动漫风格图像
        self.generated = self.generator(self.real, scope="generator")
        # 为测试过程定义的生成器实例 (复用权重)
        self.test_generated = self.generator(self.test_real, reuse=True, scope="generator")

        # 判别器处理不同类型的输入
        # 真实动漫图像的判别器输出
        anime_logit = self.discriminator(self.anime, scope="discriminator")
        # 灰度动漫图像的判别器输出 (复用判别器权重)
        anime_gray_logit = self.discriminator(self.anime_gray, reuse=True, scope="discriminator")
        # 生成的动漫图像的判别器输出 (复用判别器权重)
        generated_logit = self.discriminator(self.generated, reuse=True, scope="discriminator")
        # 平滑处理的动漫图像的判别器输出 (复用判别器权重)
        smooth_logit = self.discriminator(self.anime_smooth, reuse=True, scope="discriminator")


        """ 步骤 2: 定义损失函数 (Define Loss) """
        # -- 计算梯度惩罚 (GP) --
        # 根据 GAN 类型选择是否计算梯度惩罚
        if self.gan_type.__contains__('gp') or self.gan_type.__contains__('lp') or self.gan_type.__contains__('dragan') :
            GP = self.gradient_panalty(real=self.anime, fake=self.generated, scope="discriminator")
        else :
            GP = 0.0 # 对于非 GP/LP/DRAGAN 类型的 GAN, GP 为 0

        # -- 初始生成器训练阶段的损失 (Init Phase Loss) --
        # 仅使用内容损失 (content loss) 来预训练生成器
        init_c_loss = con_loss(self.vgg, self.real, self.generated) # 使用 VGG19 计算真实照片和生成图像之间的内容差异
        init_loss = self.con_weight * init_c_loss # 加权的内容损失
        self.init_loss = init_loss # 保存为类的属性，供训练时使用

        # -- GAN 对抗训练阶段的损失 (GAN Phase Loss) --
        # 内容损失 (c_loss) 和风格损失 (s_loss)
        # 使用 VGG19 计算真实照片(self.real)的内容 与 灰度动漫图(self.anime_gray)的风格，并与生成图像(self.generated)比较
        c_loss, s_loss = con_sty_loss(self.vgg, self.real, self.anime_gray, self.generated)
        # 全变分损失 (tv_loss)，用于平滑生成图像，减少噪点
        tv_loss = self.tv_weight * total_variation_loss(self.generated)
        # 感知损失 (perceptual loss) 或 VGG 损失 (t_loss)
        # 包含内容损失、风格损失、色彩损失 (color_loss) 和全变分损失
        t_loss = self.con_weight * c_loss + self.sty_weight * s_loss + color_loss(self.real,self.generated) * self.color_weight + tv_loss

        # 生成器对抗损失 (g_loss)
        # 基于判别器对生成图像的判断，鼓励生成器生成更逼真的图像
        g_loss = self.g_adv_weight * generator_loss(self.gan_type, generated_logit)
        # 判别器对抗损失 (d_loss)
        # 判别器试图区分真实动漫图、灰度动漫图、生成图和平滑动漫图
        d_loss = self.d_adv_weight * discriminator_loss(self.gan_type, anime_logit, anime_gray_logit, generated_logit, smooth_logit) + GP

        # 总的生成器损失和判别器损失
        self.Generator_loss =  t_loss + g_loss # 生成器总损失 = 感知损失 + 对抗损失
        self.Discriminator_loss = d_loss      # 判别器总损失 = 对抗损失 + 梯度惩罚

        """ 步骤 3: 定义优化器 (Training Optimizers) """
        # 获取所有可训练变量
        t_vars = tf.trainable_variables()
        # 筛选出生成器的可训练变量
        G_vars = [var for var in t_vars if 'generator' in var.name]
        # 筛选出判别器的可训练变量
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        # 初始生成器训练的优化器 (Adam 优化器)
        self.init_optim = tf.train.AdamOptimizer(self.init_lr, beta1=0.5, beta2=0.999).minimize(self.init_loss, var_list=G_vars)
        # 对抗训练阶段生成器的优化器
        self.G_optim = tf.train.AdamOptimizer(self.g_lr , beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        # 对抗训练阶段判别器的优化器
        self.D_optim = tf.train.AdamOptimizer(self.d_lr , beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """ 步骤 4: 设置 TensorBoard 摘要 (Summary for TensorBoard) """
        # 记录各项损失值，用于在 TensorBoard 中可视化训练过程
        self.G_loss_summary = tf.summary.scalar("Generator_loss", self.Generator_loss) # 生成器总损失
        self.D_loss_summary = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss) # 判别器总损失

        self.G_gan_summary = tf.summary.scalar("G_gan", g_loss) # 生成器对抗损失部分
        self.G_vgg_summary = tf.summary.scalar("G_vgg", t_loss) # 生成器感知损失部分 (VGG)
        self.G_init_loss_summary = tf.summary.scalar("G_init", init_loss) # 初始生成器内容损失

        # 合并摘要，方便写入 TensorBoard
        self.V_loss_merge = tf.summary.merge([self.G_init_loss_summary]) # 初始训练阶段的摘要
        self.G_loss_merge = tf.summary.merge([self.G_loss_summary, self.G_gan_summary, self.G_vgg_summary, self.G_init_loss_summary]) # 生成器所有损失的摘要
        self.D_loss_merge = tf.summary.merge([self.D_loss_summary]) # 判别器损失的摘要

    def train(self):
        """
        执行模型训练过程。

        该方法包括:
        - 初始化 TensorFlow 变量和模型保存器 (Saver)。
        - 设置 TensorBoard 日志写入器。
        - 加载数据集。
        - 尝试从检查点恢复模型。
        - 执行主训练循环:
            - 初始生成器预训练阶段: 仅优化生成器以最小化内容损失。
            - GAN 对抗训练阶段: 交替训练判别器和生成器。
        - 记录损失值到 TensorBoard。
        - 定期保存模型检查点。
        - 定期生成并保存验证集样本图像，以监控训练效果。
        """
        # 步骤 1: 初始化 (Initialization)
        # 初始化所有 TensorFlow 全局变量
        self.sess.run(tf.global_variables_initializer())

        # 创建 Saver 对象，用于保存模型检查点，max_to_keep 控制最多保存的检查点数量
        self.saver = tf.train.Saver(max_to_keep=self.epoch)

        # 创建 TensorBoard summary writer，用于将训练摘要写入磁盘
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # 步骤 2: 数据加载 (Data Loading)
        # 从数据加载器获取下一批图像的操作 (ops)
        real_img_op, anime_img_op, anime_smooth_op  = self.real_image_generator.load_images(), self.anime_image_generator.load_images(), self.anime_smooth_generator.load_images()

        # 步骤 3: 恢复检查点 (Restore Checkpoint)
        # 尝试从指定的检查点目录加载已训练的模型
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = checkpoint_counter + 1 # 如果成功加载，则从下一个 epoch 开始训练
            print(" [*] 成功加载检查点 (Load SUCCESS)")
        else:
            start_epoch = 0 # 如果加载失败，则从头开始训练
            print(" [!] 加载检查点失败 (Load failed...)")

        # 步骤 4: 主训练循环 (Main Training Loop)
        init_mean_loss = [] # 用于存储初始训练阶段的平均损失
        mean_loss = []      # 用于存储对抗训练阶段的平均损失

        # 控制生成器和判别器的训练频率 (G : D = self.training_rate : 1)
        # j 表示生成器还需要训练多少次才轮到判别器训练
        j = self.training_rate

        # 外层循环：按轮次 (epoch) 进行训练
        for epoch in range(start_epoch, self.epoch):
            # 内层循环：按批次 (batch) 或步骤 (step) 进行训练
            for idx in range(int(self.dataset_num / self.batch_size)):
                # 获取一批训练数据
                anime, anime_smooth, real = self.sess.run([anime_img_op, anime_smooth_op, real_img_op])
                # 构建 feed_dict，用于将数据送入 TensorFlow 的 placeholder
                train_feed_dict = {
                    self.real:real[0],          # 真实照片
                    self.anime:anime[0],        # 动漫风格参考图
                    self.anime_gray:anime[1],   # 灰度动漫风格参考图 (通常与 anime[0] 对应，但经过了灰度处理)
                    self.anime_smooth:anime_smooth[1] # 边缘平滑的动漫图
                }

                # -- 阶段一: 初始生成器预训练 (Init G Phase) --
                if epoch < self.init_epoch :
                    start_time = time.time() # 记录开始时间

                    # 运行优化器，更新生成器参数以最小化初始内容损失 (self.init_loss)
                    # 同时获取真实图像、生成的图像、损失值和 TensorBoard 摘要
                    real_images, generator_images, _, v_loss, summary_str = self.sess.run(
                        [self.real, self.generated, self.init_optim, self.init_loss, self.V_loss_merge],
                        feed_dict = train_feed_dict
                    )
                    self.writer.add_summary(summary_str, epoch) # 将摘要写入 TensorBoard
                    init_mean_loss.append(v_loss) # 记录当前批次的损失

                    # 打印训练信息
                    print("轮次 (Epoch): %3d 步骤 (Step): %5d / %5d  耗时 (time): %f s 初始内容损失 (init_v_loss): %.8f  平均内容损失 (mean_v_loss): %.8f" %
                          (epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, v_loss, np.mean(init_mean_loss)))

                    # 每200步清空一次平均损失列表，重新计算
                    if (idx+1)%200 ==0:
                        init_mean_loss.clear()
                # -- 阶段二: GAN 对抗训练 (Adversarial Training Phase) --
                else :
                    start_time = time.time() # 记录开始时间

                    # 根据 training_rate 控制判别器的训练频率
                    if j == self.training_rate:
                        # 更新判别器 (Update D)
                        _, d_loss, summary_str = self.sess.run(
                            [self.D_optim, self.Discriminator_loss, self.D_loss_merge],
                            feed_dict=train_feed_dict
                        )
                        self.writer.add_summary(summary_str, epoch) # 写入判别器损失摘要
                    else:
                        # 如果不是判别器的训练轮次，d_loss 可能未定义，或者保持上一次的值
                        # 为了打印方便，可以从之前的迭代中获取 d_loss，或者在下面打印时特殊处理
                        pass # d_loss 会使用上一次的值，或者在打印时注意

                    # 更新生成器 (Update G)
                    real_images, generator_images, _, g_loss, summary_str = self.sess.run(
                        [self.real, self.generated, self.G_optim, self.Generator_loss, self.G_loss_merge],
                        feed_dict = train_feed_dict
                    )
                    self.writer.add_summary(summary_str, epoch) # 写入生成器损失摘要

                    # 记录当前批次的判别器和生成器损失
                    # 注意: d_loss 可能不是当前迭代计算的，取决于 training_rate 和 j
                    if j == self.training_rate: # 只有在 D 被训练时，d_loss 才是最新的
                        mean_loss.append([d_loss, g_loss])
                    else: # 否则，d_loss 是旧的，只记录 g_loss 的变化趋势可能更有意义
                        # 或者记录上一次的 d_loss
                        if len(mean_loss) > 0 and len(mean_loss[-1]) == 2: # 确保 mean_loss 非空且有 d_loss
                           mean_loss.append([mean_loss[-1][0], g_loss]) # 使用上一次的 d_loss
                        else:
                           mean_loss.append([0, g_loss]) # 如果没有历史 d_loss, 暂时用0


                    # 打印训练信息
                    if j == self.training_rate: # 当判别器也训练时
                        print(
                            "轮次 (Epoch): %3d 步骤 (Step): %5d / %5d  耗时 (time): %f s D损失 (d_loss): %.8f, G损失 (g_loss): %.8f -- 平均D损失 (mean_d_loss): %.8f, 平均G损失 (mean_g_loss): %.8f" % (
                                epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, d_loss, g_loss, np.mean(mean_loss, axis=0)[0],
                                np.mean(mean_loss, axis=0)[1]))
                    else: # 当只训练生成器时
                        print(
                            "轮次 (Epoch): %3d 步骤 (Step): %5d / %5d 耗时 (time): %f s , G损失 (g_loss): %.8f --  平均G损失 (mean_g_loss): %.8f" % (
                                epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, g_loss, np.mean(mean_loss, axis=0)[1]))

                    # 每200步清空一次平均损失列表
                    if (idx + 1) % 200 == 0:
                        mean_loss.clear()

                    # 更新训练频率计数器 j
                    j = j - 1
                    if j < 1: # 当 j 减到小于1时，重置为 training_rate，意味着下一轮 D 和 G 都会训练
                        j = self.training_rate

            # 步骤 5: 保存模型和生成样本 (Save Model & Generate Samples)
            # 在对抗训练阶段，并且达到指定的保存频率时，保存模型
            if (epoch + 1) >= self.init_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, epoch)

            # 在对抗训练阶段的每个 epoch 结束后，生成并保存验证集样本
            if epoch >= self.init_epoch -1: # 从 init_epoch 的最后一个 epoch 开始或之后
                val_files = glob('./dataset/{}/*.*'.format('val')) # 获取验证集所有文件路径
                save_path = './{}/{:03d}/'.format(self.sample_dir, epoch) # 构造样本保存路径
                check_folder(save_path) # 确保路径存在

                for i, sample_file in enumerate(val_files): # 遍历验证集文件
                    print('验证样本 (val): '+ str(i) + sample_file)
                    # 加载测试图像，并转换为适合模型的格式
                    sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                    # 运行模型进行推理 (使用 self.test_real 和 self.test_generated)
                    test_real_eval, test_generated_eval = self.sess.run(
                        [self.test_real, self.test_generated],
                        feed_dict = {self.test_real: sample_image}
                    )
                    # 保存原始输入图像和生成的动漫风格图像
                    save_images(test_real_eval, save_path+'{:03d}_a_real.jpg'.format(i), None) # 保存真实图像
                    save_images(test_generated_eval, save_path+'{:03d}_b_generated.jpg'.format(i), None) # 保存生成图像

    @property
    def model_dir(self):
        """
        生成并返回基于模型配置的模型目录名称字符串。
        此名称用于组织检查点和日志文件。

        返回:
        - str: 模型目录的名称。
        """
        # 目录名包含了模型名、数据集名、GAN类型以及各项损失的权重
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                                          self.gan_type,
                                                          int(self.g_adv_weight), int(self.d_adv_weight),
                                                          int(self.con_weight), int(self.sty_weight),
                                                          int(self.color_weight), int(self.tv_weight))

    def save(self, checkpoint_dir, step):
        """
        保存模型检查点。

        参数:
        - checkpoint_dir (str): 检查点保存的根目录。
        - step (int): 当前的训练步数或轮数，会附加到检查点文件名中。
        """
        # 构造特定模型的检查点目录
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir): # 如果目录不存在，则创建
            os.makedirs(checkpoint_dir)
        # 使用 Saver 对象保存会话中的所有可训练变量
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)
        print(f" [*] 模型已保存于 {checkpoint_dir}，步数为 {step} (Model saved)")

    def load(self, checkpoint_dir):
        """
        加载模型检查点。

        参数:
        - checkpoint_dir (str): 检查点所在的根目录。

        返回:
        - tuple: (bool, int)
            - bool: 是否成功加载检查点。
            - int: 加载的检查点的步数或轮数。如果加载失败，则为 0。
        """
        print(" [*] 正在读取检查点 (Reading checkpoints)...")
        # 构造特定模型的检查点目录
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        # 获取检查点状态文件，该文件记录了最新的检查点信息
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path: # 如果状态文件存在且包含模型路径
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path) # 获取最新的检查点文件名
            # 使用 Saver 对象恢复模型变量
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # 从文件名中解析出步数或轮数
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] 成功读取检查点 {} (Success to read {})".format(os.path.join(checkpoint_dir, ckpt_name), ckpt_name))
            return True, counter
        else:
            print(" [*] 未找到检查点 (Failed to find a checkpoint)")
            return False, 0
