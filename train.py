# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本是训练 AnimeGANv2 模型的主要入口点。它负责以下主要任务：
1.  解析用户通过命令行传入的参数。
2.  配置 TensorFlow 会话 (Session)，例如 GPU 使用选项。
3.  实例化在 `AnimeGANv2.py` 文件中定义的 `AnimeGANv2` 类。
4.  调用 `AnimeGANv2` 实例的方法来构建计算图并启动训练过程。

使用方式:
此脚本通过命令行运行。用户可以指定不同的参数来自定义训练过程。
例如:
`python train.py --dataset Hayao --epoch 101 --batch_size 8 --g_lr 0.0001 ...`

关键的命令行参数包括数据集名称、训练轮数、批处理大小、学习率、损失函数权重等。
请使用 `python train.py --help` 查看所有可用参数及其说明。
"""
from AnimeGANv2 import AnimeGANv2
import argparse
from tools.utils import *
import os

# 设置可见的 CUDA 设备，通常用于指定使用哪块 GPU。
# "0" 表示使用第一块 GPU。如果有多块 GPU，可以指定如 "0,1"。
# 在单 GPU 环境下，此设置可能不是必需的，但有助于在多 GPU 环境中管理资源。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    """
    定义并解析命令行参数。

    返回:
    - args (argparse.Namespace): 一个包含所有已解析命令行参数的对象。
                                 如果参数校验失败 (由 `check_args` 处理)，可能返回 None。
    """
    desc = "AnimeGANv2 训练脚本" # argparse 解析器的描述信息
    parser = argparse.ArgumentParser(description=desc)

    # 数据集与训练周期参数
    parser.add_argument('--dataset', type=str, default='Hayao',
                        help='指定使用的数据集名称 (例如: Hayao, Paprika, Shinkai)。默认: Hayao')
    parser.add_argument('--epoch', type=int, default=101,
                        help='总训练轮数。默认: 101')
    parser.add_argument('--init_epoch', type=int, default=10,
                        help='初始生成器预训练的轮数 (仅使用内容损失)。默认: 10')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='每批训练的图像数量。如果使用轻量版模型，可以适当调大。默认: 12')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='保存模型检查点的频率 (以轮为单位)。默认: 1')

    # 学习率参数
    parser.add_argument('--init_lr', type=float, default=2e-4,
                        help='初始生成器预训练阶段的学习率。默认: 2e-4')
    parser.add_argument('--g_lr', type=float, default=2e-5,
                        help='生成器在对抗训练阶段的学习率。默认: 2e-5')
    parser.add_argument('--d_lr', type=float, default=4e-5,
                        help='判别器在对抗训练阶段的学习率。默认: 4e-5')

    # 损失函数相关权重和参数
    parser.add_argument('--ld', type=float, default=10.0,
                        help='梯度惩罚 (GP) 或 Lipschitz 惩罚 (LP) 的系数 lambda。默认: 10.0')
    parser.add_argument('--g_adv_weight', type=float, default=300.0,
                        help='生成器的对抗性损失权重。默认: 300.0')
    parser.add_argument('--d_adv_weight', type=float, default=300.0,
                        help='判别器的对抗性损失权重。默认: 300.0')
    parser.add_argument('--con_weight', type=float, default=1.5,
                        help='VGG19 内容损失的权重 (例如: Hayao=1.5, Paprika=2.0, Shinkai=1.2)。默认: 1.5')
    parser.add_argument('--sty_weight', type=float, default=2.5,
                        help='风格损失的权重 (例如: Hayao=2.5, Paprika=0.6, Shinkai=2.0)。默认: 2.5')
    parser.add_argument('--color_weight', type=float, default=10.,
                        help='色彩保留损失的权重 (例如: Hayao=15, Paprika=50, Shinkai=10)。默认: 10.0')
    parser.add_argument('--tv_weight', type=float, default=1.,
                        help='全变分损失 (Total Variation Loss) 的权重，用于平滑图像 (例如: Hayao=1, Paprika=0.1, Shinkai=1)。默认: 1.0')

    # GAN 配置参数
    parser.add_argument('--training_rate', type=int, default=1,
                        help='生成器 (G) 与判别器 (D) 的训练次数比例 (G:D)。例如，1 表示 G 训练1次，D 训练1次。默认: 1')
    parser.add_argument('--gan_type', type=str, default='lsgan',
                        help='GAN 损失函数的类型。可选: [gan, lsgan, wgan-gp, wgan-lp, dragan, hinge]。默认: lsgan')

    # 图像属性参数
    parser.add_argument('--img_size', type=list, default=[256,256],
                        help='输入图像的尺寸 [高度, 宽度]。默认: [256, 256]')
    parser.add_argument('--img_ch', type=int, default=3,
                        help='输入图像的通道数 (例如 RGB图像为3)。默认: 3')

    # 网络结构参数
    parser.add_argument('--ch', type=int, default=64,
                        help='网络层中卷积核数量的基础值。默认: 64')
    parser.add_argument('--n_dis', type=int, default=3,
                        help='判别器中的主要卷积层/残差块数量。默认: 3')
    parser.add_argument('--sn', type=str2bool, default=True,
                        help='是否在判别器中使用谱归一化 (Spectral Normalization)。默认: True')

    # 目录路径参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='保存模型检查点的目录名称。默认: checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='保存 TensorBoard 训练日志的目录名称。默认: logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='训练过程中保存生成样本图像的目录名称。默认: samples')

    return check_args(parser.parse_args())


def check_args(args):
    """
    校验解析后的命令行参数的有效性，并创建必要的目录。

    参数:
    - args (argparse.Namespace): `parse_args()` 函数返回的已解析参数对象。

    返回:
    - args (argparse.Namespace): 校验通过的参数对象。如果校验失败，程序可能会提前退出。
    """
    # 检查并创建检查点目录
    check_folder(args.checkpoint_dir)

    # 检查并创建日志目录
    check_folder(args.log_dir)

    # 检查并创建样本目录
    check_folder(args.sample_dir)

    # 校验 epoch 参数
    try:
        assert args.epoch >= 1 # 总轮数必须大于等于1
    except:
        print('错误: 训练总轮数 (epoch) 必须大于等于 1')
        return None # 参数错误，可以考虑返回 None 或直接退出

    # 校验 batch_size 参数
    try:
        assert args.batch_size >= 1 # 批大小必须大于等于1
    except:
        print('错误: 批处理大小 (batch_size) 必须大于等于 1')
        return None # 参数错误

    return args


def main():
    """
    主执行函数。
    该函数完成参数解析、TensorFlow 会话设置、模型实例化、计算图构建和训练启动。
    """
    # 解析命令行参数
    args = parse_args()
    if args is None: # 如果参数解析或校验失败
      print("参数错误，程序退出。")
      exit()

    # 配置 TensorFlow 会话 (Session)
    gpu_options = tf.GPUOptions(allow_growth=True) # 允许 GPU 显存按需增长，避免一开始就占用所有显存
    # allow_soft_placement=True: 如果指定的设备不可用，允许 TensorFlow 自动选择一个可用的设备。
    # inter_op_parallelism_threads 和 intra_op_parallelism_threads: 控制操作间和操作内的并行度。
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                           inter_op_parallelism_threads=8, # 通常可以根据 CPU核心数设置
                                           intra_op_parallelism_threads=8, # 通常可以根据 CPU核心数设置
                                           gpu_options=gpu_options)) as sess:
        # 实例化 AnimeGANv2 模型
        gan = AnimeGANv2(sess, args)

        # 构建模型的计算图 (包括生成器、判别器、损失函数、优化器等)
        gan.build_model()

        # (可选) 显示网络中的所有可训练变量及其形状
        show_all_variables()

        # 启动模型训练过程
        gan.train()
        print(" [*] 训练完成! (Training finished!)")


if __name__ == '__main__':
    # 当脚本作为主程序执行时，调用 main() 函数
    main()
