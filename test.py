# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本用于使用预训练的 AnimeGANv2 生成器模型进行推理。
它的主要功能是加载一个已训练好的模型检查点 (checkpoint)，
处理指定测试目录中的图像，并将转换后的动漫风格图像保存到指定的输出目录。

使用方式:
此脚本通过命令行运行。用户需要提供必要的参数，如模型检查点路径、
测试图像目录路径以及保存结果的目录名。
例如:
`python test.py --checkpoint_dir checkpoint/generator_Hayao_weight --test_dir dataset/test/HR_photo --save_dir Hayao/HR_photo`

上述命令会加载 `checkpoint/generator_Hayao_weight` 中的模型，
处理 `dataset/test/HR_photo/` 目录下的所有图像，
并将结果保存在 `results/Hayao/HR_photo/` 目录下。
可以使用 `--if_adjust_brightness` 参数来控制是否根据原图调整输出图像的亮度。
"""
import argparse
from tools.utils import *
import os
from tqdm import tqdm # 用于显示进度条
from glob import glob # 用于查找文件路径
import time
import numpy as np
from net import generator # 导入生成器网络定义

# 设置可见的 CUDA 设备，通常用于指定使用哪块 GPU。
# "0" 表示使用第一块 GPU。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    """
    定义并解析命令行参数。

    返回:
    - args (argparse.Namespace): 一个包含所有已解析命令行参数的对象。
    """
    desc = "AnimeGANv2 推理脚本" # argparse 解析器的描述信息
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/'+'generator_Shinkai_weight',
                        help='包含预训练生成器模型检查点 (checkpoint) 的目录路径。')
    parser.add_argument('--test_dir', type=str, default='dataset/test/t',
                        help='包含待转换的输入测试图像的目录路径。')
    parser.add_argument('--save_dir', type=str, default='Shinkai/t',
                        help='用于在 `./results/` 目录下创建子目录以保存风格化图像的名称。例如，若提供 "MyStyle/MyPhotos"，则结果保存在 "./results/MyStyle/MyPhotos"。')
    parser.add_argument('--if_adjust_brightness', type=bool, default=True,
                        help='是否根据原始照片调整输出图像的亮度。如果为 True，输出图像亮度会向原图靠拢。')
    # 注意: 原版代码中没有对这里的参数进行校验 (如 check_args)，实际使用时可以按需添加。
    return parser.parse_args()

def stats_graph(graph):
    """
    计算并打印模型的统计信息，如 FLOPs (浮点运算次数)。
    注意: 原始代码中计算参数量 (params) 的部分被注释掉了。

    参数:
    - graph (tf.Graph): 当前的 TensorFlow 计算图。
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    # params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()) # 参数量计算被注释
    print('FLOPs (浮点运算次数): {}'.format(flops.total_float_ops))

def test(checkpoint_dir, style_name, test_dir, if_adjust_brightness, img_size=[256,256]):
    """
    执行模型推理（测试）的主要函数。

    该函数负责:
    1. 构建生成器的 TensorFlow 计算图。
    2. 加载预训练的生成器权重。
    3. 遍历测试目录中的所有图像。
    4. 对每张图像执行风格转换。
    5. 保存转换后的图像，并可选择进行亮度调整。

    参数:
    - checkpoint_dir (str): 预训练模型检查点所在的目录。
    - style_name (str): 用于在 `results/` 目录下创建保存结果的子目录名称 (来源于 `args.save_dir`)。
    - test_dir (str): 包含输入测试图像的目录。
    - if_adjust_brightness (bool): 是否调整输出图像的亮度。
    - img_size (list, 可选): 加载测试图像时调整到的尺寸 [高度, 宽度]。
                             注意: 生成器本身在定义时使用了 `None` 作为图像尺寸占位符，
                             理论上可以处理不同尺寸的输入，但 `load_test_data` 可能会对图像进行缩放。
                             默认: [256, 256]。
    """
    # tf.reset_default_graph() # 重置默认计算图。在某些情况下（例如在同一个 Python 进程中多次调用 test 函数）可能有用，
                             # 但对于单次运行的脚本，通常不是必需的。原代码中已注释掉。

    # 构建保存结果的目录路径
    result_dir = 'results/'+style_name
    check_folder(result_dir) # 确保结果目录存在，如果不存在则创建

    # 获取测试目录中所有文件的路径 (支持多种图像格式，如 jpg, png)
    test_files = glob('{}/*.*'.format(test_dir))

    # 定义 TensorFlow placeholder，用于输入测试图像。
    # [1, None, None, 3] 表示批大小为1, 高度和宽度不固定, 通道数为3 (RGB)。
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test_input_real_image')

    # 构建生成器网络
    # 使用 tf.variable_scope 指定变量的作用域为 "generator"。
    # reuse=False 表示这是第一次创建该作用域下的变量 (如果是 True，则会尝试复用已存在的变量)。
    # 对于测试，通常我们希望加载已训练的权重，但图的构建本身是首次的。
    # 权重加载是通过 Saver 对象在会话中完成的。
    with tf.variable_scope("generator", reuse=False): # 注意：这里的 reuse 应该为 False 或 tf.AUTO_REUSE，因为权重是通过 saver.restore 加载的。
                                                  # 如果设为 True 且之前没有定义过 "generator" 作用域的变量，可能会出错。
                                                  # 然而，AnimeGANv2.py 中的 generator 定义并没有显式地创建新变量，而是复用了 G_net 内部的。
                                                  # 此处保持原样，但需注意其行为。
        test_generated = generator.G_net(test_real).fake # 获取生成器的输出张量 (伪造的动漫图像)

    # 创建 Saver 对象，用于加载模型权重
    saver = tf.train.Saver()

    # 配置 TensorFlow 会话 (Session)
    gpu_options = tf.GPUOptions(allow_growth=True) # 允许 GPU 显存按需增长
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        # sess.run(tf.global_variables_initializer()) # 不需要全局变量初始化，因为我们要加载预训练模型

        # 加载模型检查点
        print(f" [*] 正在从 '{checkpoint_dir}' 读取检查点...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # 获取检查点状态信息
        if ckpt and ckpt.model_checkpoint_path: # 如果检查点存在
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # 获取最新的检查点文件名
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name)) # 从检查点恢复模型变量
            print(" [*] 成功读取检查点: {}".format(os.path.join(checkpoint_dir, ckpt_name)))
        else:
            print(" [!] 未找到检查点，请检查 --checkpoint_dir 参数路径是否正确。")
            return # 加载失败，则退出函数

        # (可选) 打印模型统计信息，如 FLOPs。原代码中注释掉了。
        # stats_graph(tf.get_default_graph())

        # 开始计时
        begin = time.time()
        # 遍历所有测试文件并进行处理
        for sample_file in tqdm(test_files, desc="正在处理图像"): # tqdm 用于显示进度条
            # print('正在处理图像: ' + sample_file) # 使用 tqdm 后可以不用手动打印

            # 加载测试图像并转换为 NumPy 数组
            sample_image = np.asarray(load_test_data(sample_file, img_size))
            # 构建输出图像的完整保存路径
            image_path = os.path.join(result_dir,'{0}'.format(os.path.basename(sample_file)))

            # 运行 TensorFlow 会话，执行生成器网络进行推理
            fake_img = sess.run(test_generated, feed_dict = {test_real : sample_image})

            # 保存生成的图像
            if if_adjust_brightness:
                # 如果启用了亮度调整，则传入原始图像路径以进行参考
                save_images(fake_img, image_path, sample_file)
            else:
                # 否则，不进行亮度调整
                save_images(fake_img, image_path, None)

        # 结束计时并打印处理时间信息
        end = time.time()
        print(f'总测试耗时 (test-time): {end-begin:.4f} 秒 (s)')
        if test_files: # 避免除以零错误
            print(f'单张图像平均测试耗时 (one image test time): {(end-begin)/len(test_files):.4f} 秒 (s)')
        else:
            print("未找到测试图像。")

if __name__ == '__main__':
    # 当脚本作为主程序执行时
    arg = parse_args() # 解析命令行参数
    if arg is None:
        print("参数解析失败，程序退出。")
        exit()

    print(f"使用的检查点目录 (Checkpoint directory): {arg.checkpoint_dir}")
    print(f"测试图像目录 (Test directory): {arg.test_dir}")
    print(f"结果保存目录 (Save directory within ./results/): {arg.save_dir}")
    print(f"是否调整亮度 (Adjust brightness): {arg.if_adjust_brightness}")

    # 调用测试函数
    test(arg.checkpoint_dir, arg.save_dir, arg.test_dir, arg.if_adjust_brightness)
    print(" [*] 推理完成! (Inference finished!)")
