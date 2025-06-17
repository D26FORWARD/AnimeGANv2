# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本用于从一个完整的 AnimeGANv2 训练检查点 (checkpoint) 中提取并单独保存生成器 (Generator) 的权重。
完整的训练检查点通常包含生成器、判别器 (Discriminator) 的权重，以及优化器 (Optimizer) 的状态等信息。
然而，在进行推理（如图像风格转换或视频处理）时，通常只需要生成器的权重。
此脚本通过创建一个仅包含生成器参数的较小的检查点文件，方便了模型的部署和推理过程。

使用方式:
此脚本通过命令行运行。用户需要指定包含完整训练检查点的目录路径，以及一个用于命名输出的风格名称。
例如:
`python tools/get_generator_ckpt.py --checkpoint_dir ../checkpoint/AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1 --style_name Hayao`

执行后，脚本会在 `../checkpoint/` 目录下创建一个名为 `generator_<style_name>_weight` 的新目录，
并在其中保存仅包含生成器权重的检查点文件，文件名通常会包含原始检查点的步数信息 (例如 `Hayao-100.ckpt`)。
"""
import argparse
from tools.utils import * # 导入工具函数，如 check_folder
import os
from net import generator # 导入生成器网络定义

# 设置可见的 CUDA 设备，通常用于指定使用哪块 GPU。
# "0" 表示使用第一块 GPU。如果只有 CPU 或不关心特定 GPU，此设置可能影响不大或可以调整。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    """
    定义并解析命令行参数。

    返回:
    - args (argparse.Namespace): 一个包含所有已解析命令行参数的对象。
    """
    desc = "AnimeGANv2 - 提取生成器权重脚本" # argparse 解析器的描述信息
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--checkpoint_dir', type=str,
                        default='../checkpoint/' + 'AnimeGANv2_Hayao_lsgan_300_300_1_2_10_1', # 默认的完整检查点路径示例
                        help='包含完整 AnimeGANv2 训练检查点的目录路径。')
    parser.add_argument('--style_name', type=str, default='Hayao',
                        help='风格名称 (例如: Hayao, Paprika, Shinkai)。此名称将用于命名输出的生成器权重目录和检查点文件。')

    return parser.parse_args()

def save(saver, sess, checkpoint_dir, model_name):
    """
    使用提供的 Saver 对象保存当前 TensorFlow 会话中的变量到指定的检查点文件。

    参数:
    - saver (tf.train.Saver): 一个 `tf.train.Saver` 实例，它知道要保存哪些变量。
                              在此脚本中，它被配置为仅保存生成器的变量。
    - sess (tf.Session): 当前的 TensorFlow 会话，其中包含了已加载（或即将加载）的变量值。
    - checkpoint_dir (str): 保存检查点文件的目标目录路径。
    - model_name (str): 检查点文件的基础名称 (不含扩展名，例如 'Hayao-100')。

    返回:
    - str: 保存的检查点文件的完整路径。
    """
    # 构建检查点文件的完整保存路径
    save_path = os.path.join(checkpoint_dir, model_name + '.ckpt')
    # 调用 saver.save() 方法保存变量。
    # write_meta_graph=True 表示同时保存计算图的元数据 (.meta 文件)，这对于后续恢复模型结构可能有用。
    saver.save(sess, save_path, write_meta_graph=True)
    return  save_path

def main(checkpoint_dir, style_name):
    """
    执行提取生成器权重的主要逻辑。

    步骤:
    1.  构造并创建用于保存提取出权重的输出目录。
    2.  构建一个临时的（“虚拟的”）生成器计算图。这是为了在当前图中定义与原始训练时相同的生成器变量结构。
    3.  从当前计算图中筛选出所有属于生成器作用域 (`generator`) 的可训练变量。
    4.  创建一个 `tf.train.Saver` 对象，并指定它只处理上一步筛选出的生成器变量。
    5.  启动一个 TensorFlow 会话。
    6.  使用这个特制的 Saver 从原始的、完整的检查点文件中加载权重。由于 Saver 只关心生成器变量，
        因此只有生成器的权重会被加载到当前计算图的对应变量中。
    7.  从加载的检查点文件名中提取训练步数（或轮数）。
    8.  调用自定义的 `save` 函数，使用特制的 Saver 将当前会话中的生成器变量保存到一个新的检查点文件中，
        文件名中包含风格名称和原始步数。

    参数:
    - checkpoint_dir (str): 包含完整训练检查点的目录路径。
    - style_name (str): 风格名称，用于构造输出路径和文件名。
    """
    # 构造输出目录路径，例如: ../checkpoint/generator_Hayao_weight
    output_ckpt_dir = '../checkpoint/' + 'generator_' + style_name + '_weight'
    check_folder(output_ckpt_dir) # 确保输出目录存在，如果不存在则创建

    # 步骤 2: 构建临时的生成器计算图
    # 定义一个输入 placeholder，尺寸与推理时类似 ([1, None, None, 3] 表示批大小为1，宽高可变，3通道)
    placeholder = tf.placeholder(tf.float32, [1, None, None, 3], name='generator_input_dummy')
    # 在 "generator" 变量作用域内实例化生成器网络。
    # `reuse=False` (或 tf.AUTO_REUSE) 确保变量被创建。
    # 网络输出 (`_`) 在这里不重要，重要的是定义了生成器的变量。
    with tf.variable_scope("generator", reuse=False): # 确保变量被创建或可以被创建
        _ = generator.G_net(placeholder).fake

    # 步骤 3: 筛选生成器变量
    # 从图中所有可训练变量中，筛选出名称以 "generator" 开头的变量。
    generator_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]

    # 步骤 4: 创建 Saver 对象，仅用于处理生成器变量
    # 当调用 restore 时，它会尝试从检查点中找到这些变量并加载它们的值。
    # 当调用 save 时，它只会保存这些变量的值。
    saver_for_generator = tf.train.Saver(generator_vars)

    # 步骤 5: TensorFlow 会话配置和启动
    gpu_options = tf.GPUOptions(allow_growth=True) # 允许 GPU 显存按需增长
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        # 初始化图中的所有变量 (特别是新创建的 placeholder 和 generator 网络中的变量)
        sess.run(tf.global_variables_initializer())

        # 步骤 6: 从完整检查点加载权重到当前图的生成器变量中
        print(f" [*] 正在从 '{checkpoint_dir}' 读取完整检查点...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # 获取检查点状态信息
        if ckpt and ckpt.model_checkpoint_path: # 如果检查点存在
            print(f"找到的检查点文件: {ckpt.model_checkpoint_path}")
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # 获取最新的检查点文件名
            # 使用只包含生成器变量的 saver_for_generator 来恢复。
            # TensorFlow 会智能地只加载 `generator_vars` 中存在的变量。
            saver_for_generator.restore(sess, os.path.join(checkpoint_dir, ckpt_name))

            # 步骤 7: 提取训练步数
            # 检查点文件名通常格式为 "model_name.ckpt-step_number" 或 "model_name-step_number"
            # 例如 "AnimeGANv2.model-100" -> counter = "100"
            step_counter = ckpt_name.split('-')[-1]
            # 如果 .ckpt 在中间，例如 "Hayao.ckpt-100"，则需要进一步处理
            if '.ckpt' in step_counter:
                step_counter = step_counter.split('.ckpt')[0]

            print(f" [*] 成功从 '{ckpt_name}' 加载生成器权重。提取的步数为: {step_counter}")
        else:
            print(f" [!] 在 '{checkpoint_dir}' 未找到检查点。")
            return # 加载失败，则退出函数

        # 步骤 8: 保存仅包含生成器权重的新的检查点文件
        # 新的文件名格式: <style_name>-<step_counter>.ckpt
        output_model_name = f"{style_name}-{step_counter}"
        saved_path = save(saver_for_generator, sess, output_ckpt_dir, output_model_name)

        print(f'[*] 生成器权重提取并保存完成: {saved_path} ')


if __name__ == '__main__':
    # 当脚本作为主程序执行时
    arg = parse_args() # 解析命令行参数
    if arg is None:
        print("参数解析失败，程序退出。")
        exit()

    print(f"源检查点目录 (Source checkpoint directory): {arg.checkpoint_dir}")
    print(f"风格名称 (Style name): {arg.style_name}")

    # 调用主处理函数
    main(arg.checkpoint_dir, arg.style_name)
