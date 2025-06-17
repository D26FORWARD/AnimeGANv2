# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本用于将输入的视频文件转换为动漫风格。它使用预训练的 AnimeGANv2 生成器模型，
逐帧处理视频，对每一帧应用动漫风格转换，并将结果保存为一个新的视频文件。

使用方式:
此脚本通过命令行运行。用户需要指定输入视频文件的路径、包含预训练模型
的检查点目录路径，以及保存处理后视频的输出目录。
例如:
`python video2anime.py --video video/input/your_video.mp4 --checkpoint_dir checkpoint/generator_Hayao_weight --output video/output/Hayao`

上述命令会加载 `checkpoint/generator_Hayao_weight` 中的模型，
处理 `video/input/your_video.mp4` 视频，并将生成的动漫风格视频
保存在 `video/output/Hayao` 目录下。
"""
import argparse
import os
import cv2 # OpenCV 库，用于视频和图像处理
from tqdm import tqdm # 用于显示进度条
import numpy as np
import tensorflow as tf
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
    desc = "Tensorflow implementation of AnimeGANv2 (Video Conversion)" # argparse 解析器的描述信息
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--video', type=str, default='video/input/'+ '2.mp4',
                        help='输入视频文件的路径，或用于摄像头的设备号 (例如 "0")。')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint/generator_Paprika_weight',
                        help='包含预训练生成器模型检查点 (checkpoint) 的目录路径。')
    parser.add_argument('--output', type=str, default='video/output/' + 'Paprika',
                        help='保存处理后动漫风格视频的输出目录路径。')
    parser.add_argument('--output_format', type=str, default='MP4V',
                        help='用于 `cv2.VideoWriter` 的视频编码器 FourCC 代码。例如: MP4V (for .mp4), X264 (for .mp4), FMP4 (for .mkv), FLV1 (for .flv), XIVD (for .avi)。')
    """
    关于 output_format 的说明:
    - xxx.mp4 通常使用 'MP4V' 或 'X264' (H264)。
    - xxx.mkv 通常使用 'FMP4'。
    - xxx.flv 通常使用 'FLV1'。
    - xxx.avi 通常使用 'XIVD'。
    注意: FFMPEG 命令如 `ffmpeg -i input.mkv -c:v libx264 -strict -2 output.mp4`
    可以将 mkv 格式（或其他格式）转换为压缩率更高、文件更小的 mp4 文件。
    所支持的 FourCC 代码取决于系统中安装的编解码器。
    """
    return parser.parse_args()


def check_folder(path):
    """
    检查指定路径的文件夹是否存在，如果不存在则创建该文件夹。

    参数:
    - path (str): 需要检查或创建的文件夹路径。

    返回:
    - path (str): 传入的文件夹路径。
    """
    if not os.path.exists(path): # 如果路径不存在
        os.makedirs(path) # 则创建该路径对应的所有目录
        print(f"[*] 目录 '{path}' 已创建。")
    return path

def process_image(img, x32=True):
    """
    对输入的视频帧进行预处理，以适配生成器模型的输入要求。

    处理步骤包括:
    1. (可选) 将图像尺寸调整为32的倍数，这有助于某些网络架构处理。
    2. 将图像从 BGR 色彩空间 (OpenCV默认) 转换为 RGB 色彩空间。
    3. 将像素值从 [0, 255] 归一化到 [-1.0, 1.0] 范围。

    参数:
    - img (numpy.ndarray): 输入的视频帧 (通常为 BGR 格式)。
    - x32 (bool): 是否将图像尺寸调整为32的倍数。默认为 True。

    返回:
    - numpy.ndarray: 预处理后的图像，可直接作为模型输入。
    """
    h, w = img.shape[:2] # 获取图像的高度和宽度
    if x32: # 如果需要调整为32的倍数
        def to_32s(x):
            # 如果尺寸小于256，则设为256；否则，调整为不大于原尺寸的32的最大倍数。
            return 256 if x < 256 else x - x % 32
        img = cv2.resize(img, (to_32s(w), to_32s(h))) # 调整图像尺寸

    # BGR to RGB, then normalize to [-1, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def post_precess(img, wh): # 注意: 函数名存在拼写错误，应为 post_process
    """
    对生成器输出的图像帧进行后处理，以恢复为可显示的视频帧格式。
    (注意: 函数名 `post_precess` 存在拼写错误，通常应为 `post_process`)

    处理步骤包括:
    1. 将像素值从 [-1.0, 1.0] 反归一化到 [0, 255] 范围。
    2. 将数据类型转换为 `uint8`。
    3. 将图像尺寸调整回原始视频帧的宽度和高度。

    参数:
    - img (numpy.ndarray): 生成器输出的图像帧。
    - wh (tuple): 原始视频帧的 (宽度, 高度)。

    返回:
    - numpy.ndarray: 后处理完成的图像帧，可用于写入视频文件。
    """
    img = (img.squeeze() + 1.) / 2 * 255 # 反归一化: (-1,1) -> (0,2) -> (0,1) -> (0,255)
    img = img.astype(np.uint8) # 转换为无符号8位整型
    img = cv2.resize(img, (wh[0], wh[1])) # 调整回原始尺寸
    return img

def cvt2anime_video(video, output, checkpoint_dir, output_format='MP4V'):
    """
    执行视频到动漫风格转换的主要函数。

    参数:
    - video (str): 输入视频文件的路径。
    - output (str): 保存处理后视频的输出目录路径。
    - checkpoint_dir (str): 包含预训练生成器模型检查点的目录路径。
    - output_format (str, 可选): 输出视频的编码器 FourCC 代码。默认: 'MP4V'。

    返回:
    - str: 生成的动漫风格视频文件的完整路径。如果转换失败，可能返回 None 或引发异常。
    """
    # 检测 GPU 是否可用
    gpu_stat = bool(len(tf.config.experimental.list_physical_devices('GPU')))
    if gpu_stat:
        # 如果 GPU 可用，则确保 CUDA_VISIBLE_DEVICES 设置正确 (尽管已在文件顶部设置)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.GPUOptions(allow_growth=gpu_stat) # 根据 GPU 状态配置显存按需增长

    # 步骤 1: 构建 TensorFlow 计算图
    # 定义输入 placeholder，用于接收预处理后的视频帧
    # [1, None, None, 3] 表示批大小为1, 高度和宽度不固定, 通道数为3 (RGB)
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test_video_frame_input')
    # 在 "generator" 作用域内构建生成器网络
    # reuse=False 因为我们是第一次构建这个图，权重将通过 saver.restore 加载
    with tf.variable_scope("generator", reuse=False):
        test_generated = generator.G_net(test_real).fake # 获取生成器的输出张量
         
    # 创建 Saver 对象，用于加载模型权重
    saver = tf.train.Saver()

    # 步骤 2: 加载输入视频
    vid = cv2.VideoCapture(video) # 打开视频文件
    if not vid.isOpened(): # 检查视频是否成功打开
        print(f"[!] 错误: 无法打开视频文件 '{video}'")
        return None

    vid_name = os.path.basename(video) # 获取视频文件名
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # 获取视频总帧数
    fps = vid.get(cv2.CAP_PROP_FPS) # 获取视频的帧率
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) # 获取视频宽度
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 获取视频高度

    # 根据指定的 output_format 获取 FourCC 编码器代码
    fourcc_codec = cv2.VideoWriter_fourcc(*output_format)

    # 步骤 3: 设置 TensorFlow 会话并加载模型
    tfconfig = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=tfconfig) as sess:
        # 加载模型检查点
        print(f" [*] 正在从 '{checkpoint_dir}' 读取检查点...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # 获取检查点状态
        if ckpt and ckpt.model_checkpoint_path: # 如果检查点存在
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # 获取最新的检查点文件名
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name)) # 从检查点恢复模型变量
            print(" [*] 成功读取检查点: {}".format(os.path.join(checkpoint_dir, ckpt_name)))
        else:
            print(" [!] 未找到检查点，请检查 --checkpoint_dir 参数路径是否正确。")
            vid.release() # 释放视频捕获对象
            return None
         
        # 步骤 4: 设置输出视频写入器 (VideoWriter)
        # 构建输出视频文件的完整路径
        output_video_filename = vid_name.rsplit('.', 1)[0] + "_AnimeGANv2.mp4" #  确保输出是 .mp4
        output_video_path = os.path.join(output, output_video_filename)
        # 创建 VideoWriter 对象，用于将处理后的帧写入新的视频文件
        video_out = cv2.VideoWriter(output_video_path, fourcc_codec, fps, (width, height))
        if not video_out.isOpened(): # 检查 VideoWriter 是否成功打开
            print(f"[!] 错误: 无法创建输出视频文件 '{output_video_path}'")
            vid.release()
            return None

        # 步骤 5: 逐帧处理视频
        # 使用 tqdm 创建进度条
        pbar = tqdm(total=total_frames, ncols=100, unit="帧") # ncols 控制进度条宽度
        pbar.set_description(f"正在转换: {output_video_filename}")

        frame_count = 0
        while True:
            ret, frame = vid.read() # 读取一帧视频
            if not ret: # 如果 ret 为 False，表示视频已读取完毕或发生错误
                break

            # 预处理当前帧
            processed_frame = np.asarray(np.expand_dims(process_image(frame, x32=True),0)) # x32=True 可能不是所有情况下都需要
            # 通过 TensorFlow 模型进行推理
            fake_img_output = sess.run(test_generated, feed_dict={test_real: processed_frame})
            # 后处理生成的帧
            final_frame = post_precess(fake_img_output, (width, height)) # 注意：函数名拼写

            # 将处理后的帧写入输出视频 (OpenCV 使用 BGR 格式)
            video_out.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)) # 模型输出是RGB，转回BGR给OpenCV
            pbar.update(1) # 更新进度条
            frame_count += 1

        pbar.close() # 关闭进度条
        vid.release() # 释放视频捕获对象
        video_out.release() # 释放视频写入对象

        print(f"[*] 视频处理完成。总共处理 {frame_count} 帧。")
        return output_video_path

if __name__ == '__main__':
    # 当脚本作为主程序执行时
    arg = parse_args() # 解析命令行参数
    if arg is None:
        print("参数解析失败，程序退出。")
        exit()

    # 检查并创建输出目录
    check_folder(arg.output)

    print(f"输入视频 (Input video): {arg.video}")
    print(f"检查点目录 (Checkpoint directory): {arg.checkpoint_dir}")
    print(f"输出目录 (Output directory): {arg.output}")
    print(f"输出格式 (Output format FourCC): {arg.output_format}")

    # 调用视频转换函数
    output_video_info = cvt2anime_video(arg.video, arg.output, arg.checkpoint_dir, arg.output_format)

    if output_video_info:
        print(f'[*] 生成的视频已保存至 (Output video saved to): {output_video_info}')
    else:
        print("[!] 视频转换失败。")
