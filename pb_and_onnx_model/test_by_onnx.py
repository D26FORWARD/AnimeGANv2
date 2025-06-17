# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本用于测试已转换为 ONNX (Open Neural Network Exchange) 格式的 AnimeGANv2 生成器模型。
它加载一个 .onnx 模型文件，处理指定输入目录中的图像，使用 ONNX Runtime 执行推理，
并将生成的动漫风格图像保存到指定的输出目录。

作者: Xin Chen
时间: 2021/8/31 (如此信息与文件内容相关)

使用方式:
此脚本通过命令行直接运行：
`python pb_and_onnx_model/test_by_onnx.py`

在运行之前，用户需要根据实际情况修改脚本末尾 `if __name__ == '__main__':` 代码块中的
以下变量：
- `onnx_file`: ONNX 模型文件的路径。
- `input_imgs_path`: 包含待转换的输入图像的目录路径。
- `output_path`: 保存处理后动漫风格图像的目录路径。

脚本支持的图像格式由 `pic_form` 列表定义。
"""
import onnxruntime as ort # 导入 ONNX Runtime，用于执行 ONNX 模型
import time, os, cv2
import numpy as np
from glob import glob # 用于查找匹配特定模式的文件路径

# 定义支持的图片文件扩展名列表
pic_form = ['.jpeg','.jpg','.png','.JPEG','.JPG','.PNG']

def check_folder(path):
    """
    检查指定路径的文件夹是否存在，如果不存在则创建该文件夹。
    (此函数与 `tools.utils.py` 中的 `check_folder` 功能基本相同。)

    参数:
    - path (str): 需要检查或创建的文件夹路径。

    返回:
    - str: 传入的文件夹路径。
    """
    if not os.path.exists(path): # 如果路径不存在
        os.makedirs(path) # 则创建该路径对应的所有目录
        print(f"[*] 目录 '{path}' 已创建。")
    return path

def process_image(img, x32=True):
    """
    对输入的图像帧进行预处理，以适配生成器模型的输入要求。
    (此函数与 `video2anime.py` 中的 `process_image` 功能相似。)

    处理步骤包括:
    1. (可选) 将图像尺寸调整为32的倍数，这有助于某些网络架构处理。
    2. 将图像从 BGR 色彩空间 (OpenCV默认) 转换为 RGB 色彩空间。
    3. 将像素值从 [0, 255] 归一化到 [-1.0, 1.0] 范围。

    参数:
    - img (numpy.ndarray): 输入的图像 (通常为 BGR 格式，来自 cv2.imread)。
    - x32 (bool): 是否将图像尺寸调整为32的倍数。默认为 True。

    返回:
    - numpy.ndarray: 预处理后的图像，可直接作为 ONNX 模型输入。
    """
    h, w = img.shape[:2] # 获取图像的原始高度和宽度
    if x32: # 如果需要调整为32的倍数
        def to_32s(x):
            # 如果尺寸小于256，则设为256；否则，调整为不大于原尺寸的32的最大倍数。
            return 256 if x < 256 else x - x % 32
        img = cv2.resize(img, (to_32s(w), to_32s(h))) # 调整图像尺寸

    # BGR to RGB, then normalize to [-1, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return img

def load_test_data(image_path, size=[256,256]): # size 参数在此函数中未被直接使用
    """
    从指定路径加载一张测试图像，进行预处理，并返回处理后的图像及其原始形状。

    处理步骤:
    1. 使用 OpenCV 读取图像 (`img0`)。
    2. 调用 `process_image` 对图像进行预处理 (尺寸调整，BGR->RGB，归一化)。
       注意：`process_image` 中的尺寸调整逻辑优先于传入的 `size` 参数。
    3. 增加一个批处理维度。

    参数:
    - image_path (str): 图像文件的完整路径。
    - size (list or tuple, 可选): 期望的图像尺寸 [高度, 宽度]。
                                 注意：此参数在当前实现中未直接传递给 `process_image` 的 `size` 参数，
                                 `process_image` 使用其内部的 `x32` 逻辑和默认 `size` (如果适用)。
                                 此参数可能是一个遗留参数或意图用于其他目的。

    返回:
    - tuple: `(img, img0_shape)`
        - `img` (numpy.ndarray): 预处理后的图像，形状为 (1, height, width, channels)。
        - `img0_shape` (tuple): 原始图像 `img0` 的形状 (height, width, channels)。
    """
    img0 = cv2.imread(image_path).astype(np.float32) # 读取原始图像
    if img0 is None:
        raise IOError(f"无法读取图像文件: {image_path}")

    # 对图像进行预处理。注意：这里的 size 参数 (来自 load_test_data 的参数)
    # 并没有直接传递给 process_image 的 size 参数。process_image 使用其默认的 x32=True 逻辑。
    img = process_image(img0, x32=True) # x32=True 是 process_image 的默认行为

    img = np.expand_dims(img, axis=0) # 增加批处理维度 (H,W,C) -> (1,H,W,C)
    return img, img0.shape # 返回处理后的图像和原始图像的形状

def save_images(images, image_path, original_shape_wh):
    """
    对 ONNX 模型输出的图像进行后处理并保存。

    处理步骤:
    1. 反归一化像素值从 [-1.0, 1.0] 到 [0, 255] 范围。
    2. 使用 `np.clip` 确保像素值在 [0, 255] 之间。
    3. 将数据类型转换为 `uint8`。
    4. 将图像尺寸调整回原始输入图像的尺寸。
    5. 将图像从 RGB 色彩空间转换回 BGR (OpenCV `imwrite` 期望的格式)。
    6. 使用 `cv2.imwrite` 保存图像。

    参数:
    - images (numpy.ndarray): ONNX Runtime 输出的图像数据，通常是需要 `squeeze()` 去除批处理维度的。
    - image_path (str): 图像保存的完整路径。
    - original_shape_wh (tuple): 原始图像的 (宽度, 高度)，用于调整输出图像尺寸。
    """
    images = (np.squeeze(images) + 1.) / 2 * 255 # 反归一化并移除批处理维度
    images = np.clip(images, 0, 255).astype(np.uint8) # 裁剪并转换为 uint8
    # 将图像调整回原始尺寸，original_shape_wh 是 (原始宽度, 原始高度)
    images = cv2.resize(images, original_shape_wh)
    # 将图像从 RGB 转换回 BGR 以便 OpenCV 保存
    cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))

def Convert(input_imgs_path, output_path, onnx_model_path="model.onnx", img_size=[256,256]):
    """
    使用指定的 ONNX 模型对输入目录中的所有图像进行风格转换。

    参数:
    - input_imgs_path (str): 包含待转换输入图像的目录路径。
    - output_path (str): 保存处理后动漫风格图像的输出目录路径。
    - onnx_model_path (str, 可选): ONNX 模型文件的路径。默认: "model.onnx"。
    - img_size (list or tuple, 可选): 期望的图像尺寸 [高度, 宽度]。
                                     注意：此参数传递给 `load_test_data`，但如 `load_test_data`
                                     文档所述，其在预处理中的直接作用可能有限。

    主要逻辑:
    1.  创建输出目录。
    2.  查找输入目录中所有符合 `pic_form` 后缀的图像文件。
    3.  使用 `onnxruntime.InferenceSession` 加载 ONNX 模型并创建推理会话。
    4.  获取模型的输入和输出节点名称。
    5.  遍历所有找到的测试图像：
        a.  调用 `load_test_data` 加载并预处理图像，同时获取原始图像形状。
        b.  使用 `session.run` 执行 ONNX 模型推理。
        c.  调用 `save_images` 对输出图像进行后处理并保存。
    6.  计算并打印平均处理时间。
    """
    # result_dir = opj(output_path, style_name) # 原代码中此行被注释，直接使用 output_path
    result_dir = output_path # 设置结果保存目录
    check_folder(result_dir) # 确保输出目录存在

    # 获取输入目录中所有图像文件的路径
    test_files = glob('{}/*.*'.format(input_imgs_path))
    # 筛选出具有有效扩展名的图像文件
    test_files = [ x for x in test_files if os.path.splitext(x)[-1].lower() in pic_form]
    if not test_files:
        print(f"在目录 '{input_imgs_path}' 中未找到符合格式 {pic_form} 的图像文件。")
        return

    print(f"找到 {len(test_files)} 张图像进行处理。")
    # 加载 ONNX 模型并创建推理会话
    # providers 参数可以指定执行提供者，例如 ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # 如果为 None，ONNX Runtime 会选择最合适的可用提供者。
    try:
        session = ort.InferenceSession(onnx_model_path, None)
    except Exception as e:
        print(f"[错误] 加载 ONNX 模型 '{onnx_model_path}' 失败: {e}")
        return

    # 获取 ONNX 模型的输入节点名称
    # 假设模型只有一个输入节点
    input_node_name = session.get_inputs()[0].name
    # 获取 ONNX 模型的输出节点名称
    # 假设模型只有一个输出节点
    output_node_name = session.get_outputs()[0].name
    print(f"ONNX 模型输入节点: {input_node_name}, 输出节点: {output_node_name}")

    begin = time.time() # 记录开始时间
    # 遍历并处理每张测试图像
    for i, sample_file_path  in enumerate(test_files) :
        loop_start_time = time.time() # 记录单张图片处理开始时间

        # 加载并预处理测试数据
        # img_size 参数在这里的作用需要参照 load_test_data 的具体实现
        sample_image, original_shape = load_test_data(sample_file_path, img_size)

        # 构建输出图像的保存路径
        output_image_path = os.path.join(result_dir,'{0}'.format(os.path.basename(sample_file_path)))

        # 执行 ONNX 模型推理
        # session.run() 的第一个参数是期望的输出节点名称列表 (如果为 None，则返回所有输出节点)
        # 第二个参数是一个字典，键是输入节点名称，值是输入数据。
        onnx_output = session.run(None, {input_node_name : sample_image})

        # 保存处理后的图像
        # onnx_output 是一个列表，包含各个输出节点的输出结果。对于单输出模型，取其第一个元素。
        # original_shape[1] 是原始宽度, original_shape[0] 是原始高度
        save_images(onnx_output[0], output_image_path, (original_shape[1], original_shape[0]))

        print(f'处理图像: {i+1}/{len(test_files)}, 尺寸: {(original_shape[1], original_shape[0])}, 文件: {os.path.basename(sample_file_path)}, 耗时: {time.time() - loop_start_time:.3f} 秒')

    end = time.time() # 记录结束时间
    if test_files: # 避免除以零
        print(f'平均每张图像处理耗时: {(end-begin)/len(test_files):.4f} 秒')
    print(f"所有图像处理完成，结果已保存到 '{result_dir}'。")

if __name__ == '__main__':
    """
    脚本主执行块。

    当直接运行此脚本时，会执行以下操作：
    1.  定义 ONNX 模型文件路径 (`onnx_file`)。
        **用户需要根据实际情况修改此路径。**
    2.  定义包含输入图像的目录路径 (`input_imgs_path`)。
        **用户需要根据实际情况修改此路径。**
    3.  定义保存处理后图像的输出目录路径 (`output_path`)。
        **用户可以根据需要修改此名称。**
    4.  调用 `Convert` 函数执行批量图像风格转换。
    """
    # --- 用户需要配置的参数 ---
    # ONNX 模型文件的路径
    onnx_file = 'Shinkai_53.onnx' # 示例 ONNX 文件名，请修改为您的实际文件名
    # 输入图像目录路径
    input_imgs_path = '../../dataset/test/HR_photo' # 示例输入路径，请修改
                                                    # 注意路径是相对于当前脚本 (pb_and_onnx_model目录) 的
    # 输出图像目录路径
    output_path = 'Shinkai_53_output_onnx_test' # 示例输出目录名
    # --- 参数配置结束 ---

    print(f"ONNX 模型文件: {onnx_file}")
    print(f"输入图像目录: {input_imgs_path}")
    print(f"输出图像目录: {output_path}")

    # 调用转换函数
    Convert(input_imgs_path, output_path, onnx_file)
    print("\nONNX 模型测试执行完毕。")
