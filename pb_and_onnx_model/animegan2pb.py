# -*- coding: UTF-8 -*-
"""
文件描述:
此脚本用于将 TensorFlow 训练的 AnimeGANv2 生成器模型的检查点 (checkpoint) 转换为
一个“冻结”的 Protocol Buffer (.pb) 文件。冻结图 (Frozen Graph) 是指将模型架构
（计算图定义）和已训练的权重（变量值）合并到一个单独的文件中，并将变量转换为常量。
这种格式非常适合模型部署，因为它不依赖于原始的 Python 代码，并且更易于在不同平台和框架上使用。

脚本还演示了如何随后使用 `tf2onnx` 工具（通过命令行调用）将生成的 .pb 文件转换为
ONNX (Open Neural Network Exchange) 格式。ONNX 是一种开放的模型表示格式，
旨在促进不同深度学习框架之间的互操作性。

作者: Xin Chen
时间: 2021/8/31 (如此信息与文件内容相关)

使用方式:
此脚本通过命令行直接运行：
`python pb_and_onnx_model/animegan2pb.py`

在运行之前，用户需要根据实际情况修改脚本末尾 `if __name__ == '__main__':` 代码块中的
以下变量：
- `model_folder`: 指向包含 TensorFlow 检查点文件的目录路径。
                  这通常是一个仅包含生成器权重的检查点目录，例如由 `tools/get_generator_ckpt.py` 生成的目录。
- `pb_save_path`: 指定输出的 .pb 文件名及保存路径。

脚本执行后，会首先生成 .pb 文件，然后尝试调用系统命令将其转换为 .onnx 文件。
确保已安装 `tf2onnx` 包 (`pip install tf2onnx`) 以成功执行 ONNX 转换步骤。
"""
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util # 用于图操作，特别是变量冻结

def freeze_graph(model_folder, output_graph_path):
    """
    加载 TensorFlow 检查点，将图中的变量转换为常量（冻结图），并将其保存为 .pb 文件。

    参数:
    - model_folder (str): 包含 TensorFlow 检查点文件 (.meta, .data, .index) 的目录路径。
                          期望这是一个仅包含生成器权重的检查点。
    - output_graph_path (str): 输出的冻结图 (.pb) 文件的保存路径和名称。

    关键逻辑:
    1.  通过 `tf.train.get_checkpoint_state` 获取指定目录中最新的检查点信息。
    2.  指定生成器网络中的输出节点名称 (`output_node_names`)。这对于冻结过程至关重要，
        因为它告诉 TensorFlow哪些节点是模型的最终输出，从而可以裁剪掉不需要的部分。
    3.  使用 `tf.train.import_meta_graph` 从 .meta 文件加载计算图结构。
    4.  创建一个 TensorFlow 会话 (`tf.Session`)。
    5.  在会话中，使用 `tf.train.Saver().restore()` 从检查点恢复权重到加载的图中。
    6.  使用 `graph_util.convert_variables_to_constants` 将图中的变量转换为常量，实现冻结。
    7.  将冻结后的图定义序列化并写入指定的 .pb 文件。
    8.  (可选) 打印最终图中操作的数量和所有操作的名称，用于调试或验证。
    """
    try:
        # 获取检查点状态，它包含了最新检查点的信息
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        if checkpoint is None:
            print(f"[错误] 在目录 '{model_folder}' 中未找到检查点状态文件。请确保路径正确且包含 'checkpoint' 文件。")
            return
        print(f"找到的检查点信息: {checkpoint}")
        # 获取最新检查点的完整路径 (例如, 'path/to/model.ckpt-100')
        input_checkpoint_path = checkpoint.model_checkpoint_path
        if input_checkpoint_path is None:
            print(f"[错误] 检查点状态文件中未包含有效的模型检查点路径。")
            return
        print(f"准备加载的检查点文件: {input_checkpoint_path}")

        # 指定输出节点的名称。对于 AnimeGANv2 生成器，这通常是最后一个 Tanh 激活层。
        # 确保此名称与 TensorFlow 计算图中定义的节点名称完全一致。
        # 如果有多个输出节点，可以用逗号分隔它们的名称。
        output_node_names = "generator/G_MODEL/out_layer/Tanh"

        # 创建一个 Saver 对象，用于后续从检查点恢复权重。
        # import_meta_graph 会创建图结构，但变量的值是未初始化的。
        saver = tf.train.import_meta_graph(input_checkpoint_path + '.meta', clear_devices=True)

        graph = tf.get_default_graph() # 获取当前默认的计算图
        input_graph_def = graph.as_graph_def() # 获取图的定义 (GraphDef protocol buffer)

        print(f"开始冻结图模型，输入检查点: {input_checkpoint_path}")
        with tf.Session() as sess:
            saver.restore(sess, input_checkpoint_path) # 从检查点恢复变量的值到当前会话

            # 核心步骤：将图中的变量转换为常量。
            # sess: 当前会话，包含了已恢复的变量值。
            # input_graph_def: 原始的图定义。
            # output_node_names.split(","): 输出节点名称列表。
            #   冻结过程会保留从输入到这些输出节点所需的所有操作和常量。
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=output_node_names.split(","))

            # 将冻结后的图定义写入 .pb 文件
            with tf.gfile.GFile(output_graph_path, "wb") as f: # "wb" 表示以二进制写入模式打开文件
                f.write(output_graph_def.SerializeToString()) # 序列化图定义并写入文件
            print(f"成功冻结图模型! {len(output_graph_def.node)} 个操作已写入到: {output_graph_path}")

            # (可选) 打印图中所有操作的名称及其输出值，用于调试和确认图结构
            print("\n图中所有操作 (Ops in the final graph):")
            for op_idx, op in enumerate(graph.get_operations()):
                # 为了避免过多输出，可以限制打印数量或特定操作
                if op_idx < 20 or "generator" in op.name: # 示例：只打印前20个或包含'generator'的
                    print(f"- {op.name}, 输出: {op.values()}")
            if len(graph.get_operations()) > 20:
                print(f"... 以及其他 {len(graph.get_operations()) - 20} 个操作。")

    except Exception as e:
        print(f"[错误] 冻结图模型时发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    """
    脚本主执行块。

    当直接运行此脚本时，会执行以下操作：
    1.  定义包含预训练生成器权重的 TensorFlow 检查点目录 (`model_folder`)。
        **用户需要根据实际情况修改此路径。**
    2.  定义输出的冻结图 (.pb) 文件的名称和路径 (`pb_save_path`)。
        **用户可以根据需要修改此名称。**
    3.  调用 `freeze_graph` 函数，将指定的检查点转换为冻结的 .pb 文件。
    4.  构造一个系统命令 (`cmd`)，用于调用 `tf2onnx.convert` 工具，
        将上一步生成的 .pb 文件转换为 ONNX (.onnx) 格式。
        **此命令中的输入/输出节点名称 (`--inputs`, `--outputs`) 和文件名需要与
        实际情况匹配。**
    5.  使用 `os.system(cmd)` 执行该 ONNX 转换命令。
    """
    # --- 用户需要配置的参数 ---
    # 输入：包含 TensorFlow 生成器检查点文件的目录路径。
    # 这个目录通常是运行 tools/get_generator_ckpt.py 后生成的，例如 'checkpoint/generator_Hayao_weight'。
    # 请确保此路径指向正确的检查点目录。
    model_folder = "../checkpoint/generator_Shinkai_weight" # 示例路径，请修改为您的实际路径
    # 输出：冻结图 (.pb) 文件的保存名称。
    pb_save_path = "Shinkai_53.pb" # 示例输出文件名
    # --- 参数配置结束 ---

    print(f"开始处理模型转换，输入检查点目录: {model_folder}")
    # 步骤 1: 冻结 TensorFlow 图并保存为 .pb 文件
    freeze_graph(model_folder, pb_save_path)

    # 步骤 2: 将 .pb 文件转换为 .onnx 文件
    # 构造 tf2onnx 转换命令
    # --input: 指定输入的 .pb 文件路径。
    # --inputs: 指定 .pb 文件中输入节点的名称和形状 (如果需要)。
    #           格式为 "node_name:0"。这里的 "generator_input:0" 必须与
    #           tools/get_generator_ckpt.py 或原始训练脚本中定义的输入 placeholder 名称一致。
    # --outputs: 指定 .pb 文件中输出节点的名称。
    #            这里的 "generator/G_MODEL/out_layer/Tanh:0" 是 AnimeGANv2 生成器的典型输出节点。
    # --output: 指定输出的 .onnx 文件名。
    onnx_output_path = pb_save_path.replace(".pb", ".onnx") # 基于 .pb 文件名生成 .onnx 文件名

    # 确保输入节点名称与模型定义一致。
    # 在 get_generator_ckpt.py 中，输入 placeholder 被命名为 'generator_input_dummy'。
    # 如果冻结的是那个图，这里应该是 'generator_input_dummy:0'。
    # 如果冻结的是原始训练图，且输入 placeholder 名为 'test_input' 或 'real_A' (来自AnimeGANv2.py)，则需相应修改。
    # 假设冻结的图是基于 get_generator_ckpt.py 中定义的 'generator_input_dummy'。
    # 但 AnimeGANv2.py 中 test_real 的 name 是 'test_input'，训练时 real 的 name 是 'real_A'
    # 在 test.py 中，输入 placeholder name='test_input_real_image'
    # 在 video2anime.py 中，输入 placeholder name='test_video_frame_input'
    # ** 重要: 此处的 '--inputs' 必须与冻结图中的实际输入节点名匹配。**
    # ** 常见的 AnimeGANv2 生成器输入节点名可能是 'generator_input:0' (如果使用了 get_generator_ckpt.py 且其 placeholder 是 'generator_input')
    # ** 或 'test_input:0' / 'real_A:0' (如果直接冻结训练图)。**
    # ** 假设这里我们使用了 'generator_input:0' 作为标准。**
    # ** 如果转换失败，请检查 .pb 文件的实际输入输出节点名 (可以使用 Netron 等工具查看)。**
    input_node_tf_name = "generator_input:0" # 示例名称，需要核实
    output_node_tf_name = "generator/G_MODEL/out_layer/Tanh:0" # 这是模型的输出节点

    cmd = (f"python -m tf2onnx.convert "
           f"--input {pb_save_path} "
           f"--inputs {input_node_tf_name} "
           f"--outputs {output_node_tf_name} "
           f"--output {onnx_output_path}")

    print(f"\n准备执行 ONNX 转换命令: {cmd}")
    try:
        # 执行系统命令
        conversion_status = os.system(cmd)
        if conversion_status == 0:
            print(f"成功将 '{pb_save_path}' 转换为 ONNX 格式: '{onnx_output_path}'")
        else:
            print(f"[错误] ONNX 转换失败，命令返回状态码: {conversion_status}。请检查 `tf2onnx` 是否已正确安装，以及输入/输出节点名称是否正确。")
    except Exception as e:
        print(f"[错误] 执行 ONNX 转换命令时发生异常: {e}")

    print(f"\n处理完成。")
