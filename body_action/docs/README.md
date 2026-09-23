# 新视频网球动作识别与部署流程说明

本文档用于说明如何在已交付的 **MS-TCN++ 网球动作识别项目** 中，对新录制的视频进行动作识别，并将预测结果重新叠加到视频画面中。

该项目的核心流程不是直接把视频输入模型，而是先从视频中提取人体姿态关键点特征，再将特征输入 MS-TCN++ / ONNX / RKNN 模型进行时序动作识别。

整体流程如下：

```text
新视频
-> MediaPipe Pose 提取人体关键点
-> 关键点归一化与特征整理
-> 生成 .npy 特征文件，形状为 (99, T)
-> 运行 ONNX 或 RKNN 推理
-> 生成预测 txt 文件
-> 运行 overlay 脚本
-> 将预测动作标签画回原视频
```

---

## 1. 模型功能说明

本模型用于根据人体姿态关键点特征，对网球视频进行**逐帧动作识别**。

也就是说，模型会判断视频中每一个时间步属于哪一种动作类别，例如：

```text
serve
forehand
backhand
background
```

需要注意的是：

```text
模型输入不是原始视频，也不是图片帧。
模型输入是已经提取好的 MediaPipe Pose 人体关键点特征。
```

---

## 2. 模型输入与输出格式

### 2.1 输入特征来源

本项目使用 MediaPipe Pose 提取人体姿态关键点。

MediaPipe Pose 通常会输出 33 个人体关键点，每个关键点包含 3 个坐标值：

```text
x, y, z
```

因此每一帧的特征维度为：

```text
33 个关键点 × 3 个坐标值 = 99 维特征
```

所以新视频经过关键点提取后，需要转换成如下 `.npy` 特征格式：

```text
(99, T)
```

其中：

```text
99 表示每一帧的特征维度
T 表示视频的时间长度，可以理解为帧数或时间步数量
```

例如：

```text
(99, 800)
```

表示该视频共有 800 个时间步，每个时间步有 99 维人体关键点特征。

---

### 2.2 ONNX / RKNN 模型输入格式

ONNX 或 RKNN 模型推理时，输入形状通常为：

```text
(1, 99, T)
```

其中：

```text
1  表示 batch size
99 表示每一帧的关键点特征维度
T  表示视频时间长度
```

---

### 2.3 模型输出格式

模型输出形状为：

```text
(1, 4, T)
```

其中：

```text
1 表示 batch size
4 表示动作类别数量
T 表示时间长度
```

这里的 `4` 对应 4 个动作类别。

---

## 3. 类别映射关系

模型输出的是类别编号，对应关系如下：

```text
0 serve       发球
1 forehand    正手击球
2 backhand    反手击球
3 background  背景 / 非目标动作
```

类别映射文件通常存放在：

```text
data/mapping.txt
```

模型预测时，会先得到每一帧的类别编号，然后再根据 `mapping.txt` 转换成具体的动作标签。

---

## 4. 项目交付包结构说明

项目交付包大致结构如下：

```text
tennis_rk3588_delivery/
  models/
    mstcn_tennis_dynamic.onnx

  data/
    mapping.txt

  samples/
    sample_video.mp4
    test_video3.npy
    test_video3_prediction.txt

  scripts/
    export_dynamic_onnx.py
    tennis_tiqu_transform.py
    overlay_prediction_on_video.py
    onnx_infer_demo.py
    rknn_convert_demo.py
    rknn_infer_demo.py

  docs/
    README_deploy.md

  requirements_pc.txt
```

各目录和文件作用如下：

| 路径 / 文件 | 作用 |
|---|---|
| `models/` | 存放模型文件，例如 ONNX 模型和 RKNN 模型 |
| `mstcn_tennis_dynamic.onnx` | 已导出的动态 ONNX 模型 |
| `data/mapping.txt` | 类别编号与动作名称的映射文件 |
| `samples/` | 存放示例视频、示例特征和预测结果 |
| `scripts/` | 存放特征提取、ONNX 推理、RKNN 转换、RKNN 推理和视频叠加脚本 |
| `requirements_pc.txt` | PC 端运行 ONNX 测试所需的 Python 依赖 |
| `README_deploy.md` | 部署说明文档 |

---

## 5. PC 端 ONNX 测试流程

在 PC 上测试 ONNX 推理前，需要先安装依赖。

在交付包根目录下运行：

```bash
pip install -r requirements_pc.txt
```

然后进入脚本目录：

```bash
cd tennis_rk3588_delivery/scripts
```

运行 ONNX 推理示例：

```bash
python onnx_infer_demo.py
```

默认示例输入为：

```text
../samples/test_video3.npy
```

该文件是已经提取好的关键点特征文件，形状应为：

```text
(99, T)
```

运行成功后，通常会生成：

```text
../samples/onnx_prediction.txt
```

该 txt 文件中保存每一帧的预测动作标签。

---

## 6. 新视频动作识别完整流程

假设你的新视频路径为：

```text
C:\Users\15875\Desktop\MS-TCN++\new_test.mp4
```

最终希望得到带动作标签的视频：

```text
C:\Users\15875\Desktop\MS-TCN++\new_test_overlay.mp4
```

完整过程分为三步：

```text
第 1 步：提取关键点特征，生成 .npy
第 2 步：使用 ONNX 模型推理，生成预测 txt
第 3 步：将预测结果叠加回原视频
```

---

## 7. 第一步：提取关键点特征

### 7.1 作用说明

第一步的作用是从新视频中提取人体关键点，并整理成 MS-TCN++ 模型能够识别的 `.npy` 特征文件。

该步骤输入为：

```text
new_test.mp4
```

输出为：

```text
new_test.npy
```

输出文件形状应类似：

```text
(99, T)
```

---

### 7.2 修改输入和输出路径

打开脚本：

```text
tennis_tiqu_transform.py
```

路径通常为：

```text
C:/Users/15875/Desktop/MS-TCN++/ms-tcn++/tennis_rk3588_delivery/scripts/tennis_tiqu_transform.py
```

找到代码中的视频输入路径和特征输出路径，将其修改为：

```python
path = r"C:\Users\15875\Desktop\MS-TCN++\new_test.mp4"
output_path = r"C:\Users\15875\Desktop\MS-TCN++\ms-tcn++\tennis_rk3588_delivery\samples\new_test.npy"
```

其中：

```text
path
```

表示新录制的视频路径。

```text
output_path
```

表示生成的 `.npy` 特征文件保存路径。

---

### 7.3 运行关键点提取脚本

在命令行中进入脚本目录：

```bash
cd C:\Users\15875\Desktop\MS-TCN++\ms-tcn++\tennis_rk3588_delivery\scripts
```

运行：

```bash
python tennis_tiqu_transform.py
```

成功后会生成：

```text
samples/new_test.npy
```

---

### 7.4 输出结果说明

如果生成的 `.npy` 文件形状为：

```text
(99, 800)
```

则表示：

```text
该视频被转换为了 800 个时间步
每个时间步包含 99 维人体关键点特征
```

这说明新视频已经成功转换为模型可输入的时序特征。

---

## 8. 第二步：使用 ONNX 模型进行推理

### 8.1 作用说明

第二步的作用是将第一步生成的 `.npy` 特征文件输入 ONNX 模型，得到每一帧的动作预测结果。

输入文件：

```text
../samples/new_test.npy
```

输出文件：

```text
../samples/new_test_prediction.txt
```

---

### 8.2 运行 ONNX 推理命令

在脚本目录下运行：

```bash
python onnx_infer_demo.py --feature ../samples/new_test.npy --output ../samples/new_test_prediction.txt
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--feature` | 输入的 `.npy` 特征文件路径 |
| `--output` | 输出的预测结果 txt 文件路径 |

---

### 8.3 预测结果文件格式

运行成功后，会生成：

```text
samples/new_test_prediction.txt
```

预测文件内容一般类似：

```text
### Frame level recognition: ###
serve serve serve forehand forehand background ...
```

含义是：

```text
每一个单词对应一个时间步或一帧的动作预测结果
```

例如：

```text
serve
```

表示该帧被预测为发球动作。

```text
forehand
```

表示该帧被预测为正手击球。

```text
backhand
```

表示该帧被预测为反手击球。

```text
background
```

表示该帧为背景或非目标动作。

---

## 9. 第三步：将预测结果叠加回视频

### 9.1 作用说明

第三步的作用是将 ONNX 推理得到的动作标签重新画回原始视频中。

输入文件包括：

```text
原始视频：new_test.mp4
预测结果：new_test_prediction.txt
```

输出文件为：

```text
new_test_overlay.mp4
```

输出视频中会显示每一帧对应的预测动作标签，便于直观观察模型识别效果。

---

### 9.2 修改 overlay 脚本路径

打开脚本：

```text
overlay_prediction_on_video.py
```

路径通常为：

```text
C:/Users/15875/Desktop/MS-TCN++/ms-tcn++/tennis_rk3588_delivery/scripts/overlay_prediction_on_video.py
```

将视频路径、预测结果路径和输出视频路径修改为：

```python
video_path = r"C:\Users\15875\Desktop\MS-TCN++\new_test.mp4"
prediction_path = r"C:\Users\15875\Desktop\MS-TCN++\ms-tcn++\tennis_rk3588_delivery\samples\new_test_prediction.txt"
output_path = r"C:\Users\15875\Desktop\MS-TCN++\new_test_overlay.mp4"
```

其中：

| 变量 | 含义 |
|---|---|
| `video_path` | 原始视频路径 |
| `prediction_path` | ONNX 推理生成的预测 txt 文件路径 |
| `output_path` | 叠加动作标签后的视频保存路径 |

---

### 9.3 运行 overlay 脚本

在脚本目录下运行：

```bash
python overlay_prediction_on_video.py
```

成功后会生成：

```text
C:\Users\15875\Desktop\MS-TCN++\new_test_overlay.mp4
```

该视频就是最终的可视化识别结果视频。

---

## 10. 完整命令顺序

如果已经修改好 `tennis_tiqu_transform.py` 和 `overlay_prediction_on_video.py` 中的路径，那么完整命令顺序如下：

```bash
cd C:\Users\15875\Desktop\MS-TCN++\ms-tcn++\tennis_rk3588_delivery\scripts

python tennis_tiqu_transform.py

python onnx_infer_demo.py --feature ../samples/new_test.npy --output ../samples/new_test_prediction.txt

python overlay_prediction_on_video.py
```

执行完成后，最终会得到：

```text
new_test_overlay.mp4
```

---

## 11. 文件输入输出关系

| 步骤 | 输入文件 | 执行脚本 | 输出文件 | 作用 |
|---|---|---|---|---|
| 第 1 步 | `new_test.mp4` | `tennis_tiqu_transform.py` | `new_test.npy` | 从视频中提取 MediaPipe Pose 关键点特征 |
| 第 2 步 | `new_test.npy` | `onnx_infer_demo.py` | `new_test_prediction.txt` | 使用 ONNX 模型预测每帧动作 |
| 第 3 步 | `new_test.mp4` 和 `new_test_prediction.txt` | `overlay_prediction_on_video.py` | `new_test_overlay.mp4` | 将动作标签叠加回视频 |
| 可选步骤 | `mstcn_tennis_dynamic.onnx` | `rknn_convert_demo.py` | `mstcn_tennis_dynamic.rknn` | 将 ONNX 模型转换为 RKNN 模型 |
| 可选步骤 | `mstcn_tennis_dynamic.rknn` | `rknn_infer_demo.py` | `rknn_prediction.txt` | 在 RK3588 上进行模型推理 |

---

## 12. 后处理说明

模型输出的是 logits，也就是每一类的预测分数，形状为：

```text
(1, 4, T)
```

需要通过 `argmax` 取出分数最高的类别：

```python
pred = np.argmax(logits, axis=1).squeeze()
```

这一步的含义是：

```text
在 4 个动作类别中，选择每一帧得分最高的类别作为最终预测结果
```

得到类别编号后，再根据 `data/mapping.txt` 转换成具体动作标签：

```text
0 -> serve
1 -> forehand
2 -> backhand
3 -> background
```

最终即可得到逐帧动作识别结果。

---

## 13. ONNX 转 RKNN 部署说明

如果需要将模型部署到 RK3588 板端，需要先将 ONNX 模型转换为 RKNN 模型。

该步骤需要在安装了 Rockchip `rknn-toolkit2` 的 PC 或 Ubuntu 环境中运行。

进入脚本目录：

```bash
cd tennis_rk3588_delivery/scripts
```

运行转换脚本：

```bash
python rknn_convert_demo.py
```

转换成功后，预期会生成：

```text
../models/mstcn_tennis_dynamic.rknn
```

然后将整个 `tennis_rk3588_delivery` 文件夹复制到 RK3588 板子上。

在 RK3588 板端安装好 RKNN Lite2 后，运行：

```bash
cd tennis_rk3588_delivery/scripts
python rknn_infer_demo.py
```

运行成功后，预期会生成：

```text
../samples/rknn_prediction.txt
```

该文件即为 RK3588 板端推理得到的动作预测结果。

---

## 14. 关于动态 ONNX 模型的说明

当前模型为动态时间长度 ONNX 模型。

也就是说，输入特征的时间维度 `T` 可以变化，例如：

```text
(99, 300)
(99, 800)
(99, 1200)
```

只要特征维度仍然是 99，就可以进行推理。

但是需要注意：

```text
部分 RKNN 工具链或板端运行环境可能不完全支持动态时间维度。
```

如果 `rknn-toolkit2` 转换动态 ONNX 失败，或者 RK3588 板端推理不稳定，可以考虑导出固定长度模型。

对于不同长度的视频，可以使用以下方法处理：

```text
padding 补齐
```

或者：

```text
sliding window 滑动窗口切分
```

---

## 15. 注意事项

### 15.1 路径需要手动修改

目前：

```text
tennis_tiqu_transform.py
overlay_prediction_on_video.py
```

这两个脚本中的路径是写死的。

因此每次更换新视频时，需要手动修改：

```text
视频输入路径
.npy 特征输出路径
预测 txt 路径
最终 overlay 视频输出路径
```

否则脚本可能会继续处理旧视频，或者把结果输出到旧文件中。

---

### 15.2 `.npy` 特征维度必须正确

生成的 `.npy` 文件形状必须类似：

```text
(99, T)
```

其中：

```text
99 是模型训练时使用的特征维度，不能随意改变
T 是视频时间长度，可以根据视频长短变化
```

如果 `.npy` 的特征维度不是 99，ONNX 或 RKNN 推理时可能会报错。

---

### 15.3 模型不能直接识别 mp4

该模型不能直接输入：

```text
.mp4
.jpg
.png
```

它只能输入已经提取好的关键点特征：

```text
.npy
```

所以完整流程中，关键点提取是必不可少的一步。

---

### 15.4 视频与预测结果需要对应

进行 overlay 可视化时，原视频和预测 txt 必须对应同一个视频。

例如：

```text
new_test.mp4
new_test_prediction.txt
```

应当来自同一次处理流程。

如果拿 A 视频的预测结果叠加到 B 视频上，显示出来的动作标签就会错位或不准确。

---

## 16. 后续可优化方向

目前部分脚本路径是写死的，使用时需要手动打开 Python 文件修改路径。

后续可以将脚本改成命令行参数形式，例如：

```bash
python tennis_tiqu_transform.py --video new_test.mp4 --output ../samples/new_test.npy
```

或者：

```bash
python overlay_prediction_on_video.py --video new_test.mp4 --prediction ../samples/new_test_prediction.txt --output new_test_overlay.mp4
```

这样每次更换视频时，就不需要再手动修改代码，只需要在命令行中传入不同路径即可。

---

## 17. 总结

对新录制的视频进行网球动作识别时，完整流程可以概括为：

```text
视频
-> MediaPipe Pose 关键点提取
-> 关键点归一化
-> 生成 (99, T) 的 .npy 特征文件
-> 输入 ONNX 或 RKNN 模型推理
-> 得到 (1, 4, T) 的模型输出
-> argmax 得到每一帧类别编号
-> 根据 mapping.txt 转换成动作标签
-> 生成预测 txt 文件
-> 将预测标签叠加回原视频
```

最终生成的：

```text
new_test_overlay.mp4
```

可以直观展示模型对连续长视频中网球动作的识别效果，包括发球、正手击球、反手击球以及背景片段。
