**航拍无人机红外人车检测数据集 -红外无人机拍摄视角人车检测数据集7000多张图片，6个类别，方向小众，可以作为私有数据集（car,truck,person,bus,cyclist,bike）。yolo格式，配置好环境，直接用于训练。已经划分好训练集，验证集，测试集。**


| 字段 | 内容描述 |
| :--- | :--- |
| **数据集名称** | 航拍无人机红外人车检测数据集 |
| **数据视角/特点** | 红外无人机拍摄视角，方向小众，可作为私有数据集 |
| **数据量** | 7000+ 张图片 |
| **类别数量** | 6 类 |
| **类别列表** | car, truck, person, bus, cyclist, bike |
| **数据格式** | YOLO 格式 |
| **数据集划分** | 已划分训练集、验证集、测试集 |
| **使用建议** | 配置好环境后，可直接用于训练 |



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/43310a442e9f4e578ae5c25d56c2ef95.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/28cc00f251ee44a4b720222642fec85c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7f387ad42fb5470bb5907f341c7b4f40.png)
这是一个非常典型的红外/低光照场景下的目标检测任务。由于红外图像通常对比度低、缺乏纹理细节，且你的数据集包含 7000 多张图片，属于中等规模数据集，使用 **YOLOv8** 进行训练是一个非常好的选择。

YOLOv8 在特征提取方面比旧版本更强，且代码库（Ultralytics）非常成熟，适合快速上手。

以下是为你准备的完整训练指南和代码。

### 第一步：环境准备

你需要安装 PyTorch 和 Ultralytics 库。请在终端或 Anaconda Prompt 中运行以下命令：

```bash
# 安装 PyTorch (请根据你的显卡CUDA版本选择，这里以通用CPU/GPU自动适配为例)
pip install torch torchvision torchaudio

# 安装 Ultralytics (YOLOv8)
pip install ultralytics
```

### 第二步：数据集目录结构

假设你的数据集文件夹名为 `infrared_dataset`，请确保其结构如下所示。你需要创建一个 `.yaml` 配置文件来告诉模型数据的路径和类别信息。

**推荐的目录结构：**

```text
infrared_dataset/
├── images/
│   ├── train/  (存放训练集图片)
│   ├── val/    (存放验证集图片)
│   └── test/   (存放测试集图片)
├── labels/
│   ├── train/  (存放训练集txt标注)
│   ├── val/    (存放验证集txt标注)
│   └── test/   (存放测试集txt标注)
└── data.yaml   <-- 我们需要创建这个文件
```

**创建 `data.yaml` 文件：**

在 `infrared_dataset` 根目录下创建一个名为 `data.yaml` 的文件，内容如下（注意类别的顺序要和你的标签对应）：

```yaml
# data.yaml

# 数据集根路径 (根据你的实际路径修改，如果是同级目录保持 ./ 即可)
path: ./infrared_dataset

# 训练、验证、测试集的图片文件夹路径
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 6

# 类别名称 (严格按照你的描述顺序)
names: ['car', 'truck', 'person', 'bus', 'cyclist', 'bike']
```

### 第三步：训练代码

你可以使用 Python 脚本或者命令行来启动训练。为了方便你调整参数，我提供了一个完整的 Python 训练脚本。

新建一个 `train.py` 文件，将以下代码复制进去：

```python
from ultralytics import YOLO
import torch

def main():
    # 1. 检查设备 (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用设备: {device}")

    # 2. 加载模型
    # 这里我们使用 yolov8n (nano版本，速度快) 或 yolov8s (small版本，精度高)
    # 对于红外图像，建议先用 s 或 m 版本，因为特征较少，需要更强的骨干网络提取特征
    model = YOLO('yolov8s.pt')  # 如果没有预训练权重，会自动下载

    # 3. 开始训练
    # 参数解释:
    # data: 你的yaml配置文件路径
    # epochs: 训练轮数，7000张图建议 100-300 轮
    # imgsz: 图片尺寸，红外小目标建议适当调大，如 640 或 1280
    # batch: 批大小，根据显存调整，-1 为自动调整
    # device: 指定设备
    # project & name: 保存结果的路径
    results = model.train(
        data='./infrared_dataset/data.yaml',
        epochs=200,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        patience=50, # 早停机制，如果50轮没有提升则停止
        project='runs/train',
        name='infrared_exp',
        optimizer='AdamW', # 针对小目标或复杂背景，AdamW有时比SGD更稳定
        lr0=0.001, # 初始学习率
        augment=True # 开启默认增强
    )

    print("训练完成！")

if __name__ == '__main__':
    main()
```

### 第四步：针对红外数据的特别建议

由于你的数据是**红外无人机视角**，可能存在以下特点，建议在训练时注意：

1. **小目标问题：** 无人机视角下的人和车通常很小。
   - 建议在训练时将 `imgsz` 设置得大一些（如 `640` 或 `1280`），以便模型能看清小物体。
   - 如果检测效果不好，可以尝试使用 `yolov8m.pt` 或 `yolov8l.pt`，更大的模型对特征提取能力更强。

2. **数据增强：**
   - 红外图像通常对比度低。YOLOv8 默认包含 `Mosaic` 和 `MixUp` 增强，这对小目标检测非常有帮助，请保持开启（默认为开启）。

3. **验证与推理：**
   训练完成后，你可以使用以下代码对单张图片进行测试：

   ```python
   from ultralytics import YOLO

   # 加载训练好的最佳权重
   model = YOLO('runs/train/infrared_exp/weights/best.pt')

   # 预测
   results = model('path/to/your/test_image.jpg')

   # 显示结果
   results[0].show()
   ```

按照上述步骤配置环境并运行脚本，即可完成针对该红外数据集的模型训练。
