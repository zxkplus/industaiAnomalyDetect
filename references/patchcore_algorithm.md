# PatchCore 算法参考

## 目录

- [算法原理](#算法原理)
- [核心组件](#核心组件)
- [参数配置](#参数配置)
- [使用指南](#使用指南)
- [坐标映射说明](#坐标映射说明)
- [示例代码](#示例代码)

---

## 算法原理

### PatchCore 概述

PatchCore 是一种基于记忆库的异常检测方法，核心思想是:
1. 使用预训练深度网络提取正常样本的特征
2. 构建正常特征记忆库
3. 检测时计算查询特征与记忆库的相似度
4. 低相似度区域判定为异常

### 关键技术点

1. **中层特征聚合**:提取网络中间层特征，兼顾语义和空间信息
2. **随机Patch采样**:降低记忆库规模，提高检测效率
3. **余弦距离度量**:对特征进行L2归一化后使用余弦相似度

---

## 核心组件

### FeatureExtractor

特征提取器，支持多种预训练模型:

| 模型 | 特征维度(layer2+layer3) | 推荐场景 |
|------|------------------------|----------|
| resnet18 | 384 (128+256) | 轻量级，资源受限 |
| resnet34 | 384 | 平衡性能 |
| resnet50 | 1536 (512+1024) | 高精度 |
| wide_resnet50_2 | 1536 | 宽网络，更宽感受野 |

### NormalModel

正常特征模型，支持三种模式:

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| NormalModel | 基础模式 | 标准异常检测 |
| HierarchicalNormalModel | 分层索引 | 需要局部异常定位 |
| EnsembleNormalModel | 多模型集成 | 高精度需求 |

### AnomalyDetector

检测器，支持多种检测模式:

| 方法 | 说明 |
|------|------|
| detect | 单尺度基础检测 |
| detect_multi_scale | 多尺度融合检测 |
| detect_with_abnormal_library | 异常库增强检测 |
| analyze_image | 详细统计分析 |

---

## 参数配置

### FeatureExtractor 参数

```python
FeatureExtractor(
    model_name='resnet18',      # 模型名称
    layers=['layer2', 'layer3'], # 特征层，可选 ['layer1', 'layer2', 'layer3', 'layer4']
    pretrained=True,            # 是否使用预训练权重
    device=None                  # 设备，None自动选择
)
```

### NormalModel 参数

```python
NormalModel(
    extractor,                    # 特征提取器实例
    sample_size=1000,            # 每个样本采样点数，None=全部
    random_seed=42                # 随机种子，保证可复现
)
```

### AnomalyDetector 参数

```python
AnomalyDetector(
    extractor,                   # 特征提取器
    normal_model,                # 正常模型
    anomaly_library=None,        # 异常库(可选)
    k_neighbors=5,               # 计算分数的近邻数
    score_threshold=0.5,          # 异常判定阈值
    nms_threshold=0.5,           # NMS重叠阈值
    min_box_area=100              # 最小框面积
)
```

### 参数调优建议

| 参数 | 调优建议 |
|------|----------|
| k_neighbors | 增大可提高鲁棒性，减小可提高敏感性，默认5 |
| score_threshold | 根据precision/recall需求调整，高精度需求可提高至0.7 |
| nms_threshold | 控制重叠框合并，高密集异常场景可降低至0.3 |
| min_box_area | 根据实际缺陷大小设置，过小会引入噪声 |

---

## 使用指南

### 流程1: 全新部署

```
1. 收集正常样本(至少50-100张)
2. 收集异常样本并标注坐标
3. 训练阶段:构建normal_model和anomaly_library
4. 部署阶段:加载模型进行检测
```

### 流程2: 模型更新

```
1. 增量添加新正常样本
2. 调用add_sample后重新build
3. 保存更新后的模型
```

### 流程3: 异常类别扩展

```
1. 准备新类别样本及标注
2. 调用register注册新类别
3. 调用build更新异常库
```

---

## 坐标映射说明

### 坐标系定义

- **输入图像**: shape (H, W, 3)，坐标系原点在左上角
- **特征图**: shape (Fh, Fw, D)，每个位置对应原图一个区域
- **坐标格式**:
  - 输入/内部: [y1, x1, y2, x2] (左上角y,x 右下角y,x)
  - 输出框: [x1, y1, x2, y2] (标准bbox格式)

### 映射关系

```
原图尺寸: (H, W)
特征图尺寸: (Fh, Fw)

特征位置 (fh, fw) 对应原图区域:
- y1 = fh * (H / Fh)
- x1 = fw * (W / Fw)
- y2 = (fh + 1) * (H / Fh)
- x2 = (fw + 1) * (W / Fw)
```

### 特征图与原图对应示例

```python
# 原图 224x224，特征图 28x28
# 特征位置 (14, 14) 对应原图区域
y1 = 14 * (224 / 28) = 112
x1 = 14 * (224 / 28) = 112
y2 = 15 * (224 / 28) = 120
x2 = 15 * (224 / 28) = 120
# 即原图区域 [112:120, 112:120]
```

---

## 示例代码

### 完整检测流程

```python
from scripts import (
    FeatureExtractor, NormalModel, 
    AnomalyLibrary, AnomalyDetector
)
from scriptsutils import load_image

# 1. 初始化
extractor = FeatureExtractor('resnet18', ['layer2', 'layer3'])
normal_model = NormalModel(extractor, sample_size=1000)

# 2. 构建正常模型
for img_path in glob.glob('data/normal/*.png'):
    normal_model.add_sample(load_image(img_path))
normal_model.build()

# 3. 可选:构建异常库
anomaly_lib = AnomalyLibrary()
# 注册各类异常...
anomaly_lib.build()

# 4. 检测
detector = AnomalyDetector(extractor, normal_model, anomaly_lib)
result = detector.detect('test_image.png')

print(f"异常: {result['anomaly']}")
print(f"类别: {result['class']}")
print(f"区域: {result['boxes']}")
```

### 调整检测灵敏度

```python
# 高灵敏度(检测更多异常，可能增加误报)
detector_high = AnomalyDetector(
    extractor, normal_model,
    score_threshold=0.3,
    nms_threshold=0.3,
    min_box_area=50
)

# 高精度(减少误报，可能漏检)
detector_strict = AnomalyDetector(
    extractor, normal_model,
    score_threshold=0.7,
    nms_threshold=0.7,
    min_box_area=200
)
```

### 特征可视化

```python
import matplotlib.pyplot as plt

# 获取检测结果
result = detector.detect(image)

# 显示分数热力图
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(result['score_map'], cmap='hot')
plt.title('Anomaly Score Map')
plt.colorbar()

plt.subplot(1, 3, 3)
# 在原图上绘制检测框
plt.imshow(image)
for box, score in zip(result['boxes'], result['scores']):
    x1, y1, x2, y2 = box
    plt.gca().add_patch(
        plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                      fill=False, color='red', linewidth=2)
    )
plt.title(f'Detected Anomalies: {len(result["boxes"])}')
plt.show()
```
