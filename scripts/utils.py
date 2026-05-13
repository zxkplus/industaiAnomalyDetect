"""
PatchCore Anomaly Detection - 工具模块
提供图像处理、坐标映射、距离计算等通用工具
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import pickle
import json


def load_image(image_path: str) -> np.ndarray:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        RGB格式的numpy数组，形状为(H, W, 3)
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        return np.array(img)
    except ImportError:
        raise ImportError("请安装Pillow: pip install pillow")


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        image: 输入图像数组 (H, W, 3)
        target_size: 目标尺寸 (height, width)
        
    Returns:
        调整后的图像数组
    """
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    return np.array(img)


def normalize_image(image: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    标准化图像
    
    Args:
        image: 输入图像数组 (H, W, 3)，值范围 [0, 255]
        mean: RGB均值
        std: RGB标准差
        
    Returns:
        标准化后的图像数组
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    return (image - mean) / std


def feature_map_to_coords(feature_h: int, feature_w: int, original_h: int, original_w: int) -> np.ndarray:
    """
    建立特征图到原图坐标的映射关系
    
    Args:
        feature_h: 特征图高度
        feature_w: 特征图宽度
        original_h: 原图高度
        original_w: 原图宽度
        
    Returns:
        坐标映射数组，形状为 (feature_h, feature_w, 4)，每元素包含 [y1, x1, y2, x2]
    """
    # 计算每个特征点对应的原图区域大小
    stride_y = original_h / feature_h
    stride_x = original_w / feature_w
    
    coords = np.zeros((feature_h, feature_w, 4), dtype=np.float32)
    
    for h in range(feature_h):
        for w in range(feature_w):
            y1 = int(h * stride_y)
            x1 = int(w * stride_x)
            y2 = int((h + 1) * stride_y)
            x2 = int((w + 1) * stride_x)
            # 确保坐标不越界
            y2 = min(y2, original_h)
            x2 = min(x2, original_w)
            coords[h, w] = [y1, x1, y2, x2]
    
    return coords


def coords_to_feature_map(coords: np.ndarray, feature_h: int, feature_w: int) -> np.ndarray:
    """
    将原图坐标列表转换为特征图位置
    
    Args:
        coords: 原图坐标列表，每项为 [y1, x1, y2, x2] 或 [y, x]
        feature_h: 特征图高度
        feature_w: 特征图宽度
        original_h: 原图高度
        original_w: 原图宽度
        
    Returns:
        特征图位置数组
    """
    # 计算每个特征点覆盖的原图范围
    stride_y = original_h / feature_h
    stride_x = original_w / feature_w
    
    positions = []
    for coord in coords:
        if len(coord) == 4:
            cy = (coord[0] + coord[2]) / 2
            cx = (coord[1] + coord[3]) / 2
        else:
            cy, cx = coord[0], coord[1]
        
        h = min(int(cy / stride_y), feature_h - 1)
        w = min(int(cx / stride_x), feature_w - 1)
        positions.append([h, w])
    
    return np.array(positions)


def calculate_distance(query_features: np.ndarray, bank_features: np.ndarray, 
                       method: str = 'cosine') -> np.ndarray:
    """
    计算查询特征与特征库之间的距离
    
    Args:
        query_features: 查询特征，形状 (N, D) 或 (D,)
        bank_features: 特征库，形状 (M, D)
        method: 距离计算方法，'cosine' 或 'euclidean'
        
    Returns:
        距离数组，形状 (N,) 或 (M,) 取决于输入
    """
    if query_features.ndim == 1:
        query_features = query_features.reshape(1, -1)
    
    if method == 'cosine':
        # 余弦距离
        query_norm = np.linalg.norm(query_features, axis=1, keepdims=True)
        bank_norm = np.linalg.norm(bank_features, axis=1, keepdims=True)
        
        query_normalized = query_features / (query_norm + 1e-8)
        bank_normalized = bank_features / (bank_norm.T + 1e-8)
        
        similarities = np.dot(query_normalized, bank_normalized.T)
        distances = 1 - similarities
    elif method == 'euclidean':
        # 欧氏距离
        distances = np.linalg.norm(
            query_features[:, np.newaxis, :] - bank_features[np.newaxis, :, :],
            axis=2
        )
    else:
        raise ValueError(f"不支持的距离计算方法: {method}")
    
    return distances.squeeze()


def generate_boxes_from_scores(scores: np.ndarray, coords: np.ndarray,
                                threshold: float = 0.5, 
                                min_area: int = 100) -> List[Dict]:
    """
    根据异常分数生成矩形框标注
    
    Args:
        scores: 异常分数数组，形状 (H, W)
        coords: 坐标映射数组，形状 (H, W, 4)
        threshold: 分数阈值
        min_area: 最小面积阈值
        
    Returns:
        矩形框列表，每项包含 {'bbox': [x1, y1, x2, y2], 'score': float}
    """
    from scipy import ndimage
    
    # 创建二值掩码
    binary_mask = (scores > threshold).astype(np.uint8)
    
    # 标记连通区域
    labeled, num_features = ndimage.label(binary_mask)
    
    boxes = []
    for i in range(1, num_features + 1):
        # 获取当前连通区域的坐标
        positions = np.where(labeled == i)
        if len(positions[0]) == 0:
            continue
        
        # 获取该区域的最大分数
        region_scores = scores[positions]
        max_score = float(np.max(region_scores))
        
        # 计算边界框
        y_min, y_max = positions[0].min(), positions[0].max()
        x_min, x_max = positions[1].min(), positions[1].max()
        
        # 转换为原图坐标
        bbox = coords[y_min, x_min].copy()  # [y1, x1, y2, x2]
        # 转换为 [x1, y1, x2, y2] 格式
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
        
        # 计算面积
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        if area >= min_area:
            boxes.append({
                'bbox': [int(v) for v in bbox],
                'score': max_score,
                'area': area
            })
    
    # 按分数降序排序
    boxes.sort(key=lambda x: x['score'], reverse=True)
    
    return boxes


def nms(boxes: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    非极大值抑制
    
    Args:
        boxes: 矩形框列表
        iou_threshold: IOU阈值
        
    Returns:
        保留的矩形框列表
    """
    if len(boxes) == 0:
        return []
    
    # 按分数排序
    boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
    
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        
        boxes = [
            box for box in boxes
            if calculate_iou(current['bbox'], box['bbox']) < iou_threshold
        ]
    
    return keep


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个矩形的IOU
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IOU值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def upsample_scores(scores: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    上采样异常分数到目标尺寸
    
    Args:
        scores: 原始分数数组 (H, W)
        target_h: 目标高度
        target_w: 目标宽度
        
    Returns:
        上采样后的分数数组
    """
    from scipy.ndimage import zoom
    
    zoom_h = target_h / scores.shape[0]
    zoom_w = target_w / scores.shape[1]
    
    return zoom(scores, (zoom_h, zoom_w), order=1)


def save_object(obj: object, filepath: str) -> None:
    """
    保存对象到文件
    
    Args:
        obj: 要保存的对象
        filepath: 保存路径
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filepath: str) -> object:
    """
    从文件加载对象
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的对象
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: str) -> None:
    """
    保存JSON数据
    
    Args:
        data: 要保存的字典
        filepath: 保存路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Dict:
    """
    加载JSON数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        Args:
            target_size: 目标尺寸 (height, width)
            mean: 归一化均值
            std: 归一化标准差
        """
        self.target_size = target_size
        self.mean = mean
        self.std = std
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W, 3)，值范围 [0, 255]
            
        Returns:
            预处理后的张量 (3, H, W)
        """
        import torch
        
        # 调整大小
        if image.shape[:2] != self.target_size:
            image = resize_image(image, self.target_size)
        
        # 标准化
        image = normalize_image(image, self.mean, self.std)
        
        # 转换为 (C, H, W) 格式
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float()
