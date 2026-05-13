"""
PatchCore Anomaly Detection - 特征提取器
支持预训练模型的多层特征提取和坐标映射
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from utils import resize_image, normalize_image, feature_map_to_coords


class FeatureExtractor:
    """特征提取器，支持多种预训练模型和多层特征融合"""
    
    SUPPORTED_MODELS = {
        'resnet18': {
            'class': models.resnet18,
            'layers': ['layer1', 'layer2', 'layer3', 'layer4'],
            'layer_dims': [64, 128, 256, 512],
            'default_layers': ['layer2', 'layer3']
        },
        'resnet34': {
            'class': models.resnet34,
            'layers': ['layer1', 'layer2', 'layer3', 'layer4'],
            'layer_dims': [64, 128, 256, 512],
            'default_layers': ['layer2', 'layer3']
        },
        'resnet50': {
            'class': models.resnet50,
            'layers': ['layer1', 'layer2', 'layer3', 'layer4'],
            'layer_dims': [256, 512, 1024, 2048],
            'default_layers': ['layer2', 'layer3']
        },
        'wide_resnet50_2': {
            'class': models.wide_resnet50_2,
            'layers': ['layer1', 'layer2', 'layer3', 'layer4'],
            'layer_dims': [256, 512, 1024, 2048],
            'default_layers': ['layer2', 'layer3']
        }
    }
    
    def __init__(self, model_name: str = 'resnet18', 
                 layers: Optional[List[str]] = None,
                 pretrained: bool = True,
                 device: Optional[str] = None):
        """
        初始化特征提取器
        
        Args:
            model_name: 模型名称，支持 resnet18/34/50, wide_resnet50_2
            layers: 要提取的层列表，默认为 ['layer2', 'layer3']
            pretrained: 是否使用预训练权重
            device: 运行设备，'cpu', 'cuda' 或 'cuda:N'
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}，支持的模型: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_info = self.SUPPORTED_MODELS[model_name]
        self.layers = layers or self.model_info['default_layers']
        
        # 验证layers参数
        for layer in self.layers:
            if layer not in self.model_info['layers']:
                raise ValueError(f"不支持的层 {layer}，支持的层: {self.model_info['layers']}")
        
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载模型
        self.model = self._load_model(pretrained)
        self.model.eval()
        self.model.to(self.device)
        
        # 注册钩子
        self.hooks = []
        self.feature_maps = {}
        self._register_hooks()
    
    def _load_model(self, pretrained: bool) -> nn.Module:
        """加载模型"""
        model_class = self.model_info['class']
        if pretrained:
            weights = 'DEFAULT' if hasattr(model_class, 'DEFAULT') else None
            model = model_class(weights=weights)
        else:
            model = model_class(weights=None)
        return model
    
    def _register_hooks(self):
        """注册前向钩子以捕获中间层特征"""
        self.feature_maps = {}
        
        # 移除已有钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        def get_hook(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        for layer_name in self.layers:
            layer = dict(self.model.named_modules()).get(layer_name)
            if layer is not None:
                hook = layer.register_forward_hook(get_hook(layer_name))
                self.hooks.append(hook)
    
    def extract(self, image: Union[np.ndarray, torch.Tensor],
                target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取图像特征
        
        Args:
            image: 输入图像，numpy数组 (H, W, 3) 或 torch张量 (C, H, W)
            target_size: 目标尺寸 (height, width)，默认为模型输入尺寸
            
        Returns:
            features: 特征数组，形状为 (N, D)，N为特征点数，D为特征维度
            coords: 对应的坐标映射，形状为 (N, 4)，每行为 [y1, x1, y2, x2]
        """
        import torch
        
        # 预处理输入
        if isinstance(image, np.ndarray):
            original_h, original_w = image.shape[:2]
            if target_size is None:
                target_size = self._get_default_input_size()
            processed = self._preprocess(image, target_size)
            # 添加 batch 维度
            if processed.dim() == 3:
                processed = processed.unsqueeze(0)
        else:
            original_h, original_w = image.shape[1], image.shape[2]
            if target_size is None:
                target_size = (original_h, original_w)
            processed = image
            if processed.dim() == 3:
                processed = processed.unsqueeze(0)
        
        # 提取特征
        with torch.no_grad():
            self.model(processed.to(self.device))
        
        # 获取特征图并融合
        feature_maps = [self.feature_maps[layer] for layer in self.layers if layer in self.feature_maps]
        
        if len(feature_maps) == 0:
            raise RuntimeError("未能提取到任何特征，请检查模型层配置")
        
        # 上采样所有特征图到相同尺寸
        target_h, target_w = feature_maps[0].shape[2], feature_maps[0].shape[3]
        
        upsampled_maps = []
        for fm in feature_maps:
            if fm.shape[2:] != (target_h, target_w):
                fm = nn.functional.interpolate(fm, size=(target_h, target_w), mode='bilinear', align_corners=False)
            upsampled_maps.append(fm)
        
        # 拼接特征
        combined_features = torch.cat(upsampled_maps, dim=1)  # (B, D, H, W)
        
        # 重塑为 (H*W, D)
        features = combined_features.squeeze(0).permute(1, 2, 0)  # (H, W, D)
        features = features.reshape(-1, features.shape[-1]).cpu().numpy()  # (N, D)
        
        # 生成坐标映射
        coords = feature_map_to_coords(target_h, target_w, original_h, original_w)
        coords = coords.reshape(-1, 4)  # (N, 4)
        
        return features, coords
    
    def extract_for_coords(self, image: Union[np.ndarray, torch.Tensor],
                           coords: List[List[int]],
                           target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        提取指定坐标位置的图像块特征
        
        Args:
            image: 输入图像
            coords: 坐标列表，每项为 [y1, x1, y2, x2]
            target_size: 目标尺寸
            
        Returns:
            特征数组，形状为 (N, D)
        """
        import torch
        from torchvision import transforms
        
        if isinstance(image, np.ndarray):
            original_h, original_w = image.shape[:2]
            if target_size is None:
                target_size = self._get_default_input_size()
            processed = self._preprocess(image, target_size)
            # 添加 batch 维度
            if processed.dim() == 3:
                processed = processed.unsqueeze(0)
        else:
            original_h, original_w = image.shape[1], image.shape[2]
            processed = image
            if processed.dim() == 3:
                processed = processed.unsqueeze(0)
        
        # 计算缩放比例
        scale_y = target_size[0] / original_h
        scale_x = target_size[1] / original_w
        
        # 提取每个坐标区域的特征
        features = []
        patch_size = self._get_patch_size()
        
        with torch.no_grad():
            self.model(processed.to(self.device))
        
        feature_maps = [self.feature_maps[layer] for layer in self.layers if layer in self.feature_maps]
        
        # 获取特征图尺寸
        feat_h, feat_w = feature_maps[0].shape[2], feature_maps[0].shape[3]
        feat_scale_y = feat_h / target_size[0]
        feat_scale_x = feat_w / target_size[1]
        
        for coord in coords:
            # 转换为特征图坐标
            y1, x1, y2, x2 = coord
            feat_y1 = int(y1 * scale_y * feat_scale_y)
            feat_x1 = int(x1 * scale_x * feat_scale_x)
            feat_y2 = int(y2 * scale_y * feat_scale_y)
            feat_x2 = int(x2 * scale_x * feat_scale_x)
            
            # 确保坐标在范围内
            feat_y1, feat_x1 = max(0, feat_y1), max(0, feat_x1)
            feat_y2, feat_x2 = min(feat_h, feat_y2), min(feat_w, feat_x2)
            
            # 提取区域特征
            region_features = []
            for fm in feature_maps:
                region = fm[:, :, feat_y1:feat_y2, feat_x1:feat_x2]
                pooled = nn.functional.adaptive_avg_pool2d(region, (1, 1))
                region_features.append(pooled.squeeze(-1).squeeze(-1))
            
            combined = torch.cat(region_features, dim=1)
            features.append(combined.cpu().numpy())
        
        return np.array(features)
    
    def _preprocess(self, image: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """预处理图像"""
        # 调整大小
        if image.shape[:2] != target_size:
            image = resize_image(image, target_size)
        
        # 标准化
        image = normalize_image(image)
        
        # 转换为 (C, H, W) 格式
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float()
    
    def _get_default_input_size(self) -> Tuple[int, int]:
        """获取默认输入尺寸"""
        if self.model_name.startswith('resnet'):
            return (224, 224)
        return (224, 224)
    
    def _get_patch_size(self) -> int:
        """获取使用的特征层对应的感受野大小"""
        # 粗略估计：layer2 + layer3 的组合感受野
        if 'layer4' in self.layers:
            return 32
        elif 'layer3' in self.layers:
            return 16
        elif 'layer2' in self.layers:
            return 8
        return 4
    
    def get_feature_info(self) -> Dict:
        """获取特征提取器信息"""
        total_dim = sum(self.model_info['layer_dims'][self.model_info['layers'].index(l)] 
                       for l in self.layers)
        return {
            'model_name': self.model_name,
            'layers': self.layers,
            'feature_dim': total_dim,
            'device': self.device
        }
    
    @property
    def feature_dim(self) -> int:
        """获取特征维度"""
        total_dim = 0
        for layer in self.layers:
            idx = self.model_info['layers'].index(layer)
            total_dim += self.model_info['layer_dims'][idx]
        return total_dim
    
    def to(self, device: str):
        """移动到指定设备"""
        self.device = device
        self.model.to(device)
        return self
    
    def __del__(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()
    
    def __repr__(self) -> str:
        return f"FeatureExtractor(model={self.model_name}, layers={self.layers}, device={self.device})"
