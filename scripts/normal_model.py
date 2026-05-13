"""
PatchCore Anomaly Detection - 正常特征建模模块
构建和管理正常样本的特征模型
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
from feature_extractor import FeatureExtractor
from utils import save_object, load_object


class NormalModel:
    """正常品特征模型"""
    
    def __init__(self, extractor: FeatureExtractor, 
                 sample_size: Optional[int] = None,
                 random_seed: int = 42):
        """
        初始化正常模型
        
        Args:
            extractor: 特征提取器
            sample_size: 每个样本采样的特征点数量，None表示使用全部特征
            random_seed: 随机种子
        """
        self.extractor = extractor
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        self.features = []  # 存储所有正常样本的特征
        self.coords = []     # 存储对应的坐标
        self.is_built = False
        
        # PatchCore 采样器
        self.patch_indices = None
        
    def add_sample(self, image: Union[np.ndarray, torch.Tensor],
                   target_size: Optional[Tuple[int, int]] = None):
        """
        添加一个正常样本
        
        Args:
            image: 输入图像
            target_size: 目标尺寸
        """
        features, coords = self.extractor.extract(image, target_size)
        
        # 应用采样
        if self.sample_size is not None and len(features) > self.sample_size:
            np.random.seed(self.random_seed)
            indices = np.random.choice(len(features), self.sample_size, replace=False)
            features = features[indices]
            coords = coords[indices]
        
        self.features.append(features)
        self.coords.append(coords)
        self.is_built = False
    
    def build(self, normalize: bool = True):
        """
        构建特征库
        
        Args:
            normalize: 是否对特征进行L2归一化
        """
        if len(self.features) == 0:
            raise ValueError("没有添加任何正常样本")
        
        # 合并所有样本的特征
        self.all_features = np.vstack(self.features)
        self.all_coords = np.vstack(self.coords)
        
        # 归一化
        if normalize:
            norms = np.linalg.norm(self.all_features, axis=1, keepdims=True)
            self.all_features = self.all_features / (norms + 1e-8)
        
        self.is_built = True
        
        print(f"正常特征库构建完成: {len(self.all_features)} 个特征点, 维度: {self.feature_dim}")
    
    def query(self, features: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        查询与给定特征最相似的正常特征
        
        Args:
            features: 查询特征 (N, D)
            k: 返回的最近邻数量
            
        Returns:
            distances: 距离数组 (N, k)
            indices: 索引数组 (N, k)
            features: 相似特征 (N, k, D)
        """
        if not self.is_built:
            raise RuntimeError("请先调用 build() 方法构建特征库")
        
        # 计算余弦距离
        similarities = np.dot(features, self.all_features.T)
        distances = 1 - similarities
        
        # 获取k个最近邻
        indices = np.argsort(distances, axis=1)[:, :k]
        
        # 收集相似特征
        k_features = np.array([
            self.all_features[idx] for idx in indices
        ])
        
        # 获取对应的距离
        k_distances = np.array([
            distances[i, idx] for i, idx in enumerate(indices)
        ])
        
        return k_distances, indices, k_features
    
    def calculate_anomaly_score(self, features: np.ndarray, k: int = 5) -> np.ndarray:
        """
        计算异常分数（PatchCore核心算法）
        
        异常分数 = 与正常特征库中k个最近邻距离的最大值
        
        Args:
            features: 待检测特征 (N, D)
            k: 使用的最近邻数量
            
        Returns:
            异常分数数组 (N,)
        """
        if not self.is_built:
            raise RuntimeError("请先调用 build() 方法构建特征库")
        
        # 归一化查询特征
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        query_features = features / (norms + 1e-8)
        
        # 计算与所有正常特征的距离
        distances, _, _ = self.query(query_features, k=k)
        
        # 取每个查询点与k个最近邻距离的最大值作为异常分数
        scores = np.max(distances, axis=1)
        
        return scores
    
    @property
    def feature_dim(self) -> int:
        """获取特征维度"""
        if len(self.features) > 0:
            return self.features[0].shape[1]
        return self.extractor.feature_dim
    
    def save(self, filepath: str) -> None:
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        state = {
            'features': self.all_features,
            'coords': self.all_coords,
            'sample_size': self.sample_size,
            'random_seed': self.random_seed,
            'feature_dim': self.feature_dim,
            'is_built': self.is_built,
            'model_name': self.extractor.model_name,
            'layers': self.extractor.layers
        }
        save_object(state, filepath)
        print(f"正常模型已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str, extractor: Optional[FeatureExtractor] = None) -> 'NormalModel':
        """
        从文件加载模型
        
        Args:
            filepath: 模型路径
            extractor: 特征提取器，如果为None则从保存的状态恢复
            
        Returns:
            NormalModel实例
        """
        state = load_object(filepath)
        
        if extractor is None:
            extractor = FeatureExtractor(
                model_name=state['model_name'],
                layers=state['layers']
            )
        
        model = cls(extractor, state['sample_size'], state['random_seed'])
        model.all_features = state['features']
        model.all_coords = state['coords']
        model.is_built = state['is_built']
        
        print(f"正常模型已加载: {len(model.all_features)} 个特征点")
        
        return model
    
    def __len__(self) -> int:
        """返回特征库中的特征数量"""
        if self.is_built:
            return len(self.all_features)
        return sum(len(f) for f in self.features)
    
    def __repr__(self) -> str:
        status = "已构建" if self.is_built else "未构建"
        return f"NormalModel(features={len(self)}, status={status})"


class HierarchicalNormalModel(NormalModel):
    """分层正常模型，支持多尺度特征"""
    
    def __init__(self, extractor: FeatureExtractor,
                 sample_size: Optional[int] = None,
                 random_seed: int = 42):
        super().__init__(extractor, sample_size, random_seed)
        self.layer_features = {}  # 按层存储特征
    
    def add_sample(self, image: Union[np.ndarray, torch.Tensor],
                   target_size: Optional[Tuple[int, int]] = None):
        """添加样本并按层存储特征"""
        import torch
        
        # 保存当前钩子状态
        original_maps = dict(self.extractor.feature_maps)
        
        # 提取特征
        features, coords = self.extractor.extract(image, target_size)
        
        # 恢复特征图（包含各层信息）
        # 由于我们已经在extractor中提取，这里简化处理
        
        self.features.append(features)
        self.coords.append(coords)
        self.is_built = False
    
    def build(self, normalize: bool = True):
        """构建分层特征库"""
        super().build(normalize)
        
        # 按坐标位置组织特征（用于局部异常检测）
        self._build_spatial_index()
    
    def _build_spatial_index(self):
        """构建空间索引以支持局部异常检测"""
        # 简化：按网格划分
        if not self.is_built or len(self.all_coords) == 0:
            return
        
        # 使用k-d tree进行空间索引
        from scipy.spatial import cKDTree
        
        coords_2d = self.all_coords[:, :2]  # 使用左上角坐标
        self.spatial_tree = cKDTree(coords_2d)
    
    def query_local(self, coords: np.ndarray, radius: float, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        查询局部区域的正常特征
        
        Args:
            coords: 查询坐标 (N, 4) [y1, x1, y2, x2]
            radius: 查询半径
            k: 最大返回数量
            
        Returns:
            indices: 局部特征的索引
            distances: 对应距离
        """
        if not hasattr(self, 'spatial_tree'):
            raise RuntimeError("请先调用 build() 方法")
        
        # 计算查询中心点
        centers = np.mean(coords[:, :2], axis=1) if coords.shape[1] >= 2 else coords
        
        # 空间查询
        results = []
        for center in centers:
            idx = self.spatial_tree.query_ball_point(center, radius)
            if len(idx) > k:
                idx = np.random.choice(idx, k, replace=False)
            results.append(np.array(idx))
        
        return results


class EnsembleNormalModel:
    """集成正常模型，支持多个模型组合"""
    
    def __init__(self, models: List[NormalModel], weights: Optional[List[float]] = None):
        """
        初始化集成模型
        
        Args:
            models: 正常模型列表
            weights: 每个模型的权重，默认为等权重
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("权重数量必须与模型数量一致")
    
    def calculate_anomaly_score(self, features: np.ndarray, k: int = 5) -> np.ndarray:
        """
        计算集成异常分数
        
        Args:
            features: 待检测特征
            k: 最近邻数量
            
        Returns:
            加权平均的异常分数
        """
        scores_list = []
        for model, weight in zip(self.models, self.weights):
            scores = model.calculate_anomaly_score(features, k)
            scores_list.append(scores * weight)
        
        return np.sum(scores_list, axis=0)
    
    def query(self, features: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """查询多个模型的结果"""
        all_distances = []
        all_indices = []
        all_features = []
        
        for model in self.models:
            distances, indices, features_found = model.query(features, k)
            all_distances.append(distances)
            all_indices.append(indices)
            all_features.append(features_found)
        
        # 合并所有结果
        combined_distances = np.hstack(all_distances)
        combined_indices = np.hstack(all_indices)
        combined_features = np.vstack([
            f.reshape(-1, f.shape[-1]) for f in all_features
        ])
        
        # 排序并取前k个
        sorted_idx = np.argsort(combined_distances, axis=1)[:, :k]
        
        final_distances = np.array([
            combined_distances[i, idx] for i, idx in enumerate(sorted_idx)
        ])
        final_features = np.array([
            combined_features[idx] for i, idx in enumerate(sorted_idx)
        ])
        
        return final_distances, sorted_idx, final_features
    
    def __len__(self) -> int:
        return sum(len(m) for m in self.models)
