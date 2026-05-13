"""
PatchCore Anomaly Detection - 异常特征库管理模块
支持异常特征的注册、分类存储和查询
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
from utils import save_object, load_object


class AnomalyLibrary:
    """异常特征库，按类别管理异常样本特征"""
    
    def __init__(self):
        """初始化异常特征库"""
        # 按类别存储特征
        # 结构: {category: {'features': np.ndarray, 'coords': np.ndarray, 'count': int}}
        self.categories: Dict[str, Dict] = defaultdict(lambda: {
            'features': [],
            'coords': [],
            'metadata': [],
            'count': 0
        })
        
        # 所有类别的汇总特征
        self.all_features: Optional[np.ndarray] = None
        self.all_coords: Optional[np.ndarray] = None
        self.is_built = False
    
    def register(self, features: np.ndarray, 
                 coords: Optional[np.ndarray] = None,
                 category: str = 'unknown',
                 metadata: Optional[Dict] = None):
        """
        注册异常特征
        
        Args:
            features: 异常特征 (N, D) 或 (D,)
            coords: 对应的坐标 (N, 4)，每行为 [y1, x1, y2, x2]
            category: 异常类别
            metadata: 元数据（如样本路径、标注信息等）
        """
        # 确保features是2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if coords is not None and coords.ndim == 1:
            coords = coords.reshape(1, -1)
        
        self.categories[category]['features'].append(features)
        
        if coords is not None:
            self.categories[category]['coords'].append(coords)
        else:
            self.categories[category]['coords'].append(
                np.zeros((len(features), 4), dtype=np.float32)
            )
        
        if metadata is not None:
            self.categories[category]['metadata'].append(metadata)
        else:
            self.categories[category]['metadata'].append({})
        
        self.categories[category]['count'] += len(features)
        self.is_built = False
    
    def register_from_image(self, extractor, image, coords, category='unknown'):
        """
        从图像注册异常区域特征
        
        Args:
            extractor: 特征提取器
            image: 图像数据
            coords: 原图坐标列表，每项为 [y1, x1, y2, x2]
            category: 异常类别
        """
        # 提取指定坐标区域的特征
        features = extractor.extract_for_coords(image, coords)
        coords_array = np.array(coords)
        
        self.register(features, coords_array, category)
    
    def register_from_annotation_file(self, extractor, image_path: str, 
                                     annotation_path: str = None,
                                     default_category: str = 'unknown',
                                     patch_size: int = 32):
        """
        从标注文件注册异常特征（支持 txt 标注格式）
        
        标注格式: 每行一个异常点，格式为 "类别 中心x 中心y"
        例如:
            0 393 152
            1 200 300
            scratch 100 200  (也支持类别名格式)
        
        Args:
            extractor: 特征提取器
            image_path: 异常图片路径
            annotation_path: 标注文件路径，默认为图片同名的 .txt 文件
            default_category: 默认类别名（数字类别时使用）
            patch_size: 异常区域边长，以中心点为基准生成 patch
        """
        import os
        from .utils import load_image
        
        # 加载图像
        if isinstance(image_path, str):
            image = load_image(image_path)
        else:
            image = image_path
        
        # 确定标注文件路径
        if annotation_path is None:
            base, _ = os.path.splitext(image_path)
            annotation_path = base + '.txt'
        
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"标注文件不存在: {annotation_path}")
        
        # 读取标注
        coords_by_category = defaultdict(list)
        image_h, image_w = image.shape[:2]
        half_patch = patch_size // 2
        
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                # 解析标注
                if parts[0].isdigit():
                    category = f"class_{parts[0]}"
                else:
                    category = parts[0]
                
                cx = int(parts[1])  # 中心 x
                cy = int(parts[2])  # 中心 y
                
                # 转换为边界框 [y1, x1, y2, x2]
                y1 = max(0, cy - half_patch)
                x1 = max(0, cx - half_patch)
                y2 = min(image_h, cy + half_patch)
                x2 = min(image_w, cx + half_patch)
                
                coords_by_category[category].append([y1, x1, y2, x2])
        
        # 提取并注册每个类别的特征
        for category, coords_list in coords_by_category.items():
            if len(coords_list) == 0:
                continue
            
            # 提取特征
            features = extractor.extract_for_coords(image, coords_list)
            coords_array = np.array(coords_list)
            
            # 注册
            self.register(features, coords_array, category, {
                'image_path': image_path,
                'annotation_path': annotation_path
            })
        
        return dict(coords_by_category)
    
    def register_from_directory(self, extractor, directory: str, 
                                extensions: tuple = ('.png', '.jpg', '.jpeg'),
                                default_category: str = 'unknown',
                                patch_size: int = 32):
        """
        从目录批量注册异常样本（自动查找同名 txt 标注文件）
        
        Args:
            extractor: 特征提取器
            directory: 包含异常图片和标注文件的目录
            extensions: 支持的图片扩展名
            default_category: 默认类别名
            patch_size: 异常区域边长
            
        Returns:
            注册摘要 {'total_images': int, 'total_anomalies': int, 'categories': dict}
        """
        import os
        import glob
        
        summary = {
            'total_images': 0,
            'total_anomalies': 0,
            'categories': defaultdict(int)
        }
        
        # 查找所有图片文件
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, f'*{ext}')))
            image_paths.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
        
        for image_path in image_paths:
            base, _ = os.path.splitext(image_path)
            annotation_path = base + '.txt'
            
            if os.path.exists(annotation_path):
                try:
                    coords_info = self.register_from_annotation_file(
                        extractor, image_path, annotation_path,
                        default_category, patch_size
                    )
                    
                    summary['total_images'] += 1
                    for cat, coords in coords_info.items():
                        summary['total_anomalies'] += len(coords)
                        summary['categories'][cat] += len(coords)
                        
                except Exception as e:
                    print(f"警告: 处理 {image_path} 失败 - {e}")
        
        return summary
    
    def build(self, normalize: bool = True):
        """
        构建异常特征库
        
        Args:
            normalize: 是否对特征进行L2归一化
        """
        all_features_list = []
        all_coords_list = []
        
        for category, data in self.categories.items():
            if len(data['features']) == 0:
                continue
            
            features = np.vstack(data['features'])
            coords = np.vstack(data['coords'])
            
            if normalize:
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                features = features / (norms + 1e-8)
            
            data['features'] = features
            data['coords'] = coords
            data['built'] = True
            
            all_features_list.append(features)
            all_coords_list.append(coords)
        
        if all_features_list:
            self.all_features = np.vstack(all_features_list)
            self.all_coords = np.vstack(all_coords_list)
            self.is_built = True
        
        print(f"异常特征库构建完成:")
        for category, data in self.categories.items():
            print(f"  - {category}: {data['count']} 个特征点")
    
    def get_category(self, category: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定类别的特征
        
        Args:
            category: 类别名称
            
        Returns:
            features: 类别特征
            coords: 对应坐标
        """
        if category not in self.categories:
            return np.array([]), np.array([])
        
        data = self.categories[category]
        return data['features'], data['coords']
    
    def get_categories(self) -> List[str]:
        """获取所有类别列表"""
        return list(self.categories.keys())
    
    def get_category_stats(self) -> Dict[str, int]:
        """获取各类别的统计信息"""
        return {cat: data['count'] for cat, data in self.categories.items()}
    
    def query(self, features: np.ndarray, k: int = 5, 
              category: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        查询与给定特征最相似的异常特征
        
        Args:
            features: 查询特征 (N, D)
            k: 返回的最近邻数量
            category: 指定类别，None表示所有类别
            
        Returns:
            distances: 距离数组 (N, k)
            indices: 索引数组 (N, k)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if category is not None:
            library_features = self.categories[category]['features']
            library_coords = self.categories[category]['coords']
        else:
            if not self.is_built:
                raise RuntimeError("请先调用 build() 方法")
            library_features = self.all_features
            library_coords = self.all_coords
        
        # 计算余弦距离
        similarities = np.dot(features, library_features.T)
        distances = 1 - similarities
        
        # 获取k个最近邻
        indices = np.argsort(distances, axis=1)[:, :k]
        
        return distances, indices
    
    def calculate_similarity(self, features: np.ndarray, 
                             category: Optional[str] = None) -> np.ndarray:
        """
        计算与异常特征库的相似度
        
        Args:
            features: 待检测特征 (N, D)
            category: 指定类别
            
        Returns:
            最大相似度数组 (N,)
        """
        distances, _ = self.query(features, k=1, category=category)
        return 1 - distances.squeeze()
    
    def remove_category(self, category: str) -> bool:
        """
        删除指定类别
        
        Args:
            category: 类别名称
            
        Returns:
            是否成功删除
        """
        if category in self.categories:
            del self.categories[category]
            self.is_built = False
            return True
        return False
    
    def merge_categories(self, source_categories: List[str], target_category: str):
        """
        合并多个类别
        
        Args:
            source_categories: 源类别列表
            target_category: 目标类别名称
        """
        for cat in source_categories:
            if cat in self.categories:
                data = self.categories[cat]
                self.register(
                    data['features'], 
                    data['coords'], 
                    target_category
                )
                del self.categories[cat]
        
        self.is_built = False
    
    @property
    def total_features(self) -> int:
        """获取总特征数量"""
        return sum(data['count'] for data in self.categories.values())
    
    @property
    def feature_dim(self) -> int:
        """获取特征维度"""
        if self.total_features > 0:
            for data in self.categories.values():
                if len(data['features']) > 0:
                    features = np.vstack(data['features'])
                    return features.shape[1]
        return 0
    
    def save(self, filepath: str) -> None:
        """
        保存异常特征库到文件
        
        Args:
            filepath: 保存路径
        """
        state = {
            'categories': dict(self.categories),
            'all_features': self.all_features,
            'all_coords': self.all_coords,
            'is_built': self.is_built
        }
        save_object(state, filepath)
        print(f"异常特征库已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AnomalyLibrary':
        """
        从文件加载异常特征库
        
        Args:
            filepath: 文件路径
            
        Returns:
            AnomalyLibrary实例
        """
        state = load_object(filepath)
        
        library = cls()
        library.categories = defaultdict(lambda: {
            'features': [], 'coords': [], 'metadata': []
        })
        
        # 恢复类别数据
        for cat, data in state['categories'].items():
            for key in ['features', 'coords']:
                if key in data and isinstance(data[key], list):
                    library.categories[cat][key] = data[key]
                elif key in data:
                    library.categories[cat][key] = [data[key]]
            
            library.categories[cat]['metadata'] = data.get('metadata', [])
            library.categories[cat]['count'] = len(library.categories[cat]['features'])
        
        library.all_features = state['all_features']
        library.all_coords = state['all_coords']
        library.is_built = state['is_built']
        
        print(f"异常特征库已加载: {library.total_features} 个特征点, {len(library.categories)} 个类别")
        
        return library
    
    def __len__(self) -> int:
        return self.total_features
    
    def __repr__(self) -> str:
        stats = self.get_category_stats()
        return f"AnomalyLibrary(categories={len(stats)}, total_features={self.total_features})"


class AnomalyLibraryManager:
    """异常特征库管理器，支持多个库的管理"""
    
    def __init__(self):
        """初始化管理器"""
        self.libraries: Dict[str, AnomalyLibrary] = {}
        self.active_library: Optional[str] = None
    
    def create_library(self, name: str) -> AnomalyLibrary:
        """
        创建新的异常特征库
        
        Args:
            name: 库名称
            
        Returns:
            创建的库实例
        """
        library = AnomalyLibrary()
        self.libraries[name] = library
        self.active_library = name
        return library
    
    def get_library(self, name: str) -> Optional[AnomalyLibrary]:
        """
        获取指定名称的库
        
        Args:
            name: 库名称
            
        Returns:
            库实例，如果不存在则返回None
        """
        return self.libraries.get(name)
    
    def set_active(self, name: str) -> bool:
        """
        设置活跃库
        
        Args:
            name: 库名称
            
        Returns:
            是否成功
        """
        if name in self.libraries:
            self.active_library = name
            return True
        return False
    
    def get_active(self) -> Optional[AnomalyLibrary]:
        """获取当前活跃库"""
        if self.active_library:
            return self.libraries.get(self.active_library)
        return None
    
    def save_all(self, base_path: str) -> None:
        """
        保存所有库
        
        Args:
            base_path: 保存基础路径
        """
        for name, library in self.libraries.items():
            filepath = f"{base_path}_{name}.pkl"
            library.save(filepath)
    
    @classmethod
    def load_all(cls, base_path: str) -> 'AnomalyLibraryManager':
        """
        从基础路径加载所有库
        
        Args:
            base_path: 基础路径（不含后缀）
            
        Returns:
            加载的管理器
        """
        manager = cls()
        # 实际使用时需要根据实际保存的文件模式调整
        return manager