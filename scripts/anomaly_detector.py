"""
PatchCore Anomaly Detection - 异常检测器模块
整合特征提取、正常模型和异常库进行异常检测
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from feature_extractor import FeatureExtractor
from normal_model import NormalModel
from anomaly_library import AnomalyLibrary
from utils import (generate_boxes_from_scores, upsample_scores, 
                    nms, calculate_distance)


class AnomalyDetector:
    """异常检测器，整合所有组件进行异常检测"""
    
    def __init__(self, 
                 extractor: FeatureExtractor,
                 normal_model: NormalModel,
                 anomaly_library: Optional[AnomalyLibrary] = None,
                 k_neighbors: int = 5,
                 score_threshold: float = 0.5,
                 nms_threshold: float = 0.5,
                 min_box_area: int = 100):
        """
        初始化异常检测器
        
        Args:
            extractor: 特征提取器
            normal_model: 正常模型
            anomaly_library: 异常特征库（可选）
            k_neighbors: 用于计算异常分数的近邻数量
            score_threshold: 异常分数阈值
            nms_threshold: NMS阈值
            min_box_area: 最小框面积
        """
        self.extractor = extractor
        self.normal_model = normal_model
        self.anomaly_library = anomaly_library
        self.k_neighbors = k_neighbors
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.min_box_area = min_box_area
        
        self.original_size: Optional[Tuple[int, int]] = None
    
    def detect(self, image: Union[np.ndarray, str],
              target_size: Optional[Tuple[int, int]] = None,
              return_coords: bool = False) -> Dict:
        """
        检测图像中的异常
        
        Args:
            image: 输入图像（numpy数组或图像路径）
            target_size: 目标尺寸
            return_coords: 是否返回原始坐标格式
            
        Returns:
            检测结果字典:
            {
                'anomaly': bool,           # 是否检测到异常
                'has_anomaly': bool,      # 同anomaly
                'class': str,              # 异常类别（未知或具体类别）
                'confidence': float,       # 异常置信度
                'boxes': List[[x1,y1,x2,y2]],  # 异常区域矩形框
                'scores': List[float],     # 每个框的分数
                'score_map': np.ndarray,  # 异常分数图
                'feature_map_shape': Tuple[int, int]  # 特征图形状
            }
        """
        import torch
        
        # 加载图像
        if isinstance(image, str):
            from utils import load_image
            image = load_image(image)
        
        # 保存原始尺寸
        self.original_size = image.shape[:2]
        
        # 提取特征
        features, coords = self.extractor.extract(image, target_size)
        # 修复：正确计算特征图维度
        # 对于大多数CNN特征提取器，特征图是正方形的
        total_points = len(features)
        feature_size = int(np.sqrt(total_points))
        if feature_size * feature_size != total_points:
            # 如果不是完全平方数，寻找因数分解
            feature_h = feature_w = feature_size
            for h in range(feature_size, 0, -1):
                if total_points % h == 0:
                    feature_h = h
                    feature_w = total_points // h
                    break
        else:
            feature_h = feature_w = feature_size
        
        # 计算异常分数
        anomaly_scores = self.normal_model.calculate_anomaly_score(
            features, k=self.k_neighbors
        )
        
        # 重塑为特征图形状
        score_map = anomaly_scores.reshape(feature_h, feature_w)
        
        # 上采样到原图尺寸
        upsampled_scores = upsample_scores(score_map, self.original_size[0], self.original_size[1])
        
        # 生成坐标映射
        from utils import feature_map_to_coords
        full_coords = feature_map_to_coords(
            feature_h, feature_w, 
            self.original_size[0], self.original_size[1]
        )
        
        # 生成矩形框
        boxes = generate_boxes_from_scores(
            upsampled_scores, 
            full_coords,
            threshold=self.score_threshold,
            min_area=self.min_box_area
        )
        
        # NMS去重
        if boxes:
            boxes = nms(boxes, iou_threshold=self.nms_threshold)
        
        # 判断异常类别
        anomaly_class = 'unknown'
        if self.anomaly_library is not None and len(boxes) > 0:
            anomaly_class = self._classify_anomaly(features, boxes, coords)
        
        # 计算置信度
        confidence = float(np.max(anomaly_scores)) if len(anomaly_scores) > 0 else 0.0
        
        result = {
            'anomaly': len(boxes) > 0,
            'has_anomaly': len(boxes) > 0,
            'class': anomaly_class,
            'confidence': confidence,
            'boxes': [b['bbox'] for b in boxes],
            'scores': [b['score'] for b in boxes],
            'score_map': upsampled_scores,
            'feature_map_shape': (feature_h, feature_w)
        }
        
        if return_coords:
            result['raw_coords'] = coords
            result['raw_scores'] = anomaly_scores
        
        return result
    
    def detect_multi_scale(self, image: Union[np.ndarray, str],
                           target_sizes: List[Tuple[int, int]] = None) -> Dict:
        """
        多尺度异常检测
        
        Args:
            image: 输入图像
            target_sizes: 目标尺寸列表，默认为 [224, 256, 288]
            
        Returns:
            综合多尺度结果的检测结果
        """
        if target_sizes is None:
            target_sizes = [(224, 224), (256, 256), (288, 288)]
        
        all_results = []
        
        for size in target_sizes:
            result = self.detect(image, target_size=size)
            all_results.append(result)
        
        # 融合多尺度结果
        fused_score_map = np.mean([r['score_map'] for r in all_results], axis=0)
        
        # 合并boxes
        all_boxes = []
        for r in all_results:
            for bbox, score in zip(r['boxes'], r['scores']):
                all_boxes.append({'bbox': bbox, 'score': score})
        
        # NMS去重
        final_boxes = nms(all_boxes, iou_threshold=self.nms_threshold)
        
        confidence = float(np.max(fused_score_map))
        
        return {
            'anomaly': len(final_boxes) > 0,
            'has_anomaly': len(final_boxes) > 0,
            'class': 'unknown',
            'confidence': confidence,
            'boxes': [b['bbox'] for b in final_boxes],
            'scores': [b['score'] for b in final_boxes],
            'score_map': fused_score_map,
            'multi_scale_results': all_results
        }
    
    def detect_with_abnormal_library(self, image: Union[np.ndarray, str],
                                     target_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        使用异常库的增强检测
        
        Args:
            image: 输入图像
            target_size: 目标尺寸
            
        Returns:
            增强后的检测结果
        """
        # 基础检测
        base_result = self.detect(image, target_size)
        
        if self.normal_model is None or not base_result['anomaly']:
            return base_result
        
        # 加载图像
        if isinstance(image, str):
            from utils import load_image
            image = load_image(image)
        
        self.original_size = image.shape[:2]
        
        # 提取特征
        features, coords = self.extractor.extract(image, target_size)
        feature_h = int(coords[:, 0].max()) + 1
        feature_w = int(coords[:, 2].max()) + 1
        
        # 计算与异常库的相似度
        if len(features) > 0:
            anomaly_similarities = self.anomaly_library.calculate_similarity(features)
            
            # 结合正常模型和异常库的结果
            normal_scores = self.normal_model.calculate_anomaly_score(
                features, k=self.k_neighbors
            )
            
            # 综合分数：正常模型分数高 + 异常库相似度高 → 异常
            combined_scores = normal_scores * 0.5 + anomaly_similarities * 0.5
            
            # 重新生成boxes
            score_map = combined_scores.reshape(feature_h, feature_w)
            upsampled_scores = upsample_scores(
                score_map, 
                self.original_size[0], 
                self.original_size[1]
            )
            
            from utils import feature_map_to_coords
            full_coords = feature_map_to_coords(
                feature_h, feature_w,
                self.original_size[0], 
                self.original_size[1]
            )
            
            boxes = generate_boxes_from_scores(
                upsampled_scores,
                full_coords,
                threshold=self.score_threshold,
                min_area=self.min_box_area
            )
            
            boxes = nms(boxes, iou_threshold=self.nms_threshold)
            
            # 分类
            anomaly_class = self._classify_anomaly(features, boxes, coords)
            
            base_result.update({
                'class': anomaly_class,
                'confidence': float(np.max(combined_scores)),
                'boxes': [b['bbox'] for b in boxes],
                'scores': [b['score'] for b in boxes],
                'score_map': upsampled_scores,
                'combined_scores': combined_scores
            })
        
        return base_result
    
    def _classify_anomaly(self, features: np.ndarray, boxes: List[Dict],
                         coords: np.ndarray) -> str:
        """
        根据异常区域特征分类异常类型
        
        Args:
            features: 所有特征
            boxes: 检测到的异常框
            coords: 特征坐标
            
        Returns:
            分类结果类别名称
        """
        if self.anomaly_library is None or len(boxes) == 0:
            return 'unknown'
        
        # 获取每个box中心点对应的特征
        box_features = []
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 找到最近的特征点
            feat_coords = coords[:, :2]  # [y1, x1]
            distances = np.sqrt(
                (feat_coords[:, 0] - cy) ** 2 + 
                (feat_coords[:, 1] - cx) ** 2
            )
            nearest_idx = np.argmin(distances)
            box_features.append(features[nearest_idx])
        
        box_features = np.array(box_features)
        
        # 查询异常库进行分类
        category_scores = {}
        for category in self.anomaly_library.get_categories():
            similarities = self.anomaly_library.calculate_similarity(
                box_features, category=category
            )
            category_scores[category] = float(np.max(similarities))
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'unknown'
    
    def analyze_image(self, image: Union[np.ndarray, str],
                     target_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        详细分析图像
        
        Args:
            image: 输入图像
            target_size: 目标尺寸
            
        Returns:
            详细分析结果
        """
        if isinstance(image, str):
            from utils import load_image
            image = load_image(image)
        
        self.original_size = image.shape[:2]
        features, coords = self.extractor.extract(image, target_size)
        
        # 基础统计
        normal_scores = self.normal_model.calculate_anomaly_score(
            features, k=self.k_neighbors
        )
        
        stats = {
            'image_shape': image.shape,
            'feature_count': len(features),
            'feature_dim': features.shape[1],
            'score_statistics': {
                'mean': float(np.mean(normal_scores)),
                'std': float(np.std(normal_scores)),
                'min': float(np.min(normal_scores)),
                'max': float(np.max(normal_scores)),
                'median': float(np.median(normal_scores)),
                'percentile_95': float(np.percentile(normal_scores, 95)),
                'percentile_99': float(np.percentile(normal_scores, 99))
            }
        }
        
        # 如果有异常库，添加分类信息
        if self.anomaly_library is not None:
            similarities = self.anomaly_library.calculate_similarity(features)
            stats['anomaly_similarities'] = {
                'mean': float(np.mean(similarities)),
                'max': float(np.max(similarities))
            }
        
        return stats
    
    def set_threshold(self, threshold: float):
        """设置异常分数阈值"""
        self.score_threshold = threshold
    
    def set_nms_threshold(self, threshold: float):
        """设置NMS阈值"""
        self.nms_threshold = threshold
    
    def __repr__(self) -> str:
        lib_info = f", anomaly_library={len(self.anomaly_library)} categories" if self.anomaly_library else ""
        return (f"AnomalyDetector(k={self.k_neighbors}, "
                f"threshold={self.score_threshold}, "
                f"normal_features={len(self.anomaly_library)}{lib_info})")


class BatchAnomalyDetector:
    """批量异常检测器"""
    
    def __init__(self, detector: AnomalyDetector):
        """
        Args:
            detector: 单个异常检测器实例
        """
        self.detector = detector
    
    def detect_batch(self, images: List[Union[np.ndarray, str]],
                    target_size: Optional[Tuple[int, int]] = None,
                    show_progress: bool = False) -> List[Dict]:
        """
        批量检测
        
        Args:
            images: 图像列表
            target_size: 目标尺寸
            show_progress: 是否显示进度
            
        Returns:
            检测结果列表
        """
        results = []
        
        for i, image in enumerate(images):
            if show_progress:
                print(f"Processing {i+1}/{len(images)}...")
            
            result = self.detector.detect(image, target_size)
            results.append(result)
        
        return results
    
    def filter_anomalies(self, results: List[Dict], 
                        min_confidence: float = 0.0) -> List[Tuple[int, Dict]]:
        """
        过滤异常结果
        
        Args:
            results: 检测结果列表
            min_confidence: 最小置信度
            
        Returns:
            (索引, 结果) 列表
        """
        return [
            (i, r) for i, r in enumerate(results) 
            if r['has_anomaly'] and r['confidence'] >= min_confidence
        ]
    
    def generate_report(self, results: List[Dict], 
                       image_names: Optional[List[str]] = None) -> Dict:
        """
        生成检测报告
        
        Args:
            results: 检测结果列表
            image_names: 图像名称列表
            
        Returns:
            汇总报告
        """
        total = len(results)
        anomalies = sum(1 for r in results if r['has_anomaly'])
        
        report = {
            'total_images': total,
            'anomaly_count': anomalies,
            'normal_count': total - anomalies,
            'anomaly_rate': anomalies / total if total > 0 else 0.0,
            'average_confidence': np.mean([r['confidence'] for r in results]) if results else 0.0,
            'details': []
        }
        
        for i, result in enumerate(results):
            detail = {
                'index': i,
                'name': image_names[i] if image_names else f"image_{i}",
                'has_anomaly': result['has_anomaly'],
                'confidence': result['confidence'],
                'class': result['class'],
                'box_count': len(result['boxes'])
            }
            report['details'].append(detail)
        
        return report
