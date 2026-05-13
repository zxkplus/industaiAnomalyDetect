#!/usr/bin/env python3
"""
PatchCore 异常检测 - 完整使用示例
展示正负样本库建立、检测和可视化流程
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加脚本目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.feature_extractor import FeatureExtractor
from scripts.normal_model import NormalModel
from scripts.anomaly_library import AnomalyLibrary
from scripts.anomaly_detector import AnomalyDetector, BatchAnomalyDetector
from scripts.utils import load_image, save_json


class PatchCorePipeline:
    """完整的 PatchCore 异常检测流程"""
    
    def __init__(self, output_dir: str = './output'):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 组件
        self.extractor = None
        self.normal_model = None
        self.anomaly_model = None
        self.detector = None
    
    def initialize(self, model_name: str = 'resnet18', 
                   layers: list = None,
                   sample_size: int = 1000):
        """
        初始化组件
        
        Args:
            model_name: 预训练模型名称
            layers: 特征层列表
            sample_size: 每个样本采样特征点数
        """
        print("=" * 60)
        print("初始化特征提取器...")
        
        if layers is None:
            layers = ['layer2', 'layer3']
        
        self.extractor = FeatureExtractor(
            model_name=model_name,
            layers=layers,
            pretrained=True
        )
        print(f"  模型: {model_name}, 层: {layers}")
        print(f"  特征维度: {self.extractor.feature_dim}")
        print(f"  设备: {self.extractor.device}")
        
        self.normal_model = NormalModel(
            self.extractor,
            sample_size=sample_size,
            random_seed=42
        )
        print("  正常模型初始化完成")
        
        self.anomaly_model = AnomalyLibrary()
        print("  异常库初始化完成")
        
        self.detector = AnomalyDetector(
            self.extractor,
            self.normal_model,
            self.anomaly_model,
            k_neighbors=5,
            score_threshold=0.5,
            nms_threshold=0.5,
            min_box_area=100
        )
        print("  检测器初始化完成")
        print("=" * 60)
    
    def build_normal_model(self, normal_image_paths: list):
        """
        构建正常样本特征库
        
        Args:
            normal_image_paths: 正常样本图像路径列表
        """
        print("\n" + "=" * 60)
        print(f"构建正常样本库: {len(normal_image_paths)} 张图像")
        
        for i, img_path in enumerate(normal_image_paths):
            print(f"  [{i+1}/{len(normal_image_paths)}] 处理: {os.path.basename(img_path)}")
            
            try:
                image = load_image(img_path)
                self.normal_model.add_sample(image)
            except Exception as e:
                print(f"    警告: 处理失败 - {e}")
        
        print("\n构建特征库...")
        self.normal_model.build(normalize=True)
        
        # 保存模型
        model_path = os.path.join(self.output_dir, 'normal_model.pkl')
        self.normal_model.save(model_path)
        print(f"正常模型已保存: {model_path}")
        print("=" * 60)
    
    def register_anomalies(self, anomaly_directory: str = None, 
                          anomaly_data: dict = None,
                          patch_size: int = 32):
        """
        注册异常样本（支持两种方式）
        
        方式1 - 从目录批量注册（推荐）:
            目录中包含异常图片和同名 .txt 标注文件
            标注格式: 每行 "类别 中心x 中心y"
            例如:
                0 393 152
                1 200 300
                scratch 100 200
        
        方式2 - 手动指定（保留兼容）:
            anomaly_data 格式:
            {
                'scratch': [
                    {'image': 'path/to/image.png', 'coords': [[y1,x1,y2,x2], ...]},
                    ...
                ],
            }
        
        Args:
            anomaly_directory: 包含异常图片和标注文件的目录路径
            anomaly_data: 手动指定的异常数据字典（可选）
            patch_size: 异常区域边长，以中心点为基准生成 patch
        """
        print("\n" + "=" * 60)
        
        # 优先使用目录批量注册方式
        if anomaly_directory and os.path.isdir(anomaly_directory):
            print(f"从目录批量注册异常样本: {anomaly_directory}")
            print(f"标注格式: 每行 '类别 中心x 中心y'，图片和 .txt 标注文件同名")
            print(f"patch_size: {patch_size}")
            
            summary = self.anomaly_model.register_from_directory(
                self.extractor,
                anomaly_directory,
                patch_size=patch_size
            )
            
            print(f"\n注册摘要:")
            print(f"  图片数量: {summary['total_images']}")
            print(f"  异常点数: {summary['total_anomalies']}")
            print(f"  类别统计: {dict(summary['categories'])}")
            
            total_registered = summary['total_anomalies']
        
        elif anomaly_data:
            # 兼容旧的手动指定方式
            print(f"手动注册异常样本: {len(anomaly_data)} 个类别")
            
            total_registered = 0
            for category, samples in anomaly_data.items():
                print(f"\n  类别: {category}")
                category_count = 0
                
                for sample in samples:
                    img_path = sample['image']
                    coords = sample['coords']
                    
                    try:
                        image = load_image(img_path)
                        self.anomaly_model.register_from_image(
                            self.extractor, image, coords, category
                        )
                        category_count += 1
                    except Exception as e:
                        print(f"    警告: 处理失败 - {e}")
                
                print(f"    已注册: {category_count} 个样本")
                total_registered += category_count
            
            print(f"\n总计注册: {total_registered} 个异常样本")
        
        else:
            print("未提供异常数据，跳过异常库构建")
            return
        
        # 构建异常库
        if total_registered > 0:
            self.anomaly_model.build(normalize=True)
            
            lib_path = os.path.join(self.output_dir, 'anomaly_library.pkl')
            self.anomaly_model.save(lib_path)
            print(f"异常库已保存: {lib_path}")
        
        print("=" * 60)
    
    def load_models(self, normal_model_path: str, 
                   anomaly_library_path: str = None):
        """
        加载已有模型
        
        Args:
            normal_model_path: 正常模型路径
            anomaly_library_path: 异常库路径(可选)
        """
        print("\n" + "=" * 60)
        print("加载已有模型...")
        
        self.normal_model = NormalModel.load(normal_model_path, self.extractor)
        print(f"  正常模型: {len(self.normal_model)} 个特征点")
        
        if anomaly_library_path and os.path.exists(anomaly_library_path):
            self.anomaly_model = AnomalyLibrary.load(anomaly_library_path)
            print(f"  异常库: {len(self.anomaly_model)} 个特征点")
            print(f"  类别: {self.anomaly_model.get_categories()}")
        
        # 重建检测器
        self.detector = AnomalyDetector(
            self.extractor,
            self.normal_model,
            self.anomaly_model
        )
        print("=" * 60)
    
    def detect(self, image_path: str, save_visualization: bool = True) -> dict:
        """
        检测单张图像
        
        Args:
            image_path: 图像路径
            save_visualization: 是否保存可视化结果
            
        Returns:
            检测结果
        """
        print(f"\n检测图像: {image_path}")
        
        # 执行检测
        result = self.detector.detect(image_path)
        
        # 打印结果
        print(f"  异常: {'是' if result['anomaly'] else '否'}")
        print(f"  类别: {result['class']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  异常区域数: {len(result['boxes'])}")
        
        for i, (box, score) in enumerate(zip(result['boxes'], result['scores'])):
            print(f"    区域{i+1}: 坐标{[round(v) for v in box]}, 分数{score:.4f}")
        
        # 可视化
        if save_visualization:
            self._visualize_result(image_path, result)
        
        return result
    
    def detect_batch(self, image_paths: list, 
                    show_progress: bool = True) -> list:
        """
        批量检测
        
        Args:
            image_paths: 图像路径列表
            show_progress: 是否显示进度
            
        Returns:
            检测结果列表
        """
        print("\n" + "=" * 60)
        print(f"批量检测: {len(image_paths)} 张图像")
        
        batch_detector = BatchAnomalyDetector(self.detector)
        results = batch_detector.detect_batch(
            image_paths, show_progress=show_progress
        )
        
        # 生成报告
        report = batch_detector.generate_report(results)
        
        print("\n" + "-" * 40)
        print("检测报告:")
        print(f"  总数: {report['total_images']}")
        print(f"  异常: {report['anomaly_count']}")
        print(f"  正常: {report['normal_count']}")
        print(f"  异常率: {report['anomaly_rate']*100:.1f}%")
        print(f"  平均置信度: {report['average_confidence']:.4f}")
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'detection_report.json')
        save_json(report, report_path)
        print(f"\n报告已保存: {report_path}")
        
        print("=" * 60)
        return results
    
    def _visualize_result(self, image_path: str, result: dict):
        """可视化检测结果"""
        image = load_image(image_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 异常分数热力图
        im = axes[1].imshow(result['score_map'], cmap='hot')
        axes[1].set_title(f'Anomaly Score Map\n(Max: {result["confidence"]:.3f})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # 检测结果
        axes[2].imshow(image)
        
        if result['boxes']:
            colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(result['boxes'])))
            for i, (box, score) in enumerate(zip(result['boxes'], result['scores'])):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=colors[i], facecolor='none'
                )
                axes[2].add_patch(rect)
                
                # 添加标签
                label = f"{result['class']}\n{score:.2f}"
                axes[2].text(x1, y1-5, label, 
                           fontsize=8, color=colors[i],
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        title = f"Detected {'ANOMALY: ' + result['class'] if result['anomaly'] else 'NORMAL'}"
        axes[2].set_title(title)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存
        output_name = f"result_{Path(image_path).stem}.png"
        output_path = os.path.join(self.output_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  可视化已保存: {output_path}")
    
    def analyze_threshold(self, test_images: list,
                         gt_labels: list = None,
                         thresholds: list = None):
        """
        分析不同阈值下的检测性能
        
        Args:
            test_images: 测试图像列表
            gt_labels: 真实标签(1=异常, 0=正常)
            thresholds: 要测试的阈值列表
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        print("\n" + "=" * 60)
        print("阈值分析")
        
        results_by_threshold = {t: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for t in thresholds}
        
        for thresh in thresholds:
            self.detector.set_threshold(thresh)
            
            for i, img_path in enumerate(test_images):
                result = self.detect(img_path, save_visualization=False)
                pred = 1 if result['anomaly'] else 0
                
                if gt_labels and i < len(gt_labels):
                    gt = gt_labels[i]
                    if pred == 1 and gt == 1:
                        results_by_threshold[thresh]['tp'] += 1
                    elif pred == 1 and gt == 0:
                        results_by_threshold[thresh]['fp'] += 1
                    elif pred == 0 and gt == 0:
                        results_by_threshold[thresh]['tn'] += 1
                    else:
                        results_by_threshold[thresh]['fn'] += 1
        
        # 打印结果
        print("\n阈值    精确率    召回率    F1分数")
        print("-" * 45)
        
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            r = results_by_threshold[thresh]
            tp, fp, tn, fn = r['tp'], r['fp'], r['tn'], r['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f" {thresh:.1f}     {precision:.3f}      {recall:.3f}      {f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        print("-" * 45)
        print(f"最佳阈值: {best_threshold} (F1={best_f1:.3f})")
        print("=" * 60)
        
        return best_threshold


def create_demo_data(output_dir: str):
    """
    创建演示数据目录结构
    
    Args:
        output_dir: 输出目录
    """
    dirs = [
        os.path.join(output_dir, 'normal_samples'),
        os.path.join(output_dir, 'anomaly_samples'),
        os.path.join(output_dir, 'test_samples'),
        os.path.join(output_dir, 'output')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # 创建示例标注文件说明
    sample_annotation = """# 异常标注文件示例
# 格式: 每行 "类别 中心x 中心y"
# 例如:
#   0 393 152
#   1 200 300
#   scratch 100 200
# 支持数字类别(class_0, class_1)或字符串类别(scratch, dent等)
"""
    sample_file = os.path.join(dirs[1], 'README_annotation.txt')
    with open(sample_file, 'w') as f:
        f.write(sample_annotation)
    
    print(f"演示数据目录已创建: {output_dir}")
    print("请将数据文件放入相应目录:")
    print(f"  - {dirs[0]}: 正常样本图像(.png/.jpg)")
    print(f"  - {dirs[1]}: 异常样本图像(.png/.jpg) + 同名标注文件(.txt)")
    print(f"    标注格式: 每行 '类别 中心x 中心y'")
    print(f"  - {dirs[2]}: 测试图像")


def main():
    parser = argparse.ArgumentParser(
        description='PatchCore 异常检测完整流程'
    )
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'train', 'detect', 'analyze'],
                       help='运行模式')
    parser.add_argument('--data_dir', type=str, default='./demo_data',
                       help='数据目录')
    parser.add_argument('--normal_model', type=str, default=None,
                       help='正常模型路径')
    parser.add_argument('--anomaly_library', type=str, default=None,
                       help='异常库路径')
    parser.add_argument('--test_image', type=str, default=None,
                       help='测试图像路径')
    parser.add_argument('--output', type=str, default='./output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建演示数据目录
    create_demo_data(args.data_dir)
    
    # 创建流程实例
    pipeline = PatchCorePipeline(output_dir=args.output)
    
    if args.mode == 'demo':
        # 演示模式：展示完整流程（需要准备数据）
        print("\n" + "=" * 60)
        print("演示模式")
        print("=" * 60)
        print("""
请准备以下数据后使用相应模式:

1. 训练模式 (--mode train):
   - 将正常样本图像放入: demo_data/normal_samples/
   - 将异常样本图像放入: demo_data/anomaly_samples/
   - 为每个异常图片创建同名 .txt 标注文件:
     格式: 每行 "类别 中心x 中心y"
     示例:
       0 393 152
       1 200 300
       scratch 100 200

2. 检测模式 (--mode detect):
   - 加载已有模型进行检测

3. 分析模式 (--mode analyze):
   - 批量测试不同阈值性能
""")
    
    elif args.mode == 'train':
        # 训练模式：建立正负样本库
        pipeline.initialize(
            model_name='resnet18',
            layers=['layer2', 'layer3'],
            sample_size=1000
        )
        
        # 1. 获取并处理正常样本
        normal_dir = os.path.join(args.data_dir, 'normal_samples')
        normal_paths = glob.glob(os.path.join(normal_dir, '*.png'))
        normal_paths += glob.glob(os.path.join(normal_dir, '*.jpeg'))
        
        if normal_paths:
            pipeline.build_normal_model(normal_paths)
        else:
            print("警告: 未找到正常样本图像")
        
        # 2. 从目录批量注册异常样本（自动读取同名 .txt 标注文件）
        anomaly_dir = os.path.join(args.data_dir, 'anomaly_samples')
        
        # 检查目录中是否有图片和标注文件
        anomaly_images = glob.glob(os.path.join(anomaly_dir, '*.png'))
        anomaly_images += glob.glob(os.path.join(anomaly_dir, '*.jpeg'))
        
        if anomaly_images:
            # 检查是否有对应的标注文件
            has_annotations = any(
                os.path.exists(os.path.splitext(f)[0] + '.txt') 
                for f in anomaly_images
            )
            
            if has_annotations:
                pipeline.register_anomalies(
                    anomaly_directory=anomaly_dir,
                    patch_size=32  # 可调整异常区域大小
                )
            else:
                print("\n警告: 异常样本目录中没有找到 .txt 标注文件")
                print("标注文件格式: 每行 '类别 中心x 中心y'，与图片同名")
        
        print("\n训练完成！模型已保存到:", args.output)
    
    elif args.mode == 'detect':
        # 检测模式
        if not argsnormal_model:
            print("错误: 检测模式需要指定 --normal_model")
            return
        
        pipeline.initialize()
        pipeline.load_models(
            argsnormal_model,
            argsanomaly_library
        )
        
        if args.test_image:
            # 检测单张图像
            result = pipeline.detect(args.test_image, save_visualization=True)
        else:
            # 批量检测
            test_dir = os.path.join(args.data_dir, 'test_samples')
            test_paths = glob.glob(os.path.join(test_dir, '*.png'))
            test_paths += glob.glob(os.path.join(test_dir, '*.jpg'))
            
            if test_paths:
                pipeline.detect_batch(test_paths)
            else:
                print("错误: 未找到测试图像")
    
    elif args.mode == 'analyze':
        # 分析模式：测试不同阈值
        if not argsnormal_model:
            print("错误: 分析模式需要指定 --normal_model")
            return
        
        pipeline.initialize()
        pipeline.load_models(
            argsnormal_model,
            argsanomaly_library
        )
        
        test_dir = os.path.join(args.data_dir, 'test_samples')
        test_paths = glob.glob(os.path.join(test_dir, '*.png'))
        test_paths += glob.glob(os.path.join(test_dir, '*.jpg'))
        
        if test_paths:
            # 假设前一半是异常的（仅演示用）
            gt_labels = [1 if i < len(test_paths)//2 else 0 for i in range(len(test_paths))]
            pipeline.analyze_threshold(test_paths, gt_labels)
        else:
            print("错误: 未找到测试图像")


if __name__ == '__main__':
    main()
