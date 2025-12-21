import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from config.paths import Paths

def visualize_depth_quality(depth_image_path, sample_num=10):
    """
    检查深度图质量的函数
    
    参数:
        depth_image_path: 深度图文件夹路径
        sample_num: 随机检查的图片数量
    """
    depth_files = list(Path(depth_image_path).glob("*.png"))[:sample_num]
    
    for i, depth_file in enumerate(depth_files):
        # 读取深度图（假设为16位PNG格式）
        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        
        # 1. 显示原始深度图
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(depth_img, cmap='gray')
        plt.title(f'Original Depth Image {i+1}')
        plt.colorbar()
        
        # 2. 统计无效值
        invalid_mask = (depth_img == 0) | (depth_img == 65535)
        invalid_ratio = np.mean(invalid_mask) * 100
        
        plt.subplot(2, 3, 2)
        plt.imshow(invalid_mask, cmap='Reds')
        plt.title(f'Invalid Pixels: {invalid_ratio:.2f}%')
        
        # 3. 深度值分布直方图
        plt.subplot(2, 3, 3)
        valid_depths = depth_img[depth_img > 0]
        valid_depths = valid_depths[valid_depths < 65535]
        plt.hist(valid_depths.flatten(), bins=50)
        plt.title('Depth Value Distribution')
        plt.xlabel('Depth Value')
        plt.ylabel('Frequency')
        
        # 4. 检测边缘伪影
        artifact_mask, artifact_ratio = detect_edge_artifacts(depth_img)
        
        plt.subplot(2, 3, 4)
        plt.imshow(artifact_mask, cmap='hot')
        plt.title(f'Edge Artifacts: {artifact_ratio:.2f}%')
        
        # 5. 完整性分析
        completeness_info = analyze_depth_completeness(depth_img)
        
        plt.subplot(2, 3, 5)
        plt.text(0.1, 0.8, f'Completeness: {completeness_info["completeness"]:.1%}', fontsize=10)
        plt.text(0.1, 0.6, f'Holes: {completeness_info["num_holes"]}', fontsize=10)
        plt.text(0.1, 0.4, f'Max Hole: {completeness_info["max_hole_size"]}', fontsize=10)
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        processed = preprocess_depth_pipeline(depth_img)
        plt.imshow(processed, cmap='gray')
        plt.title('Final Processed')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细统计信息
        print(f"样本 {i+1}:")
        print(f"  图像尺寸: {depth_img.shape}")
        print(f"  无效像素比例: {invalid_ratio:.2f}%")
        print(f"  边缘伪影比例: {artifact_ratio:.2f}%")
        print(f"  完整性: {completeness_info['completeness']:.1%}")
        print(f"  空洞数量: {completeness_info['num_holes']}")
        print(f"  最大空洞大小: {completeness_info['max_hole_size']}")
        print(f"  有效深度范围: [{valid_depths.min() if len(valid_depths)>0 else 0}, {valid_depths.max() if len(valid_depths)>0 else 0}]")
        print(f"  深度值标准差: {valid_depths.std() if len(valid_depths)>0 else 0:.2f}")
        print("-" * 50)

def median_filter_depth(depth_img, kernel_size=3):
    """
    应用中值滤波去除椒盐噪声
    
    原理: 用邻域中位数替换当前像素值，有效去除孤立噪声点
    参数: kernel_size=3,5,7（奇数），越大平滑效果越强
    """
    # 仅对有效像素进行滤波
    valid_mask = (depth_img > 0) & (depth_img < 65535)
    valid_depth = depth_img.copy()
    
    # 无效区域设为0，滤波后还原
    valid_depth[~valid_mask] = 0
    
    # 应用中值滤波
    filtered = cv2.medianBlur(valid_depth.astype(np.uint16), kernel_size)
    
    # 恢复原始无效区域
    filtered[~valid_mask] = depth_img[~valid_mask]
    
    return filtered

def fill_depth_holes(depth_img, max_hole_size=100):
    """
    填充深度图中的空洞
    
    原理: 基于最近邻有效值填充小空洞
    参数: max_hole_size - 允许填充的最大空洞大小
    """
    # 创建掩码：0表示有效，1表示无效
    invalid_mask = (depth_img == 0) | (depth_img == 65535)
    
    # 使用OpenCV的inpainting方法（推荐Telea算法）
    depth_8bit = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 仅对小空洞进行填充
    if np.sum(invalid_mask) < max_hole_size * max_hole_size:
        # 方法1: 邻域平均填充（适合小空洞）
        from scipy import ndimage
        filled = depth_img.copy()
        
        # 计算距离最近的有效像素
        distances, indices = ndimage.distance_transform_edt(
            ~invalid_mask, return_indices=True
        )
        
        # 用最近有效像素的值填充空洞
        filled[invalid_mask] = depth_img[
            indices[0][invalid_mask], 
            indices[1][invalid_mask]
        ]
        
        # 方法2: 使用OpenCV的inpainting（适合边缘保持）
        # filled = cv2.inpaint(depth_8bit, invalid_mask.astype(np.uint8)*255, 
        #                     inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        return filled
    else:
        print(f"警告: 空洞过大({np.sum(invalid_mask)}像素)，跳过填充")
        return depth_img
    
def normalize_depth(depth_img, valid_range=(500, 4000), to_3channel=True):
    """
    将深度值归一化到0-255范围
    
    参数:
        valid_range: 有效的深度范围(毫米)，超出范围的值将被截断
        to_3channel: 是否转换为3通道（适应ResNet RGB输入）
    """
    # 1. 截断到有效范围
    depth_clipped = np.clip(depth_img, valid_range[0], valid_range[1])
    
    # 2. 归一化到0-255
    depth_normalized = ((depth_clipped - valid_range[0]) / 
                       (valid_range[1] - valid_range[0]) * 255)
    
    # 3. 转换为8位
    depth_8bit = depth_normalized.astype(np.uint8)
    
    if to_3channel:
        # 转换为3通道，适应ResNet输入
        depth_3channel = cv2.merge([depth_8bit, depth_8bit, depth_8bit])
        return depth_3channel
    else:
        return depth_8bit

def preprocess_depth_pipeline(depth_img, 
                             apply_median=True,
                             apply_fill=True,
                             apply_normalize=True,
                             to_3channel=True):
    """
    完整的深度图预处理流水线
    """
    # 1. 中值滤波去噪
    if apply_median:
        depth_img = median_filter_depth(depth_img, kernel_size=3)
    
    # 2. 空洞填充
    if apply_fill:
        depth_img = fill_depth_holes(depth_img, max_hole_size=50)
    
    # 3. 归一化（用于模型输入）
    if apply_normalize:
        # 根据你的深度传感器调整有效范围
        depth_img = normalize_depth(depth_img, valid_range=(500, 4000), to_3channel=to_3channel)
    
    return depth_img

def detect_edge_artifacts(depth_img, edge_threshold=50):
    """
    检测深度图中的边缘伪影
    
    原理: 计算深度梯度，如果梯度过大可能是伪影
    """
    # 只对有效像素计算梯度
    valid_mask = (depth_img > 0) & (depth_img < 65535)
    depth_valid = depth_img.copy()
    depth_valid[~valid_mask] = np.median(depth_valid[valid_mask])
    
    # 计算梯度
    grad_x = cv2.Sobel(depth_valid, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_valid, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 检测高梯度区域（可能为伪影）
    artifact_mask = (gradient_magnitude > edge_threshold) & valid_mask
    
    artifact_ratio = np.mean(artifact_mask)
    
    return artifact_mask, artifact_ratio

def analyze_depth_completeness(depth_img):
    """
    分析深度图的完整性（信息缺失程度）
    """
    total_pixels = depth_img.size
    invalid_pixels = np.sum((depth_img == 0) | (depth_img == 65535))
    completeness = 1.0 - (invalid_pixels / total_pixels)
    
    # 计算连通区域（大空洞）
    invalid_mask = ((depth_img == 0) | (depth_img == 65535)).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(invalid_mask, connectivity=8)
    
    # 排除背景（标签0）
    hole_sizes = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else []
    max_hole_size = max(hole_sizes) if len(hole_sizes) > 0 else 0
    
    return {
        'completeness': completeness,
        'invalid_ratio': invalid_pixels / total_pixels,
        'num_holes': len(hole_sizes),
        'max_hole_size': max_hole_size,
        'avg_hole_size': np.mean(hole_sizes) if len(hole_sizes) > 0 else 0
    }

def depth_dataset_quality_check(dataset_path, num_samples=None, invalid_threshold=0.3, noise_threshold=500, output_every_sample=False):
    """
    系统性检查数据集质量
    
    参数:
        dataset_path: 数据集路径
        num_samples: 检查样本数
        invalid_threshold: 无效像素比例阈值
        noise_threshold: 噪声标准差阈值
        output_every_sample: 是否输出每一个样本的质量评分
    """
    stats = {
        'high_invalid': 0,    # 无效像素>阈值
        'moderate_noise': 0,  # 深度标准差>阈值
        'good_quality': 0,    # 质量良好
        'edge_artifacts': 0   # 边缘伪影（可选扩展）
    }
    
    quality_scores = []

    depth_files = list(Path(dataset_path).glob("*.png"))

    if (num_samples is not None) and (num_samples < len(depth_files)):
        depth_files = depth_files[:num_samples]
    
    for i, depth_path in enumerate(depth_files):    
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # 计算质量指标
        invalid_mask = (depth_img == 0) | (depth_img == 65535)
        invalid_ratio = np.mean(invalid_mask)
        
        valid_depths = depth_img[(depth_img > 0) & (depth_img < 65535)]
        
        if len(valid_depths) > 0:
            depth_std = valid_depths.std()
            depth_mean = valid_depths.mean()
            # 计算信噪比（SNR）作为额外指标
            snr = depth_mean / depth_std if depth_std > 0 else float('inf')
        else:
            depth_std = 0
            snr = 0
        
        # 质量评分（0-1，越高越好）
        quality_score = 1.0 - min(1.0, invalid_ratio * 2 + (depth_std / 1000))
        quality_scores.append(quality_score)
        
        # 分类统计
        if invalid_ratio > invalid_threshold:
            stats['high_invalid'] += 1
            if output_every_sample: 
                print(f"样本{i}: 无效像素过多 ({invalid_ratio:.1%})")
        elif depth_std > noise_threshold and len(valid_depths) > 0:
            stats['moderate_noise'] += 1
            if output_every_sample:
                print(f"样本{i}: 噪声较大 (标准差={depth_std:.0f}, SNR={snr:.1f})")
        else:
            stats['good_quality'] += 1
            if output_every_sample:
                print(f"样本{i}: 质量良好 (SNR={snr:.1f})")
    
    print("=== 数据集质量总结 ===")
    print(f"总检查样本: {len(quality_scores)}")
    print(f"平均质量评分: {np.mean(quality_scores):.3f} (±{np.std(quality_scores):.3f})")
    print(f"质量良好: {stats['good_quality']} ({stats['good_quality']/len(quality_scores):.1%})")
    print(f"高无效值: {stats['high_invalid']} ({stats['high_invalid']/len(quality_scores):.1%})")
    print(f"高噪声: {stats['moderate_noise']} ({stats['moderate_noise']/len(quality_scores):.1%})\n")
    
    return stats, quality_scores, quality_scores

if __name__ == "__main__":
    visualize_depth_quality(os.path.join(Paths.DEPTH_TRAIN_DIR, '1'), sample_num=10)

# if __name__ == "__main__":
#     for i in range(1, 501):
#         print(f"检查视频 {i} 的深度数据质量...")
#         depth_dataset_quality_check(os.path.join(Paths.DEPTH_TRAIN_DIR, str(i)))