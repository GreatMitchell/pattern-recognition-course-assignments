import numpy as np
from skimage import feature
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import itertools
from utils.data_loader import DataLoader

class FeatureExtractor:
    def __init__(self, data_loader=DataLoader(), aggregation_method='mean'):
        """
        初始化特征提取器。
        Parameters:
        ----------
        data_loader (DataLoader): 数据加载器实例。
        aggregation_method (str): 帧特征聚合方法，支持 'mean'、 'max'、 'stat_concat'。
        """
        self.data_loader = data_loader
        self.RGB = 'rgb'
        self.DEPTH = 'depth'
        self.INFRARED = 'infrared'
        self.samples_list = {
            'train':  {
                self.RGB: self.data_loader.get_train_sample_list(self.RGB),
                self.DEPTH: self.data_loader.get_train_sample_list(self.DEPTH),
                self.INFRARED: self.data_loader.get_train_sample_list(self.INFRARED)
            },
            'val': {
                self.RGB: self.data_loader.get_val_sample_list(self.RGB),
                self.DEPTH: self.data_loader.get_val_sample_list(self.DEPTH),
                self.INFRARED: self.data_loader.get_val_sample_list(self.INFRARED)
            },
            'all': {
                self.RGB: self.data_loader.get_all_sample_list(self.RGB),
                self.DEPTH: self.data_loader.get_all_sample_list(self.DEPTH),
                self.INFRARED: self.data_loader.get_all_sample_list(self.INFRARED)
            },
            'test': {
                self.RGB: self.data_loader.get_test_sample_list(self.RGB),
                self.DEPTH: self.data_loader.get_test_sample_list(self.DEPTH),
                self.INFRARED: self.data_loader.get_test_sample_list(self.INFRARED)
            }
        }
        self.aggregation_method = aggregation_method


    def separate_fusion(self, set_type='train'):
        """
        分别提取各模态的特征。
        Parameters:
        ----------
        set_type (str): 数据集类型，'train'、'val'、'all'或'test'。

        Returns:
        -------
        rgb_features_list (ndarray): RGB模态的特征列表。
        depth_features_list (ndarray): 深度模态的特征列表。
        infrared_features_list (ndarray): 红外模态的特征列表。

        """
        samples_list = self.samples_list[set_type]

        rgb_features_list = self.extract_features_parallel(samples_list[self.RGB], 
                                                                self.RGB)
        depth_features_list = self.extract_features_parallel(samples_list[self.DEPTH], 
                                                                      self.DEPTH)
        infrared_features_list = self.extract_features_parallel(samples_list[self.INFRARED], 
                                                                           self.INFRARED)
        
        return np.array(rgb_features_list), np.array(depth_features_list), np.array(infrared_features_list)

    def early_fusion(self, set_type='train'):
        """
        提取所有模态的特征并进行早期融合。

        Parameters:
        ----------
        set_type (str): 数据集类型，'train'、'val'、'all'或'test'。

        Returns:
        -------
        fused_features (ndarray): 融合后的特征向量。

        """
        samples_list = self.samples_list[set_type]

        # 分别提取各模态特征
        rgb_features_list = self.extract_features_parallel(samples_list[self.RGB], 
                                                                        self.RGB)
        depth_features_list = self.extract_features_parallel(samples_list[self.DEPTH], 
                                                                          self.DEPTH)
        infrared_features_list = self.extract_features_parallel(samples_list[self.INFRARED], 
                                                                             self.INFRARED)
        
        
        return np.concatenate([rgb_features_list, depth_features_list, infrared_features_list], axis=1)

    def extract_features_parallel(self, sample_list, modality, frames_per_sample=32, n_workers=None):
        """
        并行提取特征。

        Parameters:
        ----------
        sample_list (list): 样本路径列表。
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。
        frames_per_sample (int): 每个样本采样的帧数。
        n_workers (int): 并行工作进程/线程数。

        Returns:
        -------
        results (list): 每个样本的聚合特征列表，每个元素是一个ndarray，对应一个样本。

        """
        if n_workers is None:
            n_workers = mp.cpu_count() - 1  # 留一个核心给系统
        
        # 方法1: 进程池（CPU密集型任务）
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(self.process_single_sample, sample_list, itertools.repeat(modality), itertools.repeat(frames_per_sample)))
        
        # 方法2: 线程池（I/O密集型任务）
        # with ThreadPoolExecutor(max_workers=n_workers) as executor:
        #     results = list(executor.map(process_single_sample, sample_list, itertools.repeat(modality), itertools.repeat(frames_per_sample)))
        
        return results

    def process_single_sample(self, sample_path, modality, frames_per_sample=32):
        """
        处理单个样本的入口函数，用于并行化。

        Parameters:
        ----------
        sample_path (str): 样本路径。
        frames_per_sample (int): 每个样本采样的帧数。

        Returns:
        -------
        aggregated_features (ndarray): 聚合后的单个样本特征。
        
        """
        # 1. 加载样本的所有帧路径
        frame_paths = self.data_loader.get_frame_paths(sample_path, modality)

        # 2. 采样关键帧（而不是处理所有帧）
        sampled_frames = self.sample_key_frames(frame_paths, n=frames_per_sample)

        # 3. 处理采样帧
        sample_features = []
        for frame_path in sampled_frames:
            frame = self.data_loader.load_single_frame(frame_path, modality)
            features = self.extract_combined_features(frame, modality=modality)
            sample_features.append(features)
        
        # 4. 聚合帧特征为样本特征（而不是保留所有帧）
        return self.aggregate_frame_features(sample_features)

    def sample_key_frames(self, frame_paths, n=32):
        """
        均匀采样关键帧。

        Parameters:
        ----------
        frame_paths (list): 样本的所有帧路径列表。
        n (int): 需要采样的关键帧数量。

        Returns:
        -------
        sampled_frames (list): 采样得到的关键帧路径列表。
        
        """
        total_frames = len(frame_paths)
        if total_frames <= n:
            return frame_paths
        
        step = total_frames // n
        indices = [i * step for i in range(n)]
        return [frame_paths[i] for i in indices]

    def extract_combined_features(self, image, modality):
        """
        提取图像的组合特征，包括HOG、SIFT和LBP特征。

        Parameters:
        ----------
        image (ndarray): 输入的2D图像数据。
        modality (str): 图像的模态类型（'RGB', 'Depth', 'Infrared'）。

        Returns:
        -------
        combined_features (ndarray): 组合后的1D特征向量。
        """
        # 根据模态选择特征提取方法
        if modality == 'rgb':
            features = self.extract_hog(image)
        elif modality == 'depth':
            features = self.extract_lbp(image)
        elif modality == 'infrared':
            features = self.extract_lbp(image)
        return features

    def extract_hog(self, image):
        """
        提取可见光图像的HOG特征。
        
        Parameters:
        ----------
        image (ndarray): 输入的2D图像数据。

        Returns:
        -------
        hog_features (ndarray): 提取的1D HOG特征向量。
        """
        # 先用随机矩阵测试函数逻辑
        if image is None:
            # 测试时用的虚拟图像
            test_img = np.random.rand(64, 128) * 255
        else:
            test_img = image
            
        hog_features = feature.hog(test_img, pixels_per_cell=(8, 8), 
                                    cells_per_block=(2, 2), visualize=False, channel_axis=2)
        return hog_features
        
    def extract_sift(self, image):
        """
        提取可见光图像的SIFT特征。
        
        Parameters:
        ----------
        image (ndarray): 输入的2D图像数据。

        Returns:
        -------
        sift_descriptors (ndarray): 提取的SIFT特征描述符，其已展开成为1D向量。
        """

        # 先用随机矩阵测试函数逻辑
        if image is None:
            test_img = (np.random.rand(256, 256) * 255).astype(np.uint8)
        else:
            test_img = image.astype(np.uint8)
        
        sift = feature.SIFT()
        sift.detect_and_extract(test_img)
        descriptors = sift.descriptors
        
        return descriptors.flatten()
        
    def extract_lbp(self, image, P=8, R=1.0):
        """
        提取红外图像和深度图像的LBP特征。

        Parameters:
        ----------
        image (ndarray): 输入的2D灰度图像数据。
        P (int): LBP算法中的邻域点数。
        R (float): LBP算法中的半径。

        Returns:
        -------
        lbp_hist (ndarray): 提取的1D LBP直方图特征向量。
        
        """
        if image is None:
            test_img = np.random.rand(64, 64) * 255
        else:
            test_img = image
            
        lbp = feature.local_binary_pattern(test_img, P, R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=P+2, range=(0, P+2))
        return hist.astype(float)

    def aggregate_frame_features(self, frame_features_list):
        """
        将多帧特征聚合成单个样本特征。

        Parameters:
        ----------
        frame_features_list (list): 多帧特征的列表。

        Returns:
        -------
        aggregated_features (ndarray): 聚合后的单个样本特征。
        
        """
        if self.aggregation_method == 'mean':
            # 方法1: 取平均值
            return np.mean(frame_features_list, axis=0)
        elif self.aggregation_method == 'max':
            # 方法2: 取最大值
            return np.max(frame_features_list, axis=0)
        elif self.aggregation_method == 'stat_concat':
            # 方法3: 拼接统计特征   
            feature_matrix = np.array(frame_features_list)
            return np.concatenate([
                np.mean(feature_matrix, axis=0),
                np.std(feature_matrix, axis=0),
                np.max(feature_matrix, axis=0)
            ])
        else:
            raise ValueError("Unsupported aggregation method: {}".format(self.aggregation_method))