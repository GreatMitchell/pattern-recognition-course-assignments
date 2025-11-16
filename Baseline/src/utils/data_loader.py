import os
import random
from config.paths import Paths
from PIL import Image
import numpy as np

class DataLoader:
    def __init__(self):
        # 设置训练集数据路径
        self.rgb_train_dir = Paths.RGB_TRAIN_DIR
        self.depth_train_dir = Paths.DEPTH_TRAIN_DIR 
        self.infrared_train_dir = Paths.INFRARED_TRAIN_DIR
        self.train_labels_path = Paths.TRAIN_LABELS_PATH
        # 随机将样本分为训练集（375个）和验证集（125个）
        self.train = sorted(random.sample(range(1, 501), 375))
        self.val = [i for i in range(1, 501) if i not in self.train]
        # 设置测试集数据路径
        self.rgb_test_dir = Paths.RGB_TEST_DIR
        self.depth_test_dir = Paths.DEPTH_TEST_DIR
        self.infrared_test_dir = Paths.INFRARED_TEST_DIR
        self.test_labels_path = Paths.TEST_LABELS_PATH
    
    def get_train_sample_list(self, modality):
        """
        获取训练集样本的标识符列表。训练集样本取随机选择的375个样本。

        Parameters:
        ----------
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。

        Returns:
        -------
        sample_list (list): 所有样本的标识符列表。
        """
        if modality == 'rgb':
            modality_dir = self.rgb_train_dir
            # print("Samples that are used for training have those numbers: ", self.train)
        elif modality == 'depth':
            modality_dir = self.depth_train_dir
        elif modality == 'infrared':
            modality_dir = self.infrared_train_dir
        else:
            raise ValueError("Unsupported modality: {}".format(modality))

        sample_list = [os.path.join(modality_dir, str(f)) for f in self.train]
        return sample_list
    
    def get_val_sample_list(self, modality):
        """
        获取验证集样本的标识符列表。验证集样本取随机选择的125个样本。

        Parameters:
        ----------
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。

        Returns:
        -------
        sample_list (list): 所有样本的标识符列表。
        """
        if modality == 'rgb':
            modality_dir = self.rgb_train_dir
        elif modality == 'depth':
            modality_dir = self.depth_train_dir
        elif modality == 'infrared':
            modality_dir = self.infrared_train_dir
        else:
            raise ValueError("Unsupported modality: {}".format(modality))

        sample_list = [os.path.join(modality_dir, str(f)) for f in self.val]
        return sample_list
    
    def get_all_sample_list(self, modality):
        """
        获取所有样本的标识符列表。样本默认取1到500号样本。

        Parameters:
        ----------
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。

        Returns:
        -------
        sample_list (list): 所有样本的标识符列表。
        """
        if modality == 'rgb':
            modality_dir = self.rgb_train_dir
        elif modality == 'depth':
            modality_dir = self.depth_train_dir
        elif modality == 'infrared':
            modality_dir = self.infrared_train_dir
        else:
            raise ValueError("Unsupported modality: {}".format(modality))

        sample_list = [os.path.join(modality_dir, str(f)) for f in range(1, 501)]
        return sample_list
    
    def get_test_sample_list(self, modality):
        """
        获取测试集样本的标识符列表。

        Parameters:
        ----------
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。

        Returns:
        -------
        sample_list (list): 所有样本的标识符列表。
        """
        if modality == 'rgb':
            modality_dir = self.rgb_test_dir
        elif modality == 'depth':
            modality_dir = self.depth_test_dir
        elif modality == 'infrared':
            modality_dir = self.infrared_test_dir
        else:
            raise ValueError("Unsupported modality: {}".format(modality))

        sample_list = [os.path.join(modality_dir, str(f)) for f in range(1, 201)]
        return sample_list
    
    def get_frame_paths(self, sample_id, modality):
        """
        获取单个样本的所有帧路径。

        Parameters:
        ----------
        sample_id (str): 样本标识符（视频序列号）。
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。

        Returns:
        -------
        frame_paths (list): 该样本的所有帧路径列表。
        """
        if modality == 'rgb':
            sample_dir = os.path.join(self.rgb_train_dir, sample_id)
        elif modality == 'depth':
            sample_dir = os.path.join(self.depth_train_dir, sample_id)
        elif modality == 'infrared':
            sample_dir = os.path.join(self.infrared_train_dir, sample_id)
        else:
            raise ValueError("Unsupported modality: {}".format(modality))
        
        frame_files = sorted(os.listdir(sample_dir))
        frame_paths = [os.path.join(sample_dir, f) for f in frame_files]
        return frame_paths

    def load_single_frame(self, frame_path, modality):
        """
        加载单帧数据的模板函数。
        
        Parameters:
        ----------
        frame_path (str): 帧图像的文件路径。
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。

        Returns:
        -------
        image (ndarray): 加载的2D图像数据。

        """
        if not os.path.isfile(frame_path):
            raise FileNotFoundError("Frame not found: {}".format(frame_path))

        # 使用 Pillow 打开图像并根据模态做稳健转换
        with Image.open(frame_path) as img:
            # RGB：强制为三通道 uint8
            if modality == 'rgb':
                img = img.convert('RGB')
                arr = np.array(img, dtype=np.uint8)

            # Depth：优先处理 16-bit/32-bit 深度，否则转为 8-bit 并归一化到 0-1
            elif modality == 'depth':
                if img.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
                    # 可能是 16-bit 或整型模式
                    arr = np.array(img).astype(np.float32)
                    # 使用最大值进行归一化以保持相对深度比例（避免除以0）
                    maxv = arr.max() if arr.max() > 0 else 1.0
                    arr = arr / maxv
                else:
                    # 其他模式统一转为 8-bit 灰度，再归一化到 0-1
                    img = img.convert('L')
                    arr = np.array(img).astype(np.float32) / 255.0

            # Infrared：强制单通道灰度，返回 float32（不归一化，按需要可归一化）
            elif modality == 'infrared':
                # 无论原来是 RGB 还是多通道，转换为单通道灰度
                img = img.convert('L')
                arr = np.array(img).astype(np.float32)

            else:
                raise ValueError("Unsupported modality: {}".format(modality))

        # 返回处理后的 numpy 数组
        return arr

    def get_train_labels(self):
        """
        获取训练集样本的标签列表。

        Returns:
        -------
        labels (list): 训练集样本的标签列表。
        """
        labels = []
        
        try:
            with open(self.train_labels_path, 'r') as f:
                for line in f:
                    # 分割每行的三列数据
                    parts = line.strip().split()
                    
                    # 确保行有三列数据
                    if len(parts) < 3:
                        continue
                    
                    # 解析样本编号、帧数和标签
                    sample_id = int(parts[0])
                    frame_num = int(parts[1])
                    label = int(parts[2])  # 根据实际情况，可能需要转换为int
                    
                    # 只保留训练集样本
                    if sample_id in self.train:
                        labels.append(label)
        
        except FileNotFoundError:
            print(f"标签文件未找到: {self.train_labels_path}")
            return []
        except Exception as e:
            print(f"读取标签文件时出错: {e}")
            return []
        
        return labels

    def get_val_labels(self):
        """
        获取验证集样本的标签列表。

        Returns:
        -------
        labels (list): 验证集样本的标签列表。
        """
        labels = []
        
        try:
            with open(self.train_labels_path, 'r') as f:
                for line in f:
                    # 分割每行的三列数据
                    parts = line.strip().split()
                    
                    # 确保行有三列数据
                    if len(parts) < 3:
                        continue
                    
                    # 解析样本编号、帧数和标签
                    sample_id = int(parts[0])
                    frame_num = int(parts[1])
                    label = int(parts[2])  # 根据实际情况，可能需要转换为int
                    
                    # 只保留验证集样本
                    if sample_id in self.val:
                        labels.append(label)
        
        except FileNotFoundError:
            print(f"标签文件未找到: {self.train_labels_path}")
            return []
        except Exception as e:
            print(f"读取标签文件时出错: {e}")
            return []
        
        return labels
    
    def get_all_labels(self):
        """
        获取所有样本的标签列表。

        Returns:
        -------
        labels (list): 所有样本的标签列表。
        """
        labels = []
        
        try:
            with open(self.train_labels_path, 'r') as f:
                for line in f:
                    # 分割每行的三列数据
                    parts = line.strip().split()
                    
                    # 确保行有三列数据
                    if len(parts) < 3:
                        continue
                    
                    # 解析样本编号、帧数和标签
                    sample_id = int(parts[0])
                    frame_num = int(parts[1])
                    label = int(parts[2])  # 根据实际情况，可能需要转换为int
                    
                    # 保留所有样本（编号1-500）
                    if 1 <= sample_id <= 500:
                        labels.append(label)
        
        except FileNotFoundError:
            print(f"标签文件未找到: {self.train_labels_path}")
            return []
        except Exception as e:
            print(f"读取标签文件时出错: {e}")
            return []
        
        return labels