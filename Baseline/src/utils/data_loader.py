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
        self.train = [1, 2, 3, 4, 9, 10, 11, 13, 14, 15, 16, 17, 20, 22, 23, 25, 27, 28, 29, 30, 34, 36, 37, 38, 39, 40, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 64, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 90, 92, 93, 94, 97, 100, 102, 103, 104, 105, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 127, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 143, 145, 146, 148, 149, 151, 152, 153, 155, 156, 158, 159, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 180, 181, 182, 183, 184, 187, 188, 189, 191, 192, 194, 196, 197, 198, 199, 200, 201, 202, 203, 206, 208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 223, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 237, 238, 240, 241, 243, 245, 248, 249, 250, 251, 252, 253, 256, 257, 259, 260, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 273, 274, 275, 276, 277, 278, 281, 282, 283, 285, 286, 287, 288, 289, 291, 292, 293, 295, 297, 298, 299, 300, 301, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 316, 317, 318, 319, 321, 322, 323, 326, 327, 329, 330, 331, 332, 334, 336, 337, 338, 339, 340, 341, 343, 344, 345, 348, 350, 351, 353, 354, 355, 356, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 372, 374, 375, 377, 378, 380, 381, 383, 384, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399, 400, 401, 403, 405, 406, 407, 408, 409, 410, 411, 413, 414, 415, 416, 417, 418, 421, 424, 425, 426, 427, 428, 429, 430, 431, 433, 434, 435, 440, 441, 444, 446, 447, 448, 449, 450, 451, 452, 453, 455, 457, 458, 459, 460, 461, 464, 465, 467, 468, 469, 470, 471, 475, 477, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 493, 494, 496, 497, 498, 499, 500]
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