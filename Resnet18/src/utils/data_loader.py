import torch
import torchvision
from config.paths import Paths
from .dataset import VideoFrameDataset
import random

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

    def get_root_dir(self, set, modality):
        """
        获取指定数据集和模态的根目录路径。

        Parameters:
        ----------
        set (str): 'train' 、 'val' 或 'test'，指定加载训练集、验证集或测试集。
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。

        Returns:
        -------
        root_dir (str): 指定数据集和模态的根目录路径。
        """
        if set == 'train' or set == 'val':
            if modality == 'rgb':
                return self.rgb_train_dir
            elif modality == 'depth':
                return self.depth_train_dir
            elif modality == 'infrared':
                return self.infrared_train_dir
            else:
                raise ValueError("Unsupported modality: {}".format(modality))
        elif set == 'test':
            if modality == 'rgb':
                return self.rgb_test_dir
            elif modality == 'depth':
                return self.depth_test_dir
            elif modality == 'infrared':
                return self.infrared_test_dir
            else:
                raise ValueError("Unsupported modality: {}".format(modality))
        else:
            raise ValueError("Unsupported set: {}".format(set))

    def get_ids_and_labels(self, set):
        """
        获取训练集、验证集或完整训练集样本的标识符列表及其对应标签。
        特别地，测试集的对应标签返回 None。

        Parameters:
        ----------
        set (str): 'train' 、 'val' 或 'all'，指定加载训练集、验证集或完整训练集的标签。测试集无标签，返回 None。

        Returns:
        -------
        sample_ids (list): 指定数据集的样本编号列表。
        labels (dict): 样本编号到标签的映射字典（sample_id -> label）。
        """
        if set == 'train':
            sample_ids = self.train
        elif set == 'val':
            sample_ids = self.val
        elif set == 'all':
            sample_ids = range(1, 501)
        else:
            return range(1, 201), None # 测试集无标签

        labels = {}
        try:
            with open(self.train_labels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    try:
                        sample_id = int(parts[0])
                        frame_num = int(parts[1])
                        label = int(parts[2])
                    except ValueError:
                        continue

                    # 只保留样本编号在 sample_ids 中的记录
                    if not (sample_id in sample_ids):
                        continue

                    # 若同一样本号出现多次但标签不一致，保留首次出现的标签
                    if sample_id in labels:
                        if labels[sample_id] != label:
                            # 不一致时忽略后续不同标签
                            continue
                    else:
                        labels[sample_id] = label

        except FileNotFoundError:
            print(f"标签文件未找到: {self.train_labels_path}")
        except Exception as e:
            print(f"读取标签文件时出错: {e}")

        return sample_ids, labels
    
    def load_dataiter(self, set='train', modality='rgb', frames_per_clip=128, sampling='uniform', frame_ext='jpg', pad_mode='repeat', 
                         batch_size=8, shuffle=True, num_workers=4):
        """
        加载数据迭代器。

        Parameters:
        ----------
        set (str): 'train' 、 'val' 、'all' 或 'test，指定加载训练集、验证集、完整训练集或测试集。
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。
        frames_per_clip (int): 每个样本返回的帧数。
        sampling (str): 采样策略。
        frame_ext (str): 帧扩展名。
        pad_mode (str): 填充模式。
        batch_size (int): 批量大小。
        shuffle (bool): 是否打乱数据。
        num_workers (int): 数据加载的子进程数。

        Returns:
        -------
        DataLoader: 训练集数据迭代器。
        """
        # 加载样本编号列表和标签（测试集无标签）
        sample_ids, labels = self.get_ids_and_labels(set)
        
        # 依据模态选择根目录
        root_dir = self.get_root_dir(set, modality)

        dataset = VideoFrameDataset(
            root_dir=root_dir,
            sample_ids=sample_ids,
            labels=labels,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
            ]),
            modality=modality,
            frames_per_clip=frames_per_clip,
            sampling=sampling,
            frame_ext=frame_ext,
            pad_mode=pad_mode
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)