import math
import torch
import torchvision
from torch.utils.data import DataLoader as TorchDataLoader
from config.paths import Paths
from .dataset import VideoFrameDataset
import random

class MultiModalDataset(torch.utils.data.Dataset):
    """
    Wrap three VideoFrameDataset instances returning aligned multi-modal items.
    每个 __getitem__ 返回 dict: {'rgb': Tensor(T,C,H,W), 'depth': ..., 'infrared': ..., 'label': int, 'length': int, 'id': int}
    注意：假设 sample_ids 在三个模态下是一致并且 VideoFrameDataset 的 sample_ids 顺序相同。
    """
    def __init__(self, rgb_dataset, depth_dataset, infrared_dataset):
        self.rgb_ds = rgb_dataset
        self.depth_ds = depth_dataset
        self.ir_ds = infrared_dataset
        assert len(self.rgb_ds) == len(self.depth_ds) == len(self.ir_ds)
        self._len = len(self.rgb_ds)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        item_r = self.rgb_ds[idx]
        item_d = self.depth_ds[idx]
        item_ir = self.ir_ds[idx]
        # 校验 sample id alignment 可选（如果你保存了 id）
        # 返回组合 item
        return {
            'rgb': item_r['frames'],
            'depth': item_d['frames'],
            'infrared': item_ir['frames'],
            'label': item_r['label'],        # 假设 label 一致
            'length': item_r['length'],      # 假设长度一致（采样策略相同）
            'id': self.rgb_ds.sample_ids[idx]
        }

def video_multimodal_collate_fn(batch):
    """
    batch: list of items as returned by MultiModalDataset.__getitem__
    Returns:
      rgb: Tensor(B, T, C, H, W)
      depth: Tensor(B, T, C, H, W)
      infrared: Tensor(B, T, C, H, W)
      lengths: Tensor(B,)
      labels: Tensor(B,)
      id: list of sample ids
    """
    import torch
    # reuse earlier collate logic but for all three modalities
    lengths = [item['length'] for item in batch]
    max_len = max(lengths) if lengths else 0
    B = len(batch)

    # find first non-empty frames to get C,H,W
    def find_shape(key):
        for item in batch:
            t = item[key]
            if t.numel() > 0:
                return t.shape[1:]  # (C,H,W) because stored as (T,C,H,W)
        return (3,224,224)

    C_rgb, H, W = find_shape('rgb')
    C_depth, _, _ = find_shape('depth')
    C_ir, _, _ = find_shape('infrared')

    rgb_out = torch.zeros((B, max_len, C_rgb, H, W), dtype=batch[0]['rgb'].dtype)
    depth_out = torch.zeros((B, max_len, C_depth, H, W), dtype=batch[0]['depth'].dtype)
    ir_out = torch.zeros((B, max_len, C_ir, H, W), dtype=batch[0]['infrared'].dtype)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        r = item['rgb']
        d = item['depth']
        ir = item['infrared']
        L = item['length']
        if L > 0:
            rgb_out[i, :L] = r[:L]
            depth_out[i, :L] = d[:L]
            ir_out[i, :L] = ir[:L]
    return {'rgb': rgb_out, 'depth': depth_out, 'infrared': ir_out, 'lengths': torch.tensor(lengths, dtype=torch.long), 'labels': labels, 'ids': [item['id'] for item in batch]}


class DataLoader:
    def __init__(self, num_samples=500):
        """
        __init__ ：初始化`Dataloader`
        
        :param num_samples: 若指定，则将使用前`num_samples`个样本。
        """

        # 设置训练集数据路径
        self.rgb_train_dir = Paths.RGB_TRAIN_DIR
        self.depth_train_dir = Paths.DEPTH_TRAIN_DIR 
        self.infrared_train_dir = Paths.INFRARED_TRAIN_DIR
        self.train_labels_path = Paths.TRAIN_LABELS_PATH

        # 随机将样本分为训练集（75%）和验证集（25%）
        num_train = math.ceil(num_samples * 0.75)
        self.train = sorted(random.sample(range(1, num_samples + 1), num_train))
        self.val = [i for i in range(1, num_samples + 1) if i not in self.train]

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
        set (str): 'train' 、 'val' 、'all' 或 'test'，指定加载训练集、验证集、完整训练集或测试集。
        modality (str): 图像的模态类型（'rgb', 'depth', 'infrared'）。

        Returns:
        -------
        root_dir (str): 指定数据集和模态的根目录路径。
        """
        if set in ['train', 'val', 'all']:
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

                    labels[sample_id] = label

        except FileNotFoundError:
            print(f"标签文件未找到: {self.train_labels_path}")
        except Exception as e:
            print(f"读取标签文件时出错: {e}")

        return sample_ids, labels
    
    def load_multi_modal_dataiter(self, set='train', frames_per_clip=128, sampling='uniform', frame_ext=['jpg', 'png', 'jpg'],
                              pad_mode='repeat', batch_size=4, shuffle=True, num_workers=4):
        '''
        加载多模态数据迭代器。

        Parameters:
        ----------
        set (str): 'train' 、 'val' 、'all' 或 'test，指定加载训练集、验证集、完整训练集或测试集。
        frames_per_clip (int): 每个样本返回的帧数。
        sampling (str): 采样策略。
        frame_ext (list of str): 三个模态的帧扩展名列表。
        pad_mode (str): 填充模式。
        batch_size (int): 批量大小。
        shuffle (bool): 是否打乱数据。
        num_workers (int): 数据加载的子进程数。
        
        Returns:
        -------
        DataLoader: 多模态数据迭代器。
        迭代器每个 batch 返回 dict:
            'rgb': Tensor(B, T, C, H, W)
            'depth': Tensor(B, T, C, H, W)
            'infrared': Tensor(B, T, C, H, W)
            'lengths': Tensor(B,)
            'labels': Tensor(B,)
            'ids': list of sample ids
        其中 T 由 frames_per_clip 和采样策略决定。
        例如，frames_per_clip=16, sampling='uniform' 则 T=16.
        '''
        sample_ids, labels = self.get_ids_and_labels(set)
        # 为每个模态构造 VideoFrameDataset（注意 transform）
        transform_rgb = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        transform_gray = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            # 这里 normalize 可改为单通道 mean/std 或占位 0.5/0.5
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        rgb_ds = VideoFrameDataset(root_dir=self.get_root_dir(set,'rgb'), sample_ids=sample_ids, labels=labels,
                                transform=transform_rgb, frames_per_clip=frames_per_clip, sampling=sampling,
                                frame_ext=frame_ext[0], pad_mode=pad_mode, modality='rgb')
        depth_ds = VideoFrameDataset(root_dir=self.get_root_dir(set,'depth'), sample_ids=sample_ids, labels=labels,
                                transform=transform_gray, frames_per_clip=frames_per_clip, sampling=sampling,
                                frame_ext=frame_ext[1], pad_mode=pad_mode, modality='depth')
        ir_ds = VideoFrameDataset(root_dir=self.get_root_dir(set,'infrared'), sample_ids=sample_ids, labels=labels,
                                transform=transform_gray, frames_per_clip=frames_per_clip, sampling=sampling,
                                frame_ext=frame_ext[2], pad_mode=pad_mode, modality='infrared')

        multimodal_ds = MultiModalDataset(rgb_ds, depth_ds, ir_ds)
        return TorchDataLoader(multimodal_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=video_multimodal_collate_fn)