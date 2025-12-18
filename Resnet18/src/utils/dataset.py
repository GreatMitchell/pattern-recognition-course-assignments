import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F

class VideoFrameDataset(Dataset):
    """
    Dataset for videos stored as frame folders:
      root_dir/<sample_id>/1.png, 2.png, ...
    Args:
      root_dir (str): 根目录，例如 self.rgb_train_dir
      sample_ids (list[int|str]): 要加载的样本 id 列表（例如 range(1, 501)）
      labels (dict or None): 可选，{sample_id: label_int}
      transform (callable or None): 应用到每帧的 transform（PIL -> Tensor + Normalize）
      modality ('rgb'|'flow'): 数据模态
      frames_per_clip (int or None): 如果为 int，则每个样本返回固定帧数（采样/填充）；None 则返回全部帧
      sampling ('uniform'|'random'|'all'): 采样策略；'all' 忽略 frames_per_clip
      frame_ext (str): 帧扩展名，默认 'png'
      pad_mode ('repeat'|'zeros'): 当帧数不足时填充策略
    Returns per item:
      dict with keys: 'frames' -> Tensor (T, C, H, W), 'label' -> int or -1, 'length' -> actual frame count
    """
    def __init__(self,
                 root_dir,
                 sample_ids,
                 labels=None,
                 transform=None,
                 modality='rgb',
                 repeat_channels=False,
                 frames_per_clip=128,
                 sampling='uniform',
                 frame_ext='png',
                 pad_mode='repeat'):
        self.root_dir = root_dir
        self.sample_ids = list(sample_ids)
        self.labels = labels if labels is None else {str(k): int(v) for k,v in labels.items()}
        self.transform = transform
        self.modality = modality
        self.repeat_channels = repeat_channels
        self.frames_per_clip = frames_per_clip
        self.sampling = sampling
        self.frame_ext = frame_ext.lower()
        self.pad_mode = pad_mode

    def __len__(self):
        return len(self.sample_ids)

    def _sorted_frame_paths(self, sample_id):
        sample_dir = os.path.join(self.root_dir, str(sample_id))
        if not os.path.isdir(sample_dir):
            raise FileNotFoundError(f"Sample dir not found: {sample_dir}")
        files = [f for f in os.listdir(sample_dir) if f.lower().endswith('.' + self.frame_ext)]
        if not files:
            raise FileNotFoundError(f"No frame files in {sample_dir} with ext {self.frame_ext}")
        # 按文件名数字排序：假设文件名为 "1.png", "2.png"
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        return [os.path.join(sample_dir, f) for f in files]

    def _sample_indices(self, n_frames):
        if self.sampling == 'all' or self.frames_per_clip is None:
            return list(range(n_frames))
        k = self.frames_per_clip
        if n_frames == 0:
            return []
        if n_frames >= k:
            if self.sampling == 'uniform':
                return np.linspace(0, n_frames - 1, k, dtype=int).tolist()
            elif self.sampling == 'random':
                idx = np.sort(np.random.choice(n_frames, k, replace=False))
                return idx.tolist()
            else:
                # fallback to uniform
                return np.linspace(0, n_frames - 1, k, dtype=int).tolist()
        else:
            # n_frames < k：需要填充，先返回全部索引，填充在加载阶段处理
            return list(range(n_frames))

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        frame_paths = self._sorted_frame_paths(sample_id)
        n = len(frame_paths)
        indices = self._sample_indices(n)

        # 读取帧
        frames = []
        for i in indices:
            p = frame_paths[i]
            with Image.open(p) as img:
                if self.modality == 'rgb':
                    img = img.convert('RGB')
                    img_t = self.transform(img) if self.transform else T.ToTensor()(img)

                elif self.modality == 'infrared' or self.modality == 'depth':
                    # 强制灰度，归一化到 [0,1]
                    img = img.convert('L')
                    img_t = self.transform(img) if self.transform else T.ToTensor()(img)
                    if self.repeat_channels:
                        img_t = img_t.repeat(3, 1, 1)  # -> 3xHxW
                else:
                    raise ValueError("Unsupported modality")
            frames.append(img_t)

        # 如果需要固定长度且原始帧不足，根据 pad_mode 填充
        if self.frames_per_clip is not None and self.sampling != 'all':
            k = self.frames_per_clip
            if len(frames) < k:
                if self.pad_mode == 'repeat' and len(frames) > 0:
                    last = frames[-1]
                    while len(frames) < k:
                        frames.append(last.clone())
                else:
                    # zeros 填充
                    C, H, W = frames[0].shape if frames else (3, 224, 224)
                    pad_tensor = torch.zeros((C, H, W), dtype=frames[0].dtype) if frames else torch.zeros((3,224,224))
                    while len(frames) < k:
                        frames.append(pad_tensor)

        # stack -> (T, C, H, W)
        frames_tensor = torch.stack(frames) if frames else torch.empty(0)

        label = -1
        key = str(sample_id)
        if self.labels is not None:
            label = self.labels.get(key, -1)

        return {
            'frames': frames_tensor,
            'label': label,
            'length': min(n, self.frames_per_clip) if (self.frames_per_clip is not None and self.sampling != 'all') else n
        }