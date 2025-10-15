import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random
identity = lambda x: x
from torch.utils.data import Dataset

from . import video_transforms, volume_transforms
from .loader import get_image_loader, get_video_loader
from .random_erasing import RandomErasing

# (Arg2, spatial_sampling, tensor_normalize)

class Arg2():
    def __init__(self):
        self.aa='rand-m7-n4-mstd0.5-inc1'
        self.train_interpolation = 'bicubic'
        self.num_sample = 2
        self.input_size = 224
        self.data_set = 'k400'
        self.reprob = 0.25
        self.remode = 'pixel'
        self.recount = 1

def spatial_sampling(frames, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=224, random_horizontal_flip=True, inverse_uniform_sampling=False, aspect_ratio=None, scale=None, motion_shift=False):
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(images=frames, min_size=min_scale, max_size=max_scale, inverse_uniform_sampling=inverse_uniform_sampling)
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (video_transforms.random_resized_crop_with_shift if motion_shift else video_transforms.random_resized_crop)
            frames = transform_func(images=frames, target_height=crop_size, target_width=crop_size, scale=scale, ratio=aspect_ratio)
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(frames, min_scale, max_scale)
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames

def tensor_normalize(tensor, mean, std):
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list: mean = torch.tensor(mean)
    if type(std) == list: std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

# --- LỚP VideoClsDataset ĐÃ ĐƯỢC SỬA LỖI ---

class VideoClsDataset():
    def __init__(self, image_size, samples, anno_path='', data_root='', mode='train', clip_len=1, frame_sample_rate=2, crop_size=224, short_side_size=256, new_height=256, new_width=340, keep_aspect_ratio=True, num_segment=16, num_crop=1, test_num_segment=10, test_num_crop=3, sparse_sample=False):
        self.image_size = image_size; self.anno_path = anno_path; self.samples = samples; self.data_root = data_root; self.mode = mode; self.clip_len = clip_len; self.frame_sample_rate = frame_sample_rate; self.crop_size = crop_size; self.short_side_size = short_side_size; self.new_height = new_height; self.new_width = new_width; self.keep_aspect_ratio = keep_aspect_ratio; self.num_segment = num_segment; self.test_num_segment = test_num_segment; self.num_crop = num_crop; self.test_num_crop = test_num_crop; self.sparse_sample = sparse_sample; self.aug = False; self.rand_erase = False
        if self.mode in ['train']: self.aug = True
        self.rand_erase = True
        self.video_loader = get_video_loader()

    def getitem(self):
        args = Arg2(); args.input_size = self.image_size; scale_t = 1; sample = self.samples
        buffer = self.load_video(sample, sample_rate_scale=scale_t)
        
        # === SỬA LỖI VÒNG LẶP VÔ HẠN TẠI ĐÂY ===
        if len(buffer) == 0:
            raise IOError(f"Could not load video buffer for {sample}")
        
        new_frames = self._aug_frame(buffer, args)
        return new_frames
        
    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(input_size=(self.crop_size, self.crop_size), auto_augment=args.aa, interpolation=args.train_interpolation)
        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer); buffer = buffer.permute(0, 2, 3, 1)
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        buffer = buffer.permute(3, 0, 1, 2)
        scl, asp = ([0.08, 1.0], [0.75, 1.3333])
        buffer = spatial_sampling(buffer, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=args.input_size, random_horizontal_flip=False if args.data_set == 'SSV2' else True, inverse_uniform_sampling=False, aspect_ratio=asp, scale=scl, motion_shift=False)
        if self.rand_erase:
            erase_transform = RandomErasing(args.reprob, mode=args.remode, max_count=args.recount, num_splits=args.recount, device="cpu")
            buffer = buffer.permute(1, 0, 2, 3); buffer = erase_transform(buffer); buffer = buffer.permute(1, 0, 2, 3)
        return buffer

    def load_video(self, sample, sample_rate_scale=1):
        fname = sample
        try: vr = self.video_loader(fname)
        except Exception as e: print(f"Failed to load video from {fname} with error {e}!"); return []
        length = len(vr)
        if self.mode == 'test':
            if self.sparse_sample:
                tick = length / float(self.num_segment); all_index = []
                for t_seg in range(self.test_num_segment): tmp_index = [int(t_seg * tick / self.test_num_segment + tick * x) for x in range(self.num_segment)]; all_index.extend(tmp_index)
                all_index = list(np.sort(np.array(all_index)))
            else:
                all_index = [x for x in range(0, length, self.frame_sample_rate)]
                while len(all_index) < self.clip_len: all_index.append(all_index[-1])
            vr.seek(0); buffer = vr.get_batch(all_index).asnumpy(); return buffer
        converted_len = int(self.clip_len * self.frame_sample_rate); seg_len = length // self.num_segment; all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len: index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate); index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len)); index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                if self.mode == 'validation': end_idx = (converted_len + seg_len) // 2
                else: end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len; index = np.linspace(str_idx, end_idx, num=self.clip_len); index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len; all_index.extend(list(index))
        all_index = all_index[::int(sample_rate_scale)]; vr.seek(0); buffer = vr.get_batch(all_index).asnumpy(); return buffer

# --- LỚP SetDataset_JSON MỚI, ĐÃ TÁI CẤU TRÚC ĐỂ GIẢI QUYẾT LỖI HẾT RAM ---
class SetDataset_JSON:
    def __init__(self, image_size, data_path, data_file, batch_size, transform):
        self.image_size = image_size
        self.batch_size = batch_size # batch_size ở đây là n_shot + n_query
        self.transform = transform
        
        json_path = os.path.join(data_path, data_file)
        with open(json_path, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = sorted(list(np.unique(self.meta['image_labels'])))
        self.class_to_paths = {cl: [] for cl in self.cl_list}
        
        # Nhóm các đường dẫn video theo class label
        for path, label in zip(self.meta['image_names'], self.meta['image_labels']):
            self.class_to_paths[label].append(path)

    def __getitem__(self, i):
        # 'i' là class index được cung cấp bởi EpisodicBatchSampler
        target_class = self.cl_list[i]
        
        video_paths_for_class = self.class_to_paths[target_class]
        
        # Ngẫu nhiên chọn ra 'batch_size' (n_shot + n_query) video từ class đó
        sampled_paths = random.choices(video_paths_for_class, k=self.batch_size)
        
        videos
