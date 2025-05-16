import os
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import json


class CustomDataset(data.Dataset):
    def __init__(
            self,
            root_dir,
            mode='train',  # 'train'/'val'/'predict'
            pad_last=True,
            seq_len=5,
            target_size=(1280, 720),
            apply_attention=True,
            verbose=True
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.pad_last = pad_last
        self.seq_len = seq_len
        self.target_w, self.target_h = target_size
        self.apply_attention = apply_attention and (mode in ['train', 'val', 'test'])
        self.verbose = verbose

        if self.mode == 'predict':
            self.video_dir = root_dir
            self.video_paths = self._collect_video_paths()
        else:
            self.video_dir = os.path.join(root_dir, 'video')
            self.label_dir = os.path.join(root_dir, 'label') if mode in ['train', 'val', 'test'] else None
            self.video_paths = self._collect_video_paths()

        print(f"[STAGE_ONE] mode={mode} Found {len(self.video_paths)} videos")

        self.video_info = self._load_video_basic_info()

        if self.mode in ['train', 'val', 'test']:
            self._load_label_information()

        self.samples = self._generate_samples()
        print(f"[STAGE_ONE] mode={mode} Generated {len(self.samples)} samples, seq_len={seq_len}")

        # 预计算所有样本的标签
        self.seq_labels = []
        if self.mode in ['train', 'val', 'test']:
            for vid_idx, start in self.samples:
                info = self.video_info[vid_idx]
                # 计算帧索引
                frame_indices = self._get_frame_indices(start, info['total_frames'])
                _, tams = self._get_label_data(frame_indices, info)
                lbl = 1 if tams.sum() > 0 else 0
                self.seq_labels.append(torch.tensor(lbl, dtype=torch.long))

    def _collect_video_paths(self):
        return sorted([
            os.path.join(self.video_dir, f)
            for f in os.listdir(self.video_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ])

    def _load_video_basic_info(self):
        return [
            {
                'path': path,
                'total_frames': self._get_total_frames(path)
            }
            for path in self.video_paths
        ]

    def _get_total_frames(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total

    def _load_label_information(self):
        for info in self.video_info:
            basename = os.path.splitext(os.path.basename(info['path']))[0]
            label_path = os.path.join(self.label_dir, f"{basename}.json")

            with open(label_path, 'r') as f:
                label_data = json.load(f)

            orig_w = cv2.VideoCapture(info['path']).get(cv2.CAP_PROP_FRAME_WIDTH)
            orig_h = cv2.VideoCapture(info['path']).get(cv2.CAP_PROP_FRAME_HEIGHT)
            sx, sy = self.target_w / orig_w, self.target_h / orig_h

            traj, tamper = {}, {}
            raw_traj = label_data.get('trajectory', {})
            raw_tampered = float(label_data.get('is_tampered', 0))

            for frame_idx in range(info['total_frames']):
                key = str(frame_idx)
                if key in raw_traj:
                    x1, y1, x2, y2 = raw_traj[key]
                    traj[frame_idx] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
                    tamper[frame_idx] = raw_tampered
                else:
                    traj[frame_idx] = [0.0, 0.0, self.target_w, self.target_h]
                    tamper[frame_idx] = 0.0

            info.update({'traj': traj, 'tamper': tamper})

    def _generate_samples(self):
        samples = []
        for vid_idx, info in enumerate(self.video_info):
            max_start = info['total_frames'] - self.seq_len + 1 if not self.pad_last else info['total_frames']
            for start in range(0, max_start, self.seq_len):
                samples.append((vid_idx, start))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_idx, start = self.samples[idx]
        info = self.video_info[vid_idx]
        frame_indices = self._get_frame_indices(start, info['total_frames'])
        frames = self._load_and_preprocess_frames(info['path'], frame_indices)

        if self.mode == 'predict':
            return frames
        else:
            boxes, tams = self._get_label_data(frame_indices, info)
            if self.apply_attention:
                frames = self._apply_attention_mask(frames, boxes, tams)
            label = self.seq_labels[idx]
            return frames, label

    def _get_frame_indices(self, start, total_frames):
        return [
            start + off if (start + off) < total_frames else total_frames - 1
            for off in range(self.seq_len)
        ]

    def _load_and_preprocess_frames(self, path, indices):
        cap = cv2.VideoCapture(path)
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_w, self.target_h))
            frames.append(
                torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            )
        cap.release()
        return torch.stack(frames, dim=1)

    def _get_label_data(self, indices, info):
        boxes = torch.tensor(
            [info['traj'][fi] for fi in indices], dtype=torch.float32
        )
        tams = torch.tensor(
            [info['tamper'][fi] for fi in indices], dtype=torch.float32
        )
        return boxes, tams

    def _apply_attention_mask(self, frames, boxes, tams):
        C, T, H, W = frames.shape
        mask = torch.full((T, H, W), 0.5, device=frames.device)
        for t in range(T):
            if tams[t] == 1:
                x1, y1, x2, y2 = boxes[t].long()
                mask[t, y1:y2, x1:x2] = 1.0
        return frames * mask.unsqueeze(0)

    def _get_seq_label(self, tams):
        return torch.tensor(1 if tams.sum() > 0 else 0, dtype=torch.long)

class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        # SRM 权重定义
        w1 = 0.25 * torch.tensor([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]])
        w2 = 1 / 12 * torch.tensor(
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]])
        w3 = 0.25 * torch.tensor([[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, 4, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]])
        f1 = torch.stack([w1, w1, w1], dim=0)
        f2 = torch.stack([w2, w2, w2], dim=0)
        f3 = torch.stack([w3, w3, w3], dim=0)
        filters = torch.stack([f1, f2, f3], dim=0)
        self.SRMconv = nn.Conv2d(3, 3, 5, padding=2, bias=False)
        self.SRMconv.weight.data = filters
        self.SRMconv.weight.requires_grad = False

    def forward(self, x):
        return self.SRMconv(x)


class C3D_BN_ReLU(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class P3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d((2, 5, 5), (2, 5, 5))

    def forward(self, x):
        return self.pool(x)


class ThreeDToTwoD(nn.Module):
    def __init__(self, in_c, out_c, t):
        super().__init__()
        self.conv3d = nn.Conv3d(in_c, out_c, (t, 1, 1))

    def forward(self, x):
        x = self.conv3d(x)
        return x.squeeze(2)


class C2D_BN_ReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, k, padding=p)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2d(x)))


class P2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d((2, 2), (2, 2))

    def forward(self, x):
        return self.pool(x)


class SharedLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.c3d_8 = C3D_BN_ReLU(3, 8)
        self.p3d = P3D()
        self.c3d_16 = C3D_BN_ReLU(8, 16)
        self.reshape_conv = ThreeDToTwoD(16, 32, 1)
        self.c2d_32 = C2D_BN_ReLU(32, 32)
        self.p2d = P2D()
        self.c2d_64 = C2D_BN_ReLU(32, 64)
        self.p2d2 = P2D()
        self.c2d_128 = C2D_BN_ReLU(64, 128, k=1, p=0)

    def forward(self, x):
        x = self.c3d_8(x)
        x = self.p3d(x)
        x = self.c3d_16(x)
        x = self.reshape_conv(x)
        x = self.c2d_32(x);
        x = self.p2d(x)
        x = self.c2d_64(x);
        x = self.p2d2(x)
        x = self.c2d_128(x)
        return x


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = SRMFilter()
        self.shared = SharedLayers()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x: [B, C, 5, H, W]          原始帧序列，5 帧
            boxes: [B, 5, 4]           每帧锚框信息
            feat_size: [B, 5, 2]       每帧特征尺寸（高度，宽度）
            is_tampered: [B, 5]        每帧是否篡改的标签（0/1）

        Returns:
            branches: [B, 3, 128]      三路特征，每路128维
        """
        B, C, T, H, W = x.shape
        assert T == 5, "Input must have 5 frames"

        # Step 1: SRM Filter - 提升高频特征
        x_srm = x.view(-1, C, H, W)  # [B*T, C, H, W]
        x_srm = self.srm(x_srm)  # [B*T, C, H, W]
        x = x_srm.view(B, C, T, H, W)  # [B, C, 5, H, W]

        # Step 2: 划分三个时间段
        indices = [
            [0, 1, 2],  # branch1
            [1, 2, 3],  # branch2
            [2, 3, 4]  # branch3
        ]

        feats = []
        for idx in indices:
            # Step 3: 提取对应帧片段及其信息
            x_part = x[:, :, idx, :, :]  # [B, C, 3, H, W]

            # Step 4: 通过共享特征网络
            out = self.shared(x_part)  # [B, 128, H', W']
            out = self.pool(out).view(B, -1)  # [B, 128]
            feats.append(out)

        # Step 5: 拼接所有 branch
        branches = torch.stack(feats, dim=1)  # [B, 3, 128]
        return branches


# ================= 修改点 =================
# 1. 将 Encoder(MainModel) 与 Decoder(BiLSTM) 封装到 FullModel 中
# 2. 统一了 Decoder 输出全连接层输入维度，改为 hidden_size*2
class FullModel(nn.Module):
    def __init__(self, decoder_hidden=64, decoder_layers=2, num_classes=2):
        super().__init__()
        self.encoder = MainModel()
        self.decoder = nn.LSTM(
            input_size=128, hidden_size=decoder_hidden,
            num_layers=decoder_layers, batch_first=True,
            bidirectional=True  # 双向 LSTM，输出维度需乘以 2
        )
        self.fc = nn.Linear(decoder_hidden * 2, num_classes)  # 输入维度为 2*hidden_size

    def forward(self, x):
        branches = self.encoder(x)  # [B, T, 128]
        out, (hn, _) = self.decoder(branches)  # out: [B, T, H*2]
        logits = self.fc(out[:, -1, :])  # 只取最后一个时刻做分类
        return logits
