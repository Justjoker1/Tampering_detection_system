import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn as nn
import xgboost as xgb
import ast

# --------- Data Utilities ---------

def str_to_list(s):
    return ast.literal_eval(s)

def align_list_lengths(row, key):
    l = row[key]
    if isinstance(l, str):
        l = ast.literal_eval(l)
    row[key] = l
    return row

def merge_lists_to_tensor(df, col):
    tensor_list = []
    for _, row in df.iterrows():
        speed = row[col]
        tensor = torch.tensor(speed, dtype=torch.float32).unsqueeze(1)
        tensor_list.append(tensor)
    return tensor_list

# --------- Dataset ---------
import os
import json
import torch
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.nn as nn

def str_to_list(s):
    return ast.literal_eval(s)

def align_list_lengths(row, key):
    l = row[key]
    if isinstance(l, str):
        l = ast.literal_eval(l)
    row[key] = l
    return row

def merge_lists_to_tensor(df, col):
    return [
        torch.tensor(speed, dtype=torch.float32).unsqueeze(1)
        for speed in df[col]
    ]

class CustomDataset(Dataset):
    def __init__(self, features_dir, events_dir, label_dir=None,
                 status_embedding_dim=8, mode='train'):
        super().__init__()
        self.features_dir = features_dir
        self.events_dir   = events_dir
        self.label_dir    = label_dir
        self.mode         = mode  # 'train' or 'predict'

        # 1) 找到所有 base 名
        names = [
            f[:-4] for f in os.listdir(features_dir)
            if f.endswith('.csv')
        ]
        # 2) 只保留同时有 event 和 label (train 模式)
        if mode != 'predict':
            names = [
                b for b in names
                if os.path.exists(os.path.join(events_dir, f"{b}_event.csv")) and
                   os.path.exists(os.path.join(label_dir, f"{b}.json"))
            ]

        # 3) 预筛：模拟 __getitem__ 中对 feats_df 的所有过滤，剔除最终会空的
        valid = []
        for b in names:
            # load feats_df
            feats_df = pd.read_csv(
                os.path.join(features_dir, f"{b}.csv"),
                converters={
                    'Speed': str_to_list,
                    'Frame_Range': ast.literal_eval
                }
            ).drop(columns=['ID'])
            # 筛掉空 Speed
            feats_df = feats_df[feats_df['Speed'].apply(len) > 0]
            # train 模式再按 frame_range 筛
            if mode != 'predict':
                ld = json.load(open(os.path.join(label_dir, f"{b}.json")))
                s0, e0 = ld['frame_range']['start'], ld['frame_range']['end']
                def in_range(fr):
                    start_f = fr[0] if fr[0] is not None else 0
                    end_f   = fr[1] if fr[1] is not None else 1e9
                    return not (start_f > e0 or end_f < s0)
                feats_df = feats_df[feats_df['Frame_Range'].apply(in_range)]

            # align Speed, 筛 Status
            feats_df = feats_df.apply(align_list_lengths, axis=1, args=('Speed',))
            feats_df = feats_df[feats_df['Status'] != 0]
            # 如果还有数据，就保留
            if len(feats_df) > 0:
                valid.append(b)

        self.sample_names = valid
        print(f"[STAGE_TWO] mode={mode}, kept {len(self.sample_names)}/{len(names)} samples")

        # 状态嵌入
        self.status_embed = nn.Embedding(4, status_embedding_dim)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        b = self.sample_names[idx]

        # 1) 读 feats_df
        feats_df = pd.read_csv(
            os.path.join(self.features_dir, f"{b}.csv"),
            converters={
                'Speed': str_to_list,
                'Frame_Range': ast.literal_eval
            }
        ).drop(columns=['ID'])
        feats_df = feats_df[feats_df['Speed'].apply(len) > 0]

        # 2) 读 events
        ev_df = pd.read_csv(os.path.join(self.events_dir, f"{b}_event.csv"))
        frames     = ev_df['Frame'].values.astype(np.float32)
        disappeared= ev_df['Disappeared'].values.astype(np.float32)
        appeared   = ev_df['Appeared'].values.astype(np.float32)
        events     = torch.tensor(
            np.vstack([frames, disappeared, appeared]).T,
            dtype=torch.float32
        )
        evt_len = events.size(0)

        # 3) train 模式下 attention 筛 & label
        if self.mode != 'predict':
            ld = json.load(open(os.path.join(self.label_dir, f"{b}.json")))
            label = torch.tensor(ld['is_tampered'], dtype=torch.long)
            s0, e0 = ld['frame_range']['start'], ld['frame_range']['end']
            def in_range(fr):
                start_f = fr[0] if fr[0] is not None else 0
                end_f = fr[1] if fr[1] is not None else 1e9
                return not (start_f > e0 or end_f < s0)
            feats_df = feats_df[feats_df['Frame_Range'].apply(in_range)]
            mask = (frames >= s0) & (frames <= e0)
            events = events[mask]
            evt_len = events.size(0)
        else:
            label = None

        # 4) speed+status → pad + pack
        feats_df = feats_df.apply(align_list_lengths, axis=1, args=('Speed',))
        feats_df = feats_df[feats_df['Status'] != 0]
        feats_list = merge_lists_to_tensor(feats_df, 'Speed')
        seg_lens   = [t.size(0) for t in feats_list]
        padded_segs= pad_sequence(feats_list, batch_first=True)
        status_idx = torch.tensor(feats_df['Status'].tolist(), dtype=torch.long)
        status_emb = self.status_embed(status_idx).unsqueeze(1)
        status_emb = status_emb.expand(-1, padded_segs.size(1), -1)
        segs = torch.cat([padded_segs, status_emb], dim=-1)
        packed_segs = pack_padded_sequence(
            segs, seg_lens, batch_first=True, enforce_sorted=False
        )

        # 5) events → pack
        ev_pad = events.unsqueeze(0)
        packed_evts = pack_padded_sequence(
            ev_pad, [evt_len], batch_first=True, enforce_sorted=False
        )

        if label is not None:
            return packed_segs, packed_evts, label
        else:
            return packed_segs, packed_evts

# --------- Dual-Branch LSTM Model ---------
class HierarchicalLSTMModel(nn.Module):
    def __init__(self, input_size, evt_size, hidden_size, num_layers=1):
        super().__init__()
        self.speed_lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True)
        self.evt_lstm   = nn.LSTM(evt_size, hidden_size, num_layers,
                                  batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, packed_segs, packed_evts):
        # —— 速度分支 ——
        # packed_segs 对应多轨迹序列的 PackedSequence
        _, (h_s_all, _) = self.speed_lstm(packed_segs)
        h_s_last = h_s_all[-1]   # [num_segments, H]
        # 聚合：对所有轨迹的 hidden 求平均，得到样本级 embedding
        h_s = h_s_last.mean(dim=0, keepdim=True)  # [1, H]

        # —— 事件分支 ——
        _, (h_e_all, _) = self.evt_lstm(packed_evts)
        h_e = h_e_all[-1]      # [1, H]

        # —— 拼接 & 分类 ——
        h = torch.cat([h_s, h_e], dim=1)           # [1, 2H]
        logit = self.fc(h).squeeze(1)              # [1]
        return logit

# --------- StageTwoModel ---------
class StageTwoModel:
    def __init__(self, lstm_params, xgb_params):
        self.lstm = HierarchicalLSTMModel(
            input_size=lstm_params['input_size'],
            evt_size=lstm_params['evt_size'],
            hidden_size=lstm_params['hidden_size'],
            num_layers=lstm_params['num_layers']
        )
        self.xgb = xgb.XGBClassifier(**xgb_params)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=1e-3)

    def train(self, dataloader, num_epochs=10, update_xgb_every=5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm.to(device)
        for epoch in range(num_epochs):
            all_feats, all_labels = [], []
            total_loss = 0
            for packed_segs, seg_lens, packed_evts, labels in dataloader:
                packed_segs = packed_segs.to(device)
                packed_evts = packed_evts.to(device)
                labels = labels.to(device).float()
                self.optimizer.zero_grad()
                logits = self.lstm(packed_segs, packed_evts)
                all_feats.append(logits.detach().cpu().numpy().reshape(-1,1))
                all_labels.append(labels.cpu().numpy())
                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch+1)%update_xgb_every==0:
                X = np.vstack(all_feats)
                y = np.hstack(all_labels)
                self.xgb.fit(X, y)
            print(f"Epoch {epoch+1} loss: {total_loss/len(dataloader):.4f}")

    def predict(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm.to(device).eval()
        feats=[]
        with torch.no_grad():
            for packed_segs, seg_lens, packed_evts, _ in dataloader:
                packed_segs = packed_segs.to(device)
                packed_evts = packed_evts.to(device)
                logits = self.lstm(packed_segs, packed_evts)
                feats.append(logits.cpu().numpy().reshape(-1,1))
        X = np.vstack(feats)
        return self.xgb.predict(X)
