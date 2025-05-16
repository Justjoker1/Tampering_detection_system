import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from stage_one.stage_one_net import CustomDataset as StageOneDataset, FullModel as StageOneModel
from stage_two.stage_two_net import CustomDataset as StageTwoDataset, StageTwoModel

# 加载Stage One模型
def load_stage_one_model(weights_path, decoder_hidden=64, decoder_layers=2, num_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StageOneModel(decoder_hidden=decoder_hidden, decoder_layers=decoder_layers, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 加载Stage Two模型
def load_stage_two_model(lstm_model_path, xgb_model_path, lstm_params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage_two_model = StageTwoModel(lstm_params=lstm_params, xgb_params={})
    stage_two_model.lstm.load_state_dict(torch.load(lstm_model_path, map_location=device))
    stage_two_model.xgb.load_model(xgb_model_path)
    stage_two_model.lstm.to(device)
    stage_two_model.lstm.eval()
    return stage_two_model

# 测试Stage One网络
def test_stage_one(model, data_dir, batch_size=4, seq_len=5, target_w=1280, target_h=720):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = StageOneDataset(
        root_dir=data_dir,
        mode='test',
        pad_last=True,
        seq_len=seq_len,
        target_size=(target_w, target_h),
        apply_attention=False,
        verbose=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device)
            logits = model(frames)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, recall, f1, cm

# 测试Stage Two网络
def test_stage_two(model, features_dir, events_dir, label_dir, status_emb_dim=8, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = StageTwoDataset(
        features_dir=features_dir,
        events_dir=events_dir,
        label_dir=label_dir,
        status_embedding_dim=status_emb_dim,
        mode='test'
    )

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for packed_segs, packed_evts, label in dataset:
            packed_segs = packed_segs.to(device)
            packed_evts = packed_evts.to(device)
            lstm_out = model.lstm(packed_segs, packed_evts).squeeze()
            lstm_features = lstm_out.cpu().numpy()
            xgb_pred = model.xgb.predict(np.array([lstm_features]))
            all_labels.append(label.item())
            all_preds.append(xgb_pred[0])

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, recall, f1, cm

# 测试两阶段模型
def test_two_stage(stage_one_model, stage_two_model, stage_one_data_dir, stage_two_features_dir, stage_two_events_dir, stage_two_label_dir, seq_len=5, target_w=1280, target_h=720, status_emb_dim=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage_one_dataset = StageOneDataset(
        root_dir=stage_one_data_dir,
        mode='test',
        pad_last=True,
        seq_len=seq_len,
        target_size=(target_w, target_h),
        apply_attention=False,
        verbose=False
    )
    stage_two_dataset = StageTwoDataset(
        features_dir=stage_two_features_dir,
        events_dir=stage_two_events_dir,
        label_dir=stage_two_label_dir,
        status_embedding_dim=status_emb_dim,
        mode='test'
    )

    # 建立视频文件名到Stage Two数据集索引的映射
    video_name_to_index = {}
    for i, sample_name in enumerate(stage_two_dataset.sample_names):
        video_name = os.path.splitext(sample_name)[0]
        video_name_to_index[video_name] = i

    all_labels = []
    all_preds = []
    for i in range(len(stage_one_dataset)):
        frames, label = stage_one_dataset[i]
        frames = frames.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = stage_one_model(frames)
            probs = torch.softmax(logits, dim=1)[:, 1]
            stage_one_pred = (probs >= 0.5).long().item()

        # 获取当前样本对应的视频文件名
        video_path = stage_one_dataset.video_info[stage_one_dataset.samples[i][0]]['path']
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if video_name in video_name_to_index:
            stage_two_index = video_name_to_index[video_name]
            if stage_one_pred == 1:
                final_pred = 1
            else:
                packed_segs, packed_evts, _ = stage_two_dataset[stage_two_index]
                packed_segs = packed_segs.to(device)
                packed_evts = packed_evts.to(device)
                with torch.no_grad():
                    lstm_out = stage_two_model.lstm(packed_segs, packed_evts).squeeze()
                    lstm_features = lstm_out.cpu().numpy()
                    xgb_pred = stage_two_model.xgb.predict(np.array([lstm_features]))
                    final_pred = xgb_pred[0]
        else:
            # 如果找不到对应的Stage Two样本，默认预测为0
            final_pred = 0

        all_labels.append(label.item())
        all_preds.append(final_pred)

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, recall, f1, cm

# 绘制混淆矩阵图并保存
def plot_and_save_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# 创建 model_performance 文件夹
if not os.path.exists('model_performance'):
    os.makedirs('model_performance')

if __name__ == "__main__":
    # 配置参数
    stage_one_weights = r""
    stage_two_lstm_model_path = r""
    stage_two_xgb_model_path = r""
    stage_one_data_dir = r""
    stage_two_features_dir = r""
    stage_two_events_dir = r""
    stage_two_label_dir = r""
    lstm_params = {
        'input_size': None,
        'evt_size': None,
        'hidden_size': None,
        'num_layers': None
    }

    # 加载模型
    stage_one_model = load_stage_one_model(stage_one_weights)
    stage_two_model = load_stage_two_model(stage_two_lstm_model_path, stage_two_xgb_model_path, lstm_params)

    # 测试Stage One网络
    stage_one_accuracy, stage_one_recall, stage_one_f1, stage_one_cm = test_stage_one(stage_one_model, stage_one_data_dir)
    print(f"Stage One - Accuracy: {stage_one_accuracy:.4f}, Recall: {stage_one_recall:.4f}, F1: {stage_one_f1:.4f}")
    stage_one_metrics = {
        'Accuracy': [stage_one_accuracy],
        'Recall': [stage_one_recall],
        'F1': [stage_one_f1]
    }

    # 测试Stage Two网络
    stage_two_accuracy, stage_two_recall, stage_two_f1, stage_two_cm = test_stage_two(stage_two_model, stage_two_features_dir, stage_two_events_dir, stage_two_label_dir)
    print(f"Stage Two - Accuracy: {stage_two_accuracy:.4f}, Recall: {stage_two_recall:.4f}, F1: {stage_two_f1:.4f}")
    stage_two_metrics = {
        'Accuracy': [stage_two_accuracy],
        'Recall': [stage_two_recall],
        'F1': [stage_two_f1]
    }

    # 测试两阶段模型
    two_stage_accuracy, two_stage_recall, two_stage_f1, two_stage_cm = test_two_stage(stage_one_model, stage_two_model, stage_one_data_dir, stage_two_features_dir, stage_two_events_dir, stage_two_label_dir)
    print(f"Two Stage - Accuracy: {two_stage_accuracy:.4f}, Recall: {two_stage_recall:.4f}, F1: {two_stage_f1:.4f}")
    two_stage_metrics = {
        'Accuracy': [two_stage_accuracy],
        'Recall': [two_stage_recall],
        'F1': [two_stage_f1]
    }
