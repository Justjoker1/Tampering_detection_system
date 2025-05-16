import os
import json
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from stage_two_net import CustomDataset, StageTwoModel

# ----------------------------
# 配置区
# ----------------------------
FEATURES_DIR = '../dataset_split/train/stagetwo/features'
EVENTS_DIR = '../dataset_split/train/stagetwo/events'
LABEL_DIR = '../dataset_split/train/label'

STATUS_EMB_DIM = 8
NUM_EPOCHS = 100
UPDATE_XGB_EVERY = 5

# LSTM + XGB 超参
LSTM_PARAMS = {
    'input_size': 1 + STATUS_EMB_DIM,
    'evt_size': 3,
    'hidden_size': 64,
    'num_layers': 2
}
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'max_depth': 4,
    'n_estimators': 200,
    'eval_metric': 'logloss',
    'verbosity': 1
}

# 保存训练信息的文件
TRAINING_LOG_FILE = '../model_performance/stagetwo_training_results.xlsx'

# ----------------------------
# 准备数据集和模型
# ----------------------------
dataset = CustomDataset(
    features_dir=FEATURES_DIR,
    events_dir=EVENTS_DIR,
    label_dir=LABEL_DIR,
    status_embedding_dim=STATUS_EMB_DIM,
    mode='train'
)

model = StageTwoModel(
    lstm_params=LSTM_PARAMS,
    xgb_params=XGB_PARAMS
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.lstm.to(device)

# 初始化最佳准确率
best_accuracy = 0.0

# 训练日志列表
training_log = []


# ----------------------------
# 训练函数
# ----------------------------
def train_model(dataset, model, num_epochs, update_xgb_every):
    global best_accuracy
    print(f"Using device: {device}")
    print(f"Total samples: {len(dataset)}")

    for epoch in range(1, num_epochs + 1):
        model.lstm.train()
        total_loss = 0.0

        # —— 1) 训练 LSTM ——
        for packed_segs, packed_evts, label in dataset:
            packed_segs = packed_segs.to(device)
            packed_evts = packed_evts.to(device)
            label = label.to(device).float()

            model.optimizer.zero_grad()
            logits = model.lstm(packed_segs, packed_evts).squeeze()
            loss = model.loss_fn(logits, label)
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{num_epochs} — Avg Loss: {avg_loss:.4f}")

        # —— 2) 更新 XGBoost 并输出列联表 ——
        if epoch % update_xgb_every == 0:
            print(f"Epoch {epoch}: Updating XGBoost...")
            model.lstm.eval()
            feats, labels_list = [], []
            all_preds = []

            with torch.no_grad():
                for packed_segs, packed_evts, label in dataset:
                    packed_segs = packed_segs.to(device)
                    packed_evts = packed_evts.to(device)

                    logit = model.lstm(packed_segs, packed_evts).squeeze()
                    feats.append(logit.cpu().numpy().reshape(-1, 1))
                    labels_list.append(label.item())

            X = np.vstack(feats)
            y = np.array(labels_list, dtype=np.int32)
            model.xgb.fit(X, y)

            preds = model.xgb.predict(X)
            all_preds = preds.astype(int)  # 确保预测值为 0/1 整数

            # 手动计算列联表
            tp = np.sum((y == 1) & (preds == 1))
            fp = np.sum((y == 0) & (preds == 1))
            tn = np.sum((y == 0) & (preds == 0))
            fn = np.sum((y == 1) & (preds == 0))

            # 计算正负样本数量（非比例，直接统计数量）
            positive_samples = np.sum(y == 1)
            negative_samples = np.sum(y == 0)

            # 输出 2x2 列联表
            print("\nConfusion Matrix (2x2):")
            print("          预测正例  预测负例")
            print(f"实际正例    {tp:^9}  {fn:^9}")
            print(f"实际负例    {fp:^9}  {tn:^9}\n")

            # 计算准确率
            acc = (tp + tn) / len(y) if len(y) != 0 else 0.0
            print(f"  [XGB Eval] Accuracy: {acc:.4f} on {len(y)} samples")
            print(f"  正负样本分布：正例 {positive_samples} 个，负例 {negative_samples} 个\n")

            # 记录训练信息（保留数量而非比例，如需比例可自行添加 /len(y)）
            epoch_log = {
                'epoch': epoch,
                'avg_loss': avg_loss,
                'accuracy': acc,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'positive_samples': positive_samples,
                'negative_samples': negative_samples
            }
            training_log.append(epoch_log)

            # 保存最佳模型
            if acc > best_accuracy:
                best_accuracy = acc
                torch.save(model.lstm.state_dict(), '../checkpoints/lstm_best_model.pth')
                model.xgb.save_model('../checkpoints/xgb_best_model.json')
                print(f"Best model saved at epoch {epoch} with accuracy {best_accuracy:.4f}")

    # 保存最后一次模型
    torch.save(model.lstm.state_dict(), '../checkpoints/lstm_last_model.pth')
    model.xgb.save_model('../checkpoints/xgb_last_model.json')
    print("Last model saved.")

    # 保存训练日志到 xlsx 文件
    df = pd.DataFrame(training_log)
    df.to_excel(TRAINING_LOG_FILE, index=False)
    print(f"Training log saved to {TRAINING_LOG_FILE}")


# ----------------------------
# 启动训练
# ----------------------------

if __name__ == '__main__':
    train_model(dataset, model, NUM_EPOCHS, UPDATE_XGB_EVERY)
    print("Training complete.")