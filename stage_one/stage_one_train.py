import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from stage_one_net import CustomDataset, FullModel
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None and self.alpha.device != targets.device:
            self.alpha = self.alpha.to(targets.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epochs=100,
    print_freq=20
):
    model.to(device)
    model.train()
    all_results = []
    best_acc = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        tn = 0  # TT：真实负例预测为负例（True Negative）
        tp = 0  # TP：真实正例预测为正例（True Positive）
        fp = 0  # PT：真实负例预测为正例（False Positive）
        fn = 0  # PP：真实正例预测为负例（False Negative）
        total_samples = 0

        for batch_idx, (frames, labels) in enumerate(dataloader):
            frames = frames.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True)
            batch_size = targets.size(0)

            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(logits, dim=1)  # 预测标签（0或1）

            # 计算当前 batch 的四分类统计
            batch_tn = ((targets == 0) & (preds == 0)).sum().item()
            batch_tp = ((targets == 1) & (preds == 1)).sum().item()
            batch_fp = ((targets == 0) & (preds == 1)).sum().item()
            batch_fn = ((targets == 1) & (preds == 0)).sum().item()

            tn += batch_tn
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            if batch_idx % print_freq == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")
                all_results.append({
                    'epoch': epoch+1,
                    'batch_idx': batch_idx,
                    'batch_loss': loss.item(),
                    'tn': batch_tn,
                    'tp': batch_tp,
                    'fp': batch_fp,
                    'fn': batch_fn
                })

        # Epoch 结束统计
        epoch_loss = total_loss / total_samples
        epoch_acc = (tn + tp) / total_samples  # 总准确率

        # 输出 2x2 列联表（TT=TN, TP=TP, PT=FP, PP=FN）
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print("预测详情（2×2列联表）:")
        print("          预测正例 | 预测负例")
        print("---------------------------------")
        print(f"真实正例 |   {tp:6} |   {fn:6}")
        print(f"真实负例 |   {fp:6} |   {tn:6}")
        print("注：TT=真实负预测负（TN），TP=真实正预测正（TP），PT=真实负预测正（FP），PP=真实正预测负（FN）")

        # 保存到结果数据
        all_results.append({
            'epoch': epoch+1,
            'epoch_loss': epoch_loss,
            'epoch_acc': epoch_acc,
            'tn': tn,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })

        # 保存模型
        torch.save(model.state_dict(), '../checkpoints/stageone_model_last.pth')
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), '../checkpoints/stageone_model_best.pth')
            print(f"保存最优模型 权重：stageone_model_best.pth")

    # 保存到 Excel
    df = pd.DataFrame(all_results)
    os.makedirs('./checkpoints', exist_ok=True)
    df.to_excel('../model_performance/stageone_training_results.xlsx', index=False)
    print("训练完成，结果已保存。")


if __name__ == '__main__':
    config = {
        'root_dir': '../dataset_split/train',
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'decoder_hidden': 64,
        'decoder_layers': 2,
        'num_classes': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'apply_attention': True,
        'seq_len': 5
    }

    dataset = CustomDataset(
        root_dir=config['root_dir'],
        mode='train',
        pad_last=True,
        seq_len=config['seq_len'],
        apply_attention=config['apply_attention']
    )

    positive_indices = [i for i, lbl in enumerate(dataset.seq_labels) if int(lbl) == 1]
    negative_indices = [i for i, lbl in enumerate(dataset.seq_labels) if int(lbl) == 0]
    num_positive = len(positive_indices)
    num_negative = min(2 * num_positive, len(negative_indices))
    np.random.shuffle(negative_indices)
    selected_negative_indices = negative_indices[:num_negative]
    selected_indices = positive_indices + selected_negative_indices
    np.random.shuffle(selected_indices)
    subset_dataset = Subset(dataset, selected_indices)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=config['batch_size'],
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model = FullModel(
        decoder_hidden=config['decoder_hidden'],
        decoder_layers=config['decoder_layers'],
        num_classes=config['num_classes']
    )
    #loss_weights = torch.tensor([0.75, 1.5], dtype=torch.float32).to(config['device'])
    loss_weights = torch.tensor([0.75, 1.5], dtype=torch.float32).to(config['device'])
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device(config['device']),
        epochs=config['num_epochs']
    )