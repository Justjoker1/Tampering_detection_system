import os
import argparse
import torch
from torch.utils.data import DataLoader
import sys
import cv2
from .stage_one_net import CustomDataset, FullModel


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Predict whether videos are tampered using a trained FullModel"
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help="视频文件夹地址"
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help="Path to the trained model weights (.pth)."
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help="Batch size for DataLoader (default: 1)."
    )
    parser.add_argument(
        '--seq_len', type=int, default=5,
        help="Number of frames per sample sequence (default: 5)."
    )
    parser.add_argument(
        '--target_w', type=int, default=1280,
        help="Target width for frame resizing (default: 1280)."
    )
    parser.add_argument(
        '--target_h', type=int, default=720,
        help="Target height for frame resizing (default: 720)."
    )
    parser.add_argument(
        '--decoder_hidden', type=int, default=64,
        help="Hidden size of LSTM decoder (default: 64)."
    )
    parser.add_argument(
        '--decoder_layers', type=int, default=2,
        help="Number of layers in LSTM decoder (default: 2)."
    )
    parser.add_argument(
        '--num_classes', type=int, default=2,
        help="Number of output classes (default: 2)."
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help="Number of DataLoader workers (default: 4)."
    )

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)
    return args


def main(input_args=None):
    args = parse_args(input_args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 构建预测数据集和数据加载器
    dataset = CustomDataset(
        root_dir=args.data_dir,
        mode='predict',
        pad_last=True,
        seq_len=args.seq_len,
        target_size=(args.target_w, args.target_h),
        apply_attention=False,
        verbose=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # pin_memory=True,
        drop_last=False
    )

    # 加载模型
    model = FullModel(
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
        num_classes=args.num_classes
    )
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    # 预测逻辑：视频累计篡改片段，>=2 则视为篡改
    with torch.no_grad():
        tamper_count = 0
        # 按批次预测
        for i in range(0, len(dataset), args.batch_size):
            batch_idxs = list(range(i, min(i + args.batch_size, len(dataset))))
            # 加载帧并拼 batch
            frames = torch.stack([dataset[j] for j in batch_idxs]).to(device)
            logits = model(frames)
            probs = torch.softmax(logits, dim=1)[:, 1]  # “篡改”类别概率
            tamper_count += (probs >= 0.5).sum().item()
            if tamper_count >= 2:
                tamper = 1
                break
        else:
            # 未达到阈值
            tamper = 0
    print(f"预测结果: {tamper}")

    return tamper


if __name__ == "__main__":
    input = [
        '--data_dir', "../output/tobedetected_video/",
        '--weights', "../checkpoints/stageone_model_best.pth",
    ]
    main(input_args=input)
