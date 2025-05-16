import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from .stage_two_net import CustomDataset, HierarchicalLSTMModel, StageTwoModel  # 替换为实际模块路径


# ======================
# 参数解析函数
# ======================
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Stage Two Model Prediction Script")

    # 数据路径参数
    parser.add_argument('--features_dir', type=str,
                        default=r'.\output\features_temp\features',
                        help='Path to features directory')
    parser.add_argument('--events_dir', type=str,
                        default=r'.\output\features_temp\events',
                        help='Path to events directory')
    parser.add_argument('--lstm_model_path', type=str,
                        default=r'.\checkpoints\lstm_best_model.pth',
                        help='Path to LSTM model weights')
    parser.add_argument('--xgb_model_path', type=str,
                        default=r'.\checkpoints\xgb_best_model.json',
                        help='Path to XGBoost model')

    # 模型参数
    parser.add_argument('--status_emb_dim', type=int, default=8,
                        help='Status embedding dimension')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for prediction')

    # LSTM模型参数
    parser.add_argument('--lstm_input_size', type=int, default=9,  # 1+STATUS_EMB_DIM
                        help='LSTM input feature size')
    parser.add_argument('--lstm_hidden_size', type=int, default=64,
                        help='LSTM hidden layer size')
    parser.add_argument('--lstm_output_size', type=int, default=1,
                        help='LSTM output size')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--lstm_evt_size', type=int, default=3,
                        help='LSTM event input size')

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)
    return args


# ======================
# 预测主函数
# ======================
def main(input_args=None):
    args = parse_args(input_args)

    # 配置参数
    FEATURES_DIR = args.features_dir
    EVENTS_DIR = args.events_dir
    LSTM_MODEL_PATH = args.lstm_model_path
    XGB_MODEL_PATH = args.xgb_model_path
    STATUS_EMB_DIM = args.status_emb_dim
    BATCH_SIZE = args.batch_size

    # LSTM模型参数
    LSTM_PARAMS = {
        'input_size': args.lstm_input_size,
        'hidden_size': args.lstm_hidden_size,
        'output_size': args.lstm_output_size,
        'num_layers': args.lstm_num_layers,
        'evt_size': args.lstm_evt_size
    }

    # 初始化数据集（预测模式）
    dataset = CustomDataset(
        features_dir=FEATURES_DIR,
        events_dir=EVENTS_DIR,
        status_embedding_dim=STATUS_EMB_DIM,
        mode='predict',
    )

    # 初始化模型
    stage_two_model = StageTwoModel(
        lstm_params=LSTM_PARAMS,
        xgb_params={}  # 预测时xgb参数从文件加载，此处传空
    )

    # 加载设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage_two_model.lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    stage_two_model.xgb.load_model(XGB_MODEL_PATH)
    stage_two_model.lstm.to(device)

    # 预测函数
    def predict(dataset, model):
        model.lstm.eval()
        predictions = []
        with torch.no_grad():
            for packed_segs, packed_evts in dataset:  # 修改此处，只解包两个值
                packed_segs = packed_segs.to(device)
                packed_evts = packed_evts.to(device)  # 添加这一行，将事件数据移动到设备上
                lstm_out = model.lstm(packed_segs, packed_evts).squeeze()  # 修改此处，传递两个参数
                lstm_features = lstm_out.cpu().numpy()
                xgb_pred = model.xgb.predict(np.array([lstm_features]))  # 包裹为二维数组
                predictions.append(xgb_pred[0])
        return np.array(predictions)

    # 执行预测
    tamper = predict(dataset, stage_two_model)
    print(f"预测结果: {tamper[0]}")
    return tamper[0]


if __name__ == "__main__":
    main()