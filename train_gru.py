import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import joblib

# --- 1. 路径与配置 ---
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "BTC")
DATA_FILE = os.path.join(DESKTOP_PATH, "okx_btc_swap_5m.csv")
MODEL_PATH = os.path.join(DESKTOP_PATH, "btc_gru_model.pth")
SCALER_PATH = os.path.join(DESKTOP_PATH, "scaler.gz")

# 检查数据文件是否存在
if not os.path.exists(DATA_FILE):
    print(f"❌ 找不到数据文件：{DATA_FILE}，请先运行抓取脚本！")
    exit()

# --- 2. 数据预处理 ---
df = pd.read_csv(DATA_FILE)

# 核心特征：增加了 MACD 和 ATR
features = ['close', 'RSI', 'EMA20', 'OBV', 'volume', 'MACD', 'ATR']
X = df[features].values
y = df['Target'].values

# 归一化（让不同量纲的数据在同一量级竞争）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 构建滑动窗口 (用过去 10 个周期预测未来)
def create_sequences(data, labels, seq_length=10):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(labels[i+seq_length])
    return torch.FloatTensor(np.array(X_seq)), torch.FloatTensor(np.array(y_seq))

X_train, y_train = create_sequences(X_scaled, y)

# --- 3. 定义 Pro 版双层 GRU 架构 ---
class BTC_Pro_GRU(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(BTC_Pro_GRU, self).__init__()
        # num_layers=2: 深度堆叠，增加模型复杂度
        # dropout=0.2: 训练时随机“关闭”神经元，防止死记硬背（过拟合）
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        
        # 全连接层：负责最终逻辑判断
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 形状: (batch, seq, features)
        out, _ = self.gru(x)
        # 取时间序列的最后一个点 [:, -1, :]
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

# 实例化模型
model = BTC_Pro_GRU(input_size=len(features))
criterion = nn.BCELoss() # 二分类交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # 调低学习率，找得更准

# --- 4. 自动化训练循环 ---
print(f"🚀 Pro 训练启动！特征维度: {len(features)} | 目标轮次: 200")
print("-" * 50)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/200] | Loss: {loss.item():.4f}")

# --- 5. 保存成果 ---
torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("-" * 50)
print(f"✅ 训练完成！'聪明'的权重已保存至：\n{MODEL_PATH}")