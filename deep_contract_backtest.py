import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

# --- 1. 配置路径 ---
PATH = os.path.join(os.path.expanduser("~"), "Desktop", "BTC")
DATA_FILE = os.path.join(PATH, "okx_btc_swap_5m.csv")
MODEL_PATH = os.path.join(PATH, "btc_gru_model.pth")
SCALER_PATH = os.path.join(PATH, "scaler.gz")

# 交易参数
CONF_THRESHOLD = 70.0  # 80% 开仓阈值
REVERSAL_EXIT = 90.0   # 90% 极端避险
INITIAL_BALANCE = 18.0 # 初始本金
LOT_SIZE = 0.001        # 单笔 0.001 BTC
FEE_RATE = 0.001 * 2   # OKX 双边手续费预估
LEVERAGE = 10          # 10倍杠杆

# --- 2. 模型结构 ---
class BTC_Pro_GRU(nn.Module):
    def __init__(self, input_size=7, hidden_size=128):
        super(BTC_Pro_GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

# --- 3. 执行回测 ---
def run_deep_backtest():
    if not os.path.exists(DATA_FILE):
        print(f"❌ 找不到文件: {DATA_FILE}")
        return

    print(f"🚀 启动深度逻辑回测 | 初始资金: {INITIAL_BALANCE}U | 杠杆: {LEVERAGE}x")
    
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.lower()
    
    time_col = 'timestamp'
    if time_col not in df.columns:
        if 'time' in df.columns: time_col = 'time'
        else: 
            df[time_col] = range(len(df))
            print("⚠️ 未找到时间列，已自动生成索引作为替代")

    scaler = joblib.load(SCALER_PATH)
    model = BTC_Pro_GRU()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    features = ['close', 'rsi', 'ema20', 'obv', 'volume', 'macd', 'atr']
    X_scaled = scaler.transform(df[features].values)

    balance = INITIAL_BALANCE
    cur_size = 0  # 1 多, -1 空, 0 空仓
    entry_price = 0
    trade_count = 0
    win, loss = 0, 0
    total_profit = 0.0   # 累计盈利金额
    total_loss_val = 0.0 # 累计亏损金额

    print("-" * 100)
    print(f"{'序号':<5} | {'时间':<18} | {'方向':<4} | {'开仓价':<10} | {'平仓价':<10} | {'净盈亏(U)':<8} | {'余额(U)':<8} | {'原因'}")
    print("-" * 100)

    for i in range(10, len(df)):
        seq = torch.FloatTensor(X_scaled[i-10:i]).unsqueeze(0)
        with torch.no_grad():
            up_p = model(seq).item() * 100
            dn_p = 100 - up_p

        price = df.loc[i, 'close']
        current_time = df.loc[i, time_col]

        if cur_size == 0:
            if up_p >= CONF_THRESHOLD:
                cur_size, entry_price = 1, price
                balance -= (price * LOT_SIZE * (FEE_RATE / 2))
            elif dn_p >= CONF_THRESHOLD:
                cur_size, entry_price = -1, price
                balance -= (price * LOT_SIZE * (FEE_RATE / 2))

        else:
            pnl = (price - entry_price) * LOT_SIZE if cur_size == 1 else (entry_price - price) * LOT_SIZE
            
            if (balance + pnl) <= 0:
                print(f"\n💥 账户于 {current_time} 强平爆仓！(价格反向波动触发)")
                total_loss_val += balance # 爆仓相当于亏损剩余全部本金
                balance = 0
                break

            should_close = False
            reason = ""

            if (cur_size == -1 and up_p >= REVERSAL_EXIT) or (cur_size == 1 and dn_p >= REVERSAL_EXIT):
                should_close, reason = True, "极端避险"
            elif (cur_size == 1 and up_p <= 50) or (cur_size == -1 and dn_p <= 50):
                should_close, reason = True, "趋势反转"

            if should_close:
                trade_count += 1
                exit_fee = (price * LOT_SIZE * (FEE_RATE / 2))
                net_pnl = pnl - exit_fee
                balance += net_pnl
                
                if net_pnl > 0: 
                    win += 1
                    total_profit += net_pnl
                else: 
                    loss += 1
                    total_loss_val += abs(net_pnl)
                
                side_str = "做多" if cur_size == 1 else "做空"
                print(f"{trade_count:<6} | {str(current_time):<18} | {side_str:<4} | {entry_price:<10.2f} | {price:<10.2f} | {net_pnl:<10.2f} | {balance:<10.2f} | {reason}")
                
                cur_size = 0
                
    print("-" * 100)
    if trade_count > 0:
        win_rate = (win / trade_count * 100)
        # 计算盈亏比：总盈利 / 总亏损 (避免除以0)
        pr_ratio = (total_profit / total_loss_val) if total_loss_val > 0 else float('inf')
        
        print(f"✅ 回测结束统计:")
        print(f"   最终余额: {balance:.2f} U")
        print(f"   累计成交: {trade_count} 笔")
        print(f"   胜率: {win_rate:.2f}% (盈利: {win} / 亏损: {loss})")
        print(f"   盈亏比: {pr_ratio:.2f}")
    else:
        print("❌ 回测期间未触发交易信号。")

if __name__ == "__main__":
    run_deep_backtest()