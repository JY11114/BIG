import ccxt
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
import joblib
import time
import os
import requests
from datetime import datetime

# ==========================================
# 1. 手机推送配置 
# ==========================================
TG_TOKEN = "87E"
TG_CHAT_ID = "854"

def send_mobile_msg(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        print(f"❌ 手机推送失败: {e}")

# ==========================================
# 2. 模型结构 
# ==========================================
class BTC_Pro_GRU(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(BTC_Pro_GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32); self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc1(out[:, -1, :]); out = self.relu(out)
        return self.sigmoid(self.fc2(out))

# ==========================================
# 3. 路径与参数 
# ==========================================
PATH = os.path.join(os.path.expanduser("~"), "Desktop", "BTC")
if not os.path.exists(PATH):
    os.makedirs(PATH)

EQUITY_LOG = os.path.join(PATH, "real_equity_log.csv")
LIVE_LOG = os.path.join(PATH, "live_training_data.csv")
MODEL_PATH = os.path.join(PATH, "btc_gru_model.pth")
SCALER_PATH = os.path.join(PATH, "scaler.gz")

SYMBOL = 'BTC/USDT:USDT'
LEVERAGE = 10           
CONF_THRESHOLD = 60.0   #胜率
REVERSAL_EXIT = 90.0    
#STOP_LOSS_U = 4.0       

exchange = ccxt.okx({
    'apiKey': 'cd6',
    'secret': '69B1',
    'password': '1',
    'enableRateLimit': True, 
    'hostname': 'www.okx.com', 
    'options': {'defaultType': 'swap'}
})

# ==========================================
# 4. 交易引擎逻辑 
# ==========================================
def trade_engine():
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='5m', limit=100)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['RSI'] = ta.rsi(df['close'], 14); df['EMA20'] = ta.ema(df['close'], 20)
    df['OBV'] = ta.obv(df['close'], df['volume'])
    macd = ta.macd(df['close']); df['MACD'] = macd['MACD_12_26_9']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], 14)
    df.dropna(inplace=True)

    scaler = joblib.load(SCALER_PATH)
    model = BTC_Pro_GRU(7)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    last_seq = df[['close', 'RSI', 'EMA20', 'OBV', 'volume', 'MACD', 'ATR']].tail(10).values
    input_tensor = torch.FloatTensor(scaler.transform(last_seq)).unsqueeze(0)
    with torch.no_grad():
        up_p = model(input_tensor).item() * 100
        dn_p = 100 - up_p

    bal = exchange.fetch_balance()
    total_assets = float(bal['total']['USDT'])
    pos = exchange.fetch_positions(symbols=[SYMBOL])
    
    # 获取实时盘口价格用于限价单
    orderbook = exchange.fetch_order_book(SYMBOL)
    bid_price = orderbook['bids'][0][0] # 买一
    ask_price = orderbook['asks'][0][0] # 卖一

    pnl = 0.0
    cur_size = 0
    side_desc = "空仓"
    raw_side = "none"
    
    if pos and float(pos[0]['contracts']) != 0:
        pnl = float(pos[0]['unrealizedPnl'])
        cur_size = abs(float(pos[0]['contracts']))
        raw_side = pos[0]['side']
        side_desc = "多单" if raw_side == 'long' else "空单"

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_price = df['close'].iloc[-1]

    if not os.path.exists(EQUITY_LOG):
        pd.DataFrame([[now, total_assets]], columns=['ts', 'balance']).to_csv(EQUITY_LOG, index=False)
    else:
        pd.DataFrame([[now, total_assets]]).to_csv(EQUITY_LOG, mode='a', index=False, header=False)
    
    log_row = df.tail(1).copy()
    log_row['up_prob'], log_row['pnl'], log_row['time'] = up_p, pnl, now
    log_row.to_csv(LIVE_LOG, mode='a', index=False, header=not os.path.exists(LIVE_LOG))

    report = (
        f"📋 *BTC 实盘巡检报告*\n⏰ 时间: `{now}`\n💰 价格: `${current_price:,.1f}`\n"
        f"📈 概率: 涨 `{up_p:.1f}%` | 跌 `{dn_p:.1f}%`\n💼 持仓: `{side_desc} ({cur_size}张)`\n"
        f"💵 盈亏: `{pnl:+.2f} USDT`\n🏦 总资产: `{total_assets:.2f} USDT`"
    )
    print("\n" + "="*30 + "\n" + report.replace('*','').replace('`','') + "\n" + "="*30)

    # --- 交易执行 ---
    if cur_size == 0:
        if up_p > CONF_THRESHOLD:
            # 限价开多 
            exchange.create_limit_buy_order(SYMBOL, 0.1, ask_price)
            send_mobile_msg(f"🚀 *【执行限价开多】*\n价格: {ask_price}\n{report}")
        elif dn_p > CONF_THRESHOLD:
            # 限价开空 
            exchange.create_limit_sell_order(SYMBOL, 0.1, bid_price)
            send_mobile_msg(f"❄️ *【执行限价开空】*\n价格: {bid_price}\n{report}")
        else:
            send_mobile_msg(report)
    else:
        should_close, reason = False, ""
        if (raw_side == 'short' and up_p >= REVERSAL_EXIT) or (raw_side == 'long' and dn_p >= REVERSAL_EXIT):
            should_close, reason = True, "🔥 90% 极端避险"
        elif (raw_side == 'long' and up_p <= 50) or (raw_side == 'short' and up_p >= 50):
            should_close, reason = True, "🔄 趋势反转平仓"

        if should_close:
            order_side = 'buy' if raw_side == 'short' else 'sell'
            # 平仓限价逻辑
            close_price = ask_price if order_side == 'buy' else bid_price
            exchange.create_order(SYMBOL, 'limit', order_side, cur_size, close_price)
            send_mobile_msg(f"✅ *【限价平仓成功】*\n价格: {close_price}\n原因: {reason}\n{report}")
        else:
            send_mobile_msg(report)

# ==========================================
# 5. 主循环
# ==========================================
if __name__ == "__main__":
    try: exchange.set_leverage(LEVERAGE, SYMBOL)
    except: pass
    
    send_mobile_msg("✅ BTC 量化系统 ")
    while True:
        try:
            trade_engine()
        except Exception as e:
            print(f"运行时提示: {e}")
        time.sleep(300)