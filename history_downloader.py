import ccxt
import pandas as pd
import pandas_ta as ta
import time

def fetch_okx_contract_data(symbol='BTC/USDT:USDT', timeframe='5m', target_count=20000):
    # 初始化 OKX 交易所对象
    exchange = ccxt.okx({
        'options': {'defaultType': 'swap'}  # 强制指定为掉期/永续合约
    })
    
    all_klines = []
    limit = 100  # 单次请求限额
    
    print(f"📡 正在从 OKX 获取 {symbol} 永续合约数据 (目标: {target_count} 条)...")

    # 获取初始 since（当前时间往回推足够长的时间）
    # 5分钟 * 20000条 = 100,000 分钟
    duration_ms = target_count * 5 * 60 * 1000
    since = exchange.milliseconds() - duration_ms
    
    while len(all_klines) < target_count:
        try:
            # fetch_ohlcv 会根据 since 自动处理分页
            klines = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            
            if not klines or len(klines) == 0:
                print("⚠️ 未能抓取到更多数据，停止翻页。")
                break
            
            all_klines.extend(klines)
            
            # 更新 since：取最后一条数据的时间戳 + 5分钟，向后抓取
            # 或者取最后一条的时间戳 + 1ms 以避免重复抓取最后一条
            since = klines[-1][0] + 1 
            
            if len(all_klines) % 500 == 0:
                print(f"已获取: {len(all_klines)} 条数据...")
            
            # 礼貌访问，防止被 API 限制频率
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"请求出错: {e}")
            break

    # 数据整理
    df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 转换时间格式前先去重，防止内存浪费
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 截取最后的 target_count 条
    final_df = df.tail(target_count).copy()
    print(f"✅ 抓取完成，实际有效数据量: {len(final_df)} 条")
    return final_df

def apply_indicators(df):
    print("📈 正在计算指标...")
    # 保持原有 7 大指标逻辑不变
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['EMA20'] = ta.ema(df['close'], length=20)
    df['OBV'] = ta.obv(df['close'], df['volume'])
    
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # 定义 Target (未来3根K线是否收涨)
    df['Target'] = (df['close'].shift(-3) > df['close']).astype(int)
    
    # 清理预热期的空值
    df.dropna(inplace=True)
    return df

# --- 执行 ---
if __name__ == "__main__":
    # 1. 抓取数量改为 20000
    contract_df = fetch_okx_contract_data(symbol='BTC/USDT:USDT', timeframe='5m', target_count=20000)
    
    # 2. 计算指标
    processed_df = apply_indicators(contract_df)
    
    # 3. 打印末尾 5 行检查（20000条数据建议看尾部，因为头部数据因指标预热会被删除）
    print("\n✅ 处理后的合约数据（最后5行）：")
    print(processed_df.tail())
    
    # 4. 保存为 CSV
    processed_df.to_csv('okx_btc_swap_5m.csv', index=False)
    print(f"💾 数据已保存至 okx_btc_swap_5m.csv")