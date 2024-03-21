import logging
import os
import pandas as pd
from datetime import datetime
import asyncio
import ccxt.async_support as ccxt
import talib
from dotenv import load_dotenv

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

async def fetch_candles(symbol, timeframes):
    logger.info(f"Fetching candles for {symbol}")
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
    })
    
    supported_timeframes = ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '8H', '12H', '1D', '3D', '1w', '1M']
    results = {}
    
    try:
        for timeframe in timeframes:
            if timeframe not in supported_timeframes:
                logger.error(f"Invalid timeframe: {timeframe}")
                continue  # Skip unsupported timeframes

            logger.info(f"Fetching {timeframe} candles for {symbol}")
            candles = await exchange.fetch_ohlcv(symbol.upper() + '/USDT', timeframe, limit=1000)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Assuming 'dropna' is a custom or pandas function to remove NA/NaN values
            results[timeframe] = df.dropna()

            logger.info(f"Successfully fetched {len(df)} candles for {symbol} with timeframe {timeframe}")

    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {e}")
    finally:
        await exchange.close()
    
    return results

def calculate_target_price(df):
    # Использование простой средней цены закрытия для расчета целевой цены.
    target_price = df['close'].mean()
    return target_price

def calculate_stop_loss(df_analyzed):
    # Использование простой логики для расчета стоп-лосса.
    # Например, стоп-лосс на 5% ниже минимальной цены закрытия в анализируемом DataFrame.
    stop_loss = df_analyzed['close'].min() * 0.95
    return stop_loss    

def analyze_data(df):
    logger.info("Analyzing data")
    if df.empty:
        logger.info("DataFrame is empty.")
        return df

    # Technical indicators using TA-Lib
    df['momentum_macd'], _, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['momentum_rsi'] = talib.RSI(df['close'], timeperiod=14)
    upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_high'] = upperband
    df['bb_low'] = lowerband

    logger.info("Data analysis completed")
    return df

def generate_trade_signal(df_analyzed, symbol, target_price, stop_loss):
    logger.info(f"Generating trade signal for {symbol} with target_price: {target_price}, stop_loss: {stop_loss}")
    if df_analyzed.empty:
        return "No data for analysis."

    last_row = df_analyzed.iloc[-1]
    direction = "🟩 LONG" if last_row['momentum_macd'] > 0 and last_row['momentum_rsi'] > 50 else "🟥 SHORT"
    entry_price = last_row['close']

    # Forming the message
    message = f"""
Analysis for: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Current price for {symbol}/USDT: {entry_price:.2f}
Signal: {direction}
📈 Entry Point: {entry_price:.2f}
🎯 Target: {target_price:.2f}
🚫 Stop Loss: {stop_loss:.2f}
    """

    logger.info(f"Trade signal generated for {symbol}")
    return message

def get_target_stop_loss(df_analyzed):
    logger.info("Calculating target price and stop loss")
    close_price = df_analyzed['close'].iloc[-1]
    target_price = close_price * 1.05
    stop_loss = close_price * 0.95
    logger.info(f"Target price: {target_price}, Stop loss: {stop_loss}")
    return target_price, stop_loss

async def main(symbol, timeframes):
    logger.info(f"Starting analysis for {symbol} with timeframes {timeframes}")
    results = await fetch_candles(symbol, timeframes)
    for timeframe, df in results.items():
        if df.empty:
            logger.info(f"No data for analysis for timeframe {timeframe}. Symbol: {symbol}")
            continue

        df_analyzed = analyze_data(df)
        target_price = calculate_target_price(df_analyzed)
        stop_loss = calculate_stop_loss(df_analyzed)
        logger.info(f"Timeframe: {timeframe}, Target price calculated: {target_price}, Stop loss: {stop_loss}")
        signal = generate_trade_signal(df_analyzed, symbol, target_price, stop_loss)
        logger.info(signal)

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    timeframes = ['1D']  # Убедитесь, что это список, даже если таймфрейм один
    asyncio.run(main(symbol, timeframes))
