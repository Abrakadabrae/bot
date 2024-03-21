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

async def fetch_candles(symbol, timeframe):
    logger.info(f"Fetching candles for {symbol} with timeframe {timeframe}")
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
    })
    try:
        supported_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if timeframe not in supported_timeframes:
            logger.error(f"Invalid timeframe: {timeframe}")
            return pd.DataFrame()  # Return empty DataFrame in case of error

        candles = await exchange.fetch_ohlcv(symbol.upper() + '/USDT', timeframe, limit=1000)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"Successfully fetched {len(df)} candles for {symbol} with timeframe {timeframe}")
        return df
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {e}")
        return pd.DataFrame()
    finally:
        await exchange.close()

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

async def main(symbol, timeframe):
    logger.info(f"Starting analysis for {symbol} with timeframe {timeframe}")
    df = await fetch_candles(symbol, timeframe)
    if not df.empty:
        df_analyzed = analyze_data(df)
        target_price, stop_loss = get_target_stop_loss(df_analyzed)
        logger.info(f"Before calling generate_trade_signal — Target price: {target_price}, Stop loss: {stop_loss}")
        signal = generate_trade_signal(df_analyzed, symbol, target_price, stop_loss)
        logger.info(signal)
    else:
        logger.info(f"No data for analysis. Symbol: {symbol}, Timeframe: {timeframe}")

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    timeframe = '1h'
    asyncio.run(main(symbol, timeframe))
