import logging
import os
import pandas as pd
from datetime import datetime
import asyncio
import ccxt.async_support as ccxt
import talib
from dotenv import load_dotenv
from aiogram import types

# Настройка логгирования и загрузка переменных окружения
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

async def fetch_candles(symbol, timeframe):
    # Если timeframe передан как список, используем первый элемент списка
    if isinstance(timeframe, list):
        timeframe = timeframe[0] if len(timeframe) > 0 else '1D'
    
    logger.info(f"Fetching candles for {symbol.upper()} + '/USDT' with timeframe {timeframe}")
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
    })

    # Словарь для преобразования таймфреймов
    timeframe_map = {
        '1D': '1d',
        '1H': '1h',
        '5M': '5m',        
    }

    # Проверяем, есть ли заданный таймфрейм в словаре преобразований
    api_timeframe = timeframe_map.get(timeframe)
    if not api_timeframe:
        logger.error(f"Unsupported timeframe: {timeframe}")
        return pd.DataFrame()

    try:
        # Выполняем запрос к API для получения данных о свечах
        candles = await exchange.fetch_ohlcv(symbol.upper() + '/USDT', api_timeframe, limit=100)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Добавляем строку логирования для отслеживания количества успешно полученных строк (свечей)
        logger.info(f"Data fetched successfully for symbol: {symbol}, rows: {len(df)}")

        return df.dropna()
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {e}")
        return pd.DataFrame()
    finally:
        await exchange.close()

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
    # Добавляем логирование в начале анализа данных
    logger.info(f"Starting data analysis for {len(df)} data points")

    if df.empty:
        logger.info("DataFrame is empty.")
        return df

    # Technical indicators using TA-Lib
    df['momentum_macd'], _, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['momentum_rsi'] = talib.RSI(df['close'], timeperiod=14)
    upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_high'] = upperband
    df['bb_low'] = lowerband

    # Добавляем логирование после успешного завершения анализа данных
    logger.info("Data analysis completed successfully")
    return df

def generate_trade_signal(df_analyzed, symbol, target_price, stop_loss):
    # Начальное логирование перед генерацией торгового сигнала
    logger.info(f"Generating trade signal for {symbol}")

    if df_analyzed.empty:
        logger.info(f"No data for analysis for {symbol}")
        return "No data for analysis."

    last_row = df_analyzed.iloc[-1]
    direction = "🟩 LONG" if last_row['momentum_macd'] > 0 and last_row['momentum_rsi'] > 50 else "🟥 SHORT"
    entry_price = last_row['close']

    # Логирование деталей сигнала
    logger.info(f"Direction: {direction}, Entry Price: {entry_price}, Target Price: {target_price}, Stop Loss: {stop_loss}")

    # Формирование сообщения
    message = f"""
Анализ рынка на: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Текущая цена для {symbol}/USDT: {entry_price:.2f}
Сигнал: {direction}
📈 Точка входа: {entry_price:.2f}
🎯 Цель: {target_price:.2f}
🚫 Стоп лосс: {stop_loss:.2f}
    """
    # Завершающее логирование после успешной генерации торгового сигнала
    logger.info(f"Trade signal generated successfully for {symbol}")
    return message

def get_target_stop_loss(df_analyzed):
    logger.info("Calculating target price and stop loss")
    close_price = df_analyzed['close'].iloc[-1]
    target_price = close_price * 1.05
    stop_loss = close_price * 0.95
    logger.info(f"Target price: {target_price}, Stop loss: {stop_loss}")
    return target_price, stop_loss

async def main(symbol, timeframes):
    logger.info(f"Starting analysis for {symbol} across timeframes: {timeframes}")

    results = await fetch_candles(symbol, timeframes)
    for timeframe, df in results.items():
        if df.empty:
            logger.info(f"No data for analysis for timeframe: {timeframe}. Symbol: {symbol}")
            continue
        
        logger.info(f"Analyzing data for symbol: {symbol} with timeframe: {timeframe}")
        df_analyzed = analyze_data(df)
        target_price = calculate_target_price(df_analyzed)
        stop_loss = calculate_stop_loss(df_analyzed)
        
        logger.info(f"Generating trade signal for symbol: {symbol} with timeframe: {timeframe}")
        signal = generate_trade_signal(df_analyzed, symbol, target_price, stop_loss)
        logger.info(signal)

    logger.info(f"Analysis completed for {symbol} across timeframes: {timeframes}")

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    timeframe_key = '1D'  # Пример использования одного таймфрейма
    asyncio.run(main(symbol, timeframe_key))
