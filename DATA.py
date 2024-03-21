import logging
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, executor
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from config import API_TOKEN
from handlers import register_handlers

def add_technical_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    # Добавление простой скользящей средней (SMA)
    ohlc['SMA'] = ohlc['close'].rolling(window=5).mean()
    
    # Добавление экспоненциальной скользящей средней (EMA)
    ohlc['EMA'] = ohlc['close'].ewm(span=5, adjust=False).mean()
    
    # Расчёт RSI
    delta = ohlc['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    ohlc['RSI'] = 100 - (100 / (1 + rs))
    
    # Расчёт Bollinger Bands
    ohlc['MA20'] = ohlc['close'].rolling(window=20).mean()
    ohlc['STD20'] = ohlc['close'].rolling(window=20).std()
    ohlc['Upper_Band'] = ohlc['MA20'] + (ohlc['STD20'] * 2)
    ohlc['Lower_Band'] = ohlc['MA20'] - (ohlc['STD20'] * 2)
    
    return ohlc

# Создание и обогащение DataFrame техническими индикаторами
data = {
    'open': [98, 101, 102, 104, 107, 109, 111, 108, 113, 115],
    'high': [101, 106, 104, 108, 109, 111, 113, 112, 116, 118],
    'low': [97, 100, 101, 103, 106, 108, 110, 107, 112, 114],
    'close': [100, 105, 103, 107, 108, 110, 112, 110, 115, 117],
    'volume': [1000, 1500, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
}
df = pd.DataFrame(data)
df_with_indicators = add_technical_indicators(df)
print(df_with_indicators)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Регистрация обработчиков команд
register_handlers(dp)

if __name__ == '__main__':
    # Запуск бота
    executor.start_polling(dp, skip_updates=True)
