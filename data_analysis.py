import logging
import asyncio
import ccxt.async_support as ccxt
import aiohttp
import pandas as pd
import os
import talib
from config import CRYPTOCOMPARE_API_KEY, CRYPTOCOMPARE_URL

# Загрузка переменных окружения
dotenv_path = os.path.join(os.path.dirname(__file__), 'API.env')
load_dotenv(dotenv_path)

# Создаем объект DataFrame
df = pd.DataFrame({'A': []})

# Проверяем, пуст ли он или нет
if df.empty:
    print("DataFrame пуст.")
else:
    print("DataFrame не пуст.")

async def fetch_price_cryptocompare(symbol: str):
    """
    Асинхронное получение цены для указанной криптовалюты с использованием CryptoCompare API.
    """
    url = f"{CRYPTOCOMPARE_URL}?fsym={symbol}&tsyms=USD"
    headers = {'Apikey': CRYPTOCOMPARE_API_KEY}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                response_data = await response.json()
                price = response_data.get('USD', None)
                if price:
                    return price
                else:
                    logger.error(f"No USD price found for {symbol} in CryptoCompare response.")
                    return None
        except Exception as e:
            logger.error(f"Error fetching price for {symbol} from CryptoCompare: {e}")
            return None

async def fetch_historical_data(symbol: str, timeframe: str = '1D', limit: int = 100):
    """
    Асинхронное получение исторических данных о криптовалюте с использованием библиотеки ccxt.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'apiKey': os.getenv('API_KEY'),
        'secret': os.getenv('API_SECRET'),
    })
    try:
        since = exchange.parse8601('2021-01-01T00:00:00Z')
        candles = await exchange.fetch_ohlcv(symbol.upper() + '/USDT', timeframe, since, limit)
        await exchange.close()
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        await exchange.close()
        return pd.DataFrame()

# Пример функции для анализа данных
def detect_candlestick_patterns(df):
    """
    Обнаружение свечных паттернов в DataFrame.

    :param df: DataFrame с данными о ценах.
    :return: DataFrame с обнаруженными свечными паттернами.
    """
    # Инициализация столбца для хранения обнаруженных паттернов
    df['pattern'] = None

    # Обнаружение различных свечных паттернов
    # Hammer
    hammer = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    df.loc[hammer != 0, 'pattern'] = 'Hammer'

    # Inverted Hammer
    inverted_hammer = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
    df.loc[inverted_hammer != 0, 'pattern'] = 'Inverted Hammer'

    # Doji
    doji = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    df.loc[doji != 0, 'pattern'] = 'Doji'

    # Engulfing Pattern
    engulfing = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df.loc[engulfing != 0, 'pattern'] = 'Engulfing'

    # Morning Star
    morning_star = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
    df.loc[morning_star != 0, 'pattern'] = 'Morning Star'

    return df

# Пример использования
async def main(symbol: str):
    df = await fetch_historical_data(symbol, '1D')  # Предполагается, что эта функция возвращает соответствующий DataFrame
    if not df.empty:
        df_with_patterns = detect_candlestick_patterns(df)
        # Теперь df_with_patterns содержит колонку 'pattern' с обнаруженными паттернами
        print(df_with_patterns[df_with_patterns['pattern'].notnull()])
    else:
        print(f"No historical data for {symbol}")

if __name__ == "__main__":
    asyncio.run(main("BTC"))
