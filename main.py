import logging
from aiogram import Bot, Dispatcher, executor
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from config import API_TOKEN
from handlers import register_handlers  # Убедитесь, что функция register_handlers корректно реализована в handlers.py
from DATA import add_technical_indicators
import pandas as pd

# Загрузка или создание DataFrame df
data = {
    'open': [98, 101, 102, 104, 107, 109, 111, 108, 113, 115],
    'high': [101, 106, 104, 108, 109, 111, 113, 112, 116, 118],
    'low': [97, 100, 101, 103, 106, 108, 110, 107, 112, 114],
    'close': [100, 105, 103, 107, 108, 110, 112, 110, 115, 117],
    'volume': [1000, 1500, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
}
df = pd.DataFrame(data)

# Применение функции add_technical_indicators
df_with_indicators = add_technical_indicators(df)
print(df_with_indicators)

# Печатаем значение переменной API_TOKEN
print("API_TOKEN:", API_TOKEN)

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
