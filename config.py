from dotenv import load_dotenv
import os
import logging

# Загрузка переменных окружения
load_dotenv('D:/БОт/БОТ/API.env')

# Извлечение переменных окружения
API_TOKEN = os.getenv('API_TOKEN', '')
API_KEY = os.getenv('API_KEY', '')
API_SECRET = os.getenv('API_SECRET', '')
CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY', '')

# External service URLs
CRYPTOCOMPARE_URL = 'https://min-api.cryptocompare.com/data/price'

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка наличия критически важных переменных окружения
required_env_vars = ['API_TOKEN', 'API_KEY', 'API_SECRET', 'CRYPTOCOMPARE_API_KEY']
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Required environment variable {var} is not set.")
        raise Exception(f"Required environment variable {var} is not set.")

# Прочие конфигурации
ENABLED_LOGGING = True if os.getenv('ENABLED_LOGGING', 'True').lower() == 'true' else False  # Добавлено преобразование строки в булево значение
DEFAULT_TIMEFRAME = '1d'
