import logging
from aiogram import types, Dispatcher
from aiogram.dispatcher import Dispatcher
from trading_signals import fetch_candles, analyze_data, generate_trade_signal

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start_command(message: types.Message):
    await message.reply("Привет! Я твой бот.")

def register_handlers(dp: Dispatcher):
    dp.register_message_handler(start_command, commands=['start', 'help'])

async def cmd_start(message: types.Message):
    """
    Ответ на команду /start.
    """
    await message.reply("Привет! Я бот для анализа криптовалют. Используйте команду /analyse <символ> для получения аналитической информации.")

async def cmd_help(message: types.Message):
    """
    Ответ на команду /help.
    """
    help_text = (
        "Список доступных команд:\n"
        "/start - начальное приветствие.\n"
        "/help - показать эту справку.\n"
        "/analyse <символ> - анализ указанной криптовалюты.\n"
    )
    await message.reply(help_text)

async def cmd_analyse(message: types.Message):
    args = message.get_args().split()
    if not args:
        await message.reply("Пожалуйста, укажите символ криптовалюты. Например: /analyse BTC")
        return

    symbol = args[0].upper()
    selected_timeframe = '1d'  # Пример временного интервала

    try:
        # Предполагается, что fetch_candles возвращает DataFrame
        df = await fetch_candles(symbol, selected_timeframe)
        if df.empty:
            await message.reply("Нет данных для анализа.")
            return

        df_analyzed = analyze_data(df)
        signal = generate_trade_signal(df_analyzed, symbol, target_price, stop_loss)
        await message.reply(f"Аналитическая информация для {symbol}: {signal}")
    except Exception as e:
        logger.error(f"Ошибка при выполнении анализа для {symbol}: {e}")
        await message.reply(f"Произошла ошибка при попытке анализа для {symbol}.")

def register_handlers(dp: Dispatcher):
    """
    Регистрация обработчиков команд.
    """
    dp.register_message_handler(cmd_start, commands=["start"])
    dp.register_message_handler(cmd_help, commands=["help"])
    dp.register_message_handler(cmd_analyse, commands=["analyse"])
