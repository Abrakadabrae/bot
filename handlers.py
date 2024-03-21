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
    selected_timeframes = ['1D']  # Пример временного интервала в виде списка

    try:
        # Теперь fetch_candles возвращает словарь DataFrame'ов
        results = await fetch_candles(symbol, selected_timeframes)
        timeframe_arg = ['1D']  # Пример полученного аргумента как список
        timeframe_str = timeframe_arg[0] if isinstance(timeframe_arg, list) and len(timeframe_arg) > 0 else '1D'
        
        # Проверяем, есть ли данные для выбранного таймфрейма
        if '1D' in results and not results['1D'].empty:
            df_analyzed = analyze_data(results['1D'])
            # Предполагается, что calculate_target_price и calculate_stop_loss определены
            target_price = calculate_target_price(df_analyzed)
            stop_loss = calculate_stop_loss(df_analyzed)
            signal = generate_trade_signal(df_analyzed, symbol, target_price, stop_loss)
            await message.reply(f"Аналитическая информация для {symbol}: {signal}")
        else:
            await message.reply("Нет данных для анализа.")
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
