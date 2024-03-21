import logging
from aiogram import types, Dispatcher

from trading_signals import fetch_candles, analyze_data, generate_trade_signal

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start_command(message: types.Message):
    await message.reply("Привет! Я твой бот.")



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
        # Добавляем логирование перед запросом данных
        logger.info(f"Analyzing symbol: {symbol} with timeframe: {selected_timeframes}")

        # Теперь fetch_candles возвращает словарь DataFrame'ов
        results = await fetch_candles(symbol, selected_timeframes)

        # Добавляем логирование после получения данных
        if results:
            logger.info(f"Candles fetched for {symbol}, processing data...")
        else:
            logger.info(f"No data returned for {symbol}")
            await message.reply("Нет данных для анализа.")
            return

        # Проверяем, есть ли данные для выбранного таймфрейма
        if '1D' in results and not results['1D'].empty:
            df_analyzed = analyze_data(results['1D'])
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
