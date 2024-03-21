import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.contrib.fsm_storage.memory import MemoryStorage  # Добавлен импорт MemoryStorage
import os
from dotenv import load_dotenv
from trading_signals import fetch_candles, analyze_data, generate_trade_signal

# Загрузка переменных окружения
load_dotenv('D:/БОт/БОТ/API.env')

API_TOKEN = os.getenv('API_TOKEN')  # Используется переменная окружения для токена
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()  # Подтверждено использование MemoryStorage
dp = Dispatcher(bot, storage=storage)

# Инициализация генетического алгоритма DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) if not hasattr(creator, "FitnessMax") else None
creator.create("Individual", list, fitness=creator.FitnessMax) if not hasattr(creator, "Individual") else None

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 10, 50)
toolbox.register("attr_float", random.uniform, 20, 50)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_int, toolbox.attr_float, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda individual: (random.uniform(0, 1),))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Популяция и генетический алгоритм
population = toolbox.population(n=100)
NGEN = 40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

best_ind = tools.selBest(population, 1)[0]
print("Лучшие параметры:", best_ind)
print("Пригодность:", best_ind.fitness.values)

async def fetch_historical_data(symbol, timeframe='1D', limit=100):
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    since = exchange.parse8601('2021-01-01T00:00:00Z')  # Пример: начало 2021 года
    candles = await exchange.fetch_ohlcv(symbol.upper() + '/USDT', timeframe, since, limit)
    await exchange.close()
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def get_timeframe_keyboard(symbol):
    keyboard = InlineKeyboardMarkup(row_width=3)
    timeframes = ['15m', '30m', '1H', '4H', '1D']
    for timeframe in timeframes:
        callback_data = f"analyse:{symbol}:{timeframe}"
        button = InlineKeyboardButton(text=timeframe, callback_data=callback_data)
        keyboard.insert(button)
    return keyboard

def get_analyse_keyboard():
    keyboard = InlineKeyboardMarkup()
    analyse_button = InlineKeyboardButton(text="Анализ", switch_inline_query_current_chat="/analyse ")
    keyboard.add(analyse_button)
    return keyboard

@dp.callback_query_handler(lambda c: c.data and c.data.startswith('analyse:'))
async def handle_timeframe_selection(callback_query: types.CallbackQuery):
    _, symbol, timeframe = callback_query.data.split(":")

    data = await fetch_candles(symbol, timeframe)
    analysis_results = analyze_data(data)
    if timeframe in analysis_results:
        signal = generate_trade_signal(df_analyzed, symbol, target_price, stop_loss)
        await bot.answer_callback_query(callback_query.id)
        await bot.send_message(callback_query.from_user.id, f"Анализ для {symbol} за {timeframe}...")

@dp.message_handler(commands=['analyse'])
async def prompt_timeframe_selection(message: types.Message):
    args = message.text.split()[1:]  # Извлекаем аргументы после команды
    if not args:
        await message.reply("Введите символ для анализа. Например: /analyse BTC")
        return
    symbol = args[0].upper()
    keyboard = get_timeframe_keyboard(symbol)
    await message.reply("Выберите таймфрейм для анализа:", reply_markup=keyboard)
    timeframes = ['15m', '30m', '1H', '4H', '1D']
    for timeframe in timeframes:
        callback_data = f"analyse:{symbol}:{timeframe}"
        keyboard.insert(InlineKeyboardButton(timeframe, callback_data=callback_data))

    await message.reply("Выберите таймфрейм для анализа:", reply_markup=keyboard)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
