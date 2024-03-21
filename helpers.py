from datetime import datetime

def parse_date(date_str: str, date_format: str = "%Y-%m-%d") -> datetime:
    """Преобразует строку в объект datetime."""
    return datetime.strptime(date_str, date_format)

if __name__ == "__main__":
    # Пример использования функции parse_date
    date_str = "2023-01-01"
    parsed_date = parse_date(date_str)
    print(f"Строка даты '{date_str}' преобразована в объект datetime: {parsed_date}")
