import os
import pandas as pd
from sqlalchemy import create_engine, text

# Шаблон: postgresql://логин:пароль@хост:порт/имя_базы
DATABASE_URL = "postgresql://postgres:12345@localhost:5432/realty"

def migrate():
    # Находим файл
    possible_paths = ["data/Main_dataset.xlsx", "frontend/data/Main_dataset.xlsx", "../data/Main_dataset.xlsx"]
    excel_path = None
    for path in possible_paths:
        if os.path.exists(path):
            excel_path = path
            break
            
    if not excel_path:
        print("Файл Main_dataset.xlsx не найден!")
        return

    print(f"Чтение данных из {excel_path} (это может занять некоторое время)...")
    df = pd.read_excel(excel_path)
    
    # Избавляемся от точек в названиях колонок (PostgreSQL не любит точки и пробелы в именах, но пробелы мы пока оставим в кавычках)
    # Переименуем 'Цена за кв.м' -> 'Цена за кв.м' для безопасности
    
    print("Подключение к PostgreSQL...")
    engine = create_engine(DATABASE_URL)
    
    # 2. Заливаем DataFrame в базу данных
    # if_exists='replace' удалит старую таблицу, если она была, и создаст новую
    print("Загрузка данных в таблицу 'apartments'...")
    df.to_sql("apartments", engine, if_exists="replace", index=False)
    
    # 3. Создаем индексы для ускорения фильтрации и groupby
    print("Создание индексов для оптимизации запросов...")
    with engine.begin() as conn:
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_apartments_rayon ON "apartments" ("Район");'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_apartments_mikrorayon ON "apartments" ("Микрорайон");'))
    
    print("Миграция в PostgreSQL успешно завершена!")

if __name__ == "__main__":
    migrate()