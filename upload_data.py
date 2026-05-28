from urllib.parse import quote_plus
import pandas as pd
from sqlalchemy import create_engine

raw_password = "PL1aF233orM"

password = "PL1aF233orM"

DB_URL = (
    "postgresql+psycopg2://"
    "postgres.mbgsggsdxqykuqosetoe:"
    f"{password}"
    "@aws-1-eu-central-1.pooler.supabase.com:6543/postgres"
)

def upload_excel_to_supabase():
    file_name = "data/Main_dataset3.xlsx"

    df = pd.read_excel(file_name)

    engine = create_engine(DB_URL)

    try:
        with engine.connect() as conn:
            print("Подключение успешно!")

        

        print("Успешно загружено!")

    except Exception as e:
        print("Ошибка:")
        print(e)

if __name__ == "__main__":
    upload_excel_to_supabase()