import os
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Krasnodar Real Estate Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/house_price_model.pkl"
if not os.path.exists(MODEL_PATH) and os.path.exists("../models/house_price_model.pkl"):
    MODEL_PATH = "../models/house_price_model.pkl"

model = None
explainer = None
all_features = []
medians = {}

@app.on_event("startup")
def load_ml_package():
    """Загрузка модели, инициализация SHAP Explainer и заполнение медиан при старте приложения"""
    global model, explainer, all_features, medians
    if os.path.exists(MODEL_PATH):
        try:
            package = joblib.load(MODEL_PATH)
            if isinstance(package, dict):
                model = package.get("model")
                all_features = package.get("features", [])
                medians = package.get("medians", {})
            else:
                model = package
                all_features = model.feature_names_ if hasattr(model, 'feature_names_') else []
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            print(f"Ошибка при инициализации: {e}")

@app.get("/api/v1/health", tags=["Системные"])
def health_check():
    """Проверка доступности бэкенда"""
    return {"status": "ok"}

class ApartmentPredictRequest(BaseModel):
    district: str
    microdistrict: str
    renovation: str
    rooms: str
    heating: str
    is_furnished: str
    view_from_windows: str
    balcony: str
    bathroom: str
    floor_type: str
    housing_type: str
    total_area: float
    calculated_kitchen_area: float
    calculated_living_area: float
    kitchen_share_val: float
    living_share_val: float
    ceil_height: float
    house_age: float
    floor: float
    total_floors: float
    dist_center: float
    dist_train_station: float
    dist_airport: float
    dist_park: float
    dist_school_geo: float
    dist_kinder: float
    dist_child_clinic: float
    dist_additional_edu: float
    dist_adult_clinic: float
    sub_kinder: float
    sub_parks: float
    sub_schools: float
    sub_med: float

@app.post("/api/v1/predict", tags=["Аналитика"])
def predict_price(req: ApartmentPredictRequest):
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Модель или SHAP компонент не готовы")

    input_dict = {
        'Район': req.district, 'Микрорайон': req.microdistrict, 'Ремонт': req.renovation,
        'Комнаты/Планировка': req.rooms, 'Отопление': req.heating, 'Продаётся с мебелью': req.is_furnished,
        'Вид из окон': req.view_from_windows, 'Балкон/лоджия': req.balcony, 'Санузел': req.bathroom,
        'Тип этажа': req.floor_type, 'Тип жилья': req.housing_type,
        'Общая площадь': req.total_area, 
        'Площадь кухни': req.calculated_kitchen_area, 
        'Жилая площадь': req.calculated_living_area,
        'Доля кухни': req.kitchen_share_val, 
        'Доля жилой площади': req.living_share_val, 
        'Высота потолков': req.ceil_height, 'Возраст дома': req.house_age, 
        'Этаж': req.floor, 'Этажность дома': req.total_floors, 
        'Расстояние до центра (м)': req.dist_center, 'Расстояние до вокзала (м)': req.dist_train_station,
        'Расстояние до аэропорта (м)': req.dist_airport,
        'Расстояние до park (м)' if 'Расстояние до park (м)' in all_features else 'Расстояние до парка (м)': req.dist_park, 
        'Расстояние до школы (м)': req.dist_school_geo, 'Расстояние до детсада (м)': req.dist_kinder, 
        'Расстояние до дет поликлиники (м)': req.dist_child_clinic, 'Расстояние до доп. образования (м)': req.dist_additional_edu,
        'Расстояние до взр поликлиники (м)': req.dist_adult_clinic, 
        'Кол-во детсадов в радиусе 1 км': req.sub_kinder, 'Кол-во парков в радиусе 1 км': req.sub_parks, 
        'Кол-во школ в радиусе 1 км': req.sub_schools, 'Кол-во школ доп. образования в радиусе 1 км': 1.0,
        'Кол-во мед. учреждений в радиусе 1 км': req.sub_med
    }

    input_data = pd.DataFrame([input_dict])

    for col in all_features:
        if col not in input_data.columns:
            input_data[col] = medians.get(col, 0)

    input_data = input_data[all_features]

    try:
        predicted_log_res = model.predict(input_data)
        single_log_price = predicted_log_res[0].item() if hasattr(predicted_log_res[0], 'item') else predicted_log_res[0]
        predicted_price = np.expm1(single_log_price)

        shap_explanation = explainer(input_data)
        current_shap_values = shap_explanation[0]

        base_log_value = shap_explanation.base_values
        if isinstance(base_log_value, np.ndarray):
            base_log_value = base_log_value[0]

        current_accumulated_log = base_log_value
        factors_up = []
        factors_down = []

        feature_labels = {
            'Общая площадь': 'Общая площадь', 'Жилая площадь': 'Жилая площадь', 'Площадь кухни': 'Площадь кухни',
            'Комнаты/Планировка': 'Комнаты/Планировка', 'Район': 'Район', 'Микрорайон': 'Микрорайон', 'Ремонт': 'Ремонт',
            'Возраст дома': 'Возраст дома', 'Этаж': 'Этаж', 'Тип этажа': 'Тип этажа', 'Высота потолков': 'Высота потолков',
            'Расстояние до центра (м)': 'Расстояние до центра', 'Расстояние до вокзала (м)': 'Расстояние до вокзала',
            'Расстояние до парка (м)': 'Расстояние до парка', 'Расстояние до park (м)': 'Расстояние до парка', 'Отопление': 'Отопление'
        }

        for feature_name, shap_val in zip(all_features, current_shap_values.values):
            user_val = input_data.iloc[0][feature_name]
            
            price_before = np.expm1(current_accumulated_log)
            price_after = np.expm1(current_accumulated_log + shap_val)
            
            rub_contribution = int(price_after - price_before)
            abs_contribution = abs(rub_contribution)
            
            current_accumulated_log += shap_val
            
            if abs_contribution <= 5000:
                continue

            if isinstance(user_val, float):
                val_str = str(int(user_val)) if user_val.is_integer() else f"{user_val:.1f}".replace('.', ',')
            elif isinstance(user_val, (int, np.integer)):
                val_str = f"{user_val:,}".replace(",", " ") if user_val > 1000 else str(user_val)
            else:
                val_str = str(user_val)

            if 'площадь' in feature_name.lower(): val_str += " м²"
            elif 'расстояние' in feature_name.lower(): val_str += " м"
            elif 'высота' in feature_name.lower(): val_str += " м"
            elif 'возраст' in feature_name.lower(): val_str += " лет"

            contribution_str = f"{abs_contribution:,} ₽".replace(",", " ")
            display_name = feature_labels.get(feature_name, feature_name.replace('Расстояние до ', '').replace(' (м)', ''))

            if rub_contribution > 0:
                text = f"Параметр <b>{display_name}</b> со значением <b>{val_str}</b> увеличивает стоимость на <b>+{contribution_str}</b>"
                factors_up.append({'text': text, 'val': abs_contribution})
            else:
                text = f"Параметр <b>{display_name}</b> со значением <b>{val_str}</b> снижает стоимость на <b>-{contribution_str}</b>"
                factors_down.append({'text': text, 'val': abs_contribution})

        return {
            "predicted_price": float(predicted_price),
            "factors_up": sorted(factors_up, key=lambda x: x['val'], reverse=True),
            "factors_down": sorted(factors_down, key=lambda x: x['val'], reverse=True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

@app.get("/api/v1/analytics/top-districts", tags=["Аналитика"])
def get_top_districts():
    """Возвращает метрики для первых 4 районов города напрямую из БД"""
    try:
        query = """
            SELECT "Район" as district, 
                   AVG("Цена за кв.м") as avg_price_m2, 
                   COUNT(*) as count 
            FROM "apartments" 
            WHERE "Район" IS NOT NULL AND "Район" != ''
            GROUP BY "Район"
            LIMIT 4;
        """
        df_top = pd.read_sql_query(query, engine)
        return df_top.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка БД: {str(e)}")

@app.get("/api/v1/analytics/districts-list", tags=["Аналитика"])
def get_districts_list():
    """Возвращает отсортированный список всех уникальных районов из БД"""
    try:
        query = """
            SELECT DISTINCT "Район" 
            FROM "apartments" 
            WHERE "Район" IS NOT NULL AND "Район" != ''
            ORDER BY "Район" ASC;
        """
        df_districts = pd.read_sql_query(query, engine)
        return df_districts["Район"].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка БД: {str(e)}")

@app.get("/api/v1/analytics/district-info", tags=["Аналитика"])
def get_district_info(district: str):
    """Возвращает агрегированные данные для графиков по конкретному району"""
    try:
        query = 'SELECT "Район", "Микрорайон", "Цена", "Цена за кв.м" FROM "apartments" WHERE "Район" = %s'
        district_sub_df = pd.read_sql_query(query, engine, params=(district,))
        
        if district_sub_df.empty:
            raise HTTPException(status_code=404, detail="Указанный район не найден")
            
        avg_price_microraion = district_sub_df.groupby("Микрорайон")['Цена за кв.м'].mean().reset_index()
        avg_price_microraion = avg_price_microraion.sort_values(by='Цена за кв.м', ascending=False)
        bar_data = avg_price_microraion.to_dict(orient="records")
        
        hist_prices = district_sub_df['Цена'].dropna().tolist()
        
        stats_df = district_sub_df.groupby(["Микрорайон"]).agg(
            Количество_объектов=('Цена', 'count'),
            Средняя_цена=('Цена', 'mean'),
            Медианная_цена=('Цена', 'median'),
            Средняя_цена_м2=('Цена за кв.м', 'mean')
        ).reset_index()
        stats_df = stats_df.sort_values(by='Средняя_цена_м2', ascending=False)
        table_data = stats_df.to_dict(orient="records")
        
        return {
            "bar_data": bar_data,
            "hist_prices": hist_prices,
            "table_data": table_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка БД: {str(e)}")

@app.get("/api/v1/apartments/search", tags=["Объявления"])
def search_apartments(
    district: Optional[str] = None,
    rooms: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None
):
    """Динамический поиск и фильтрация объявлений недвижимости в базе данных PostgreSQL с именованными параметрами"""
    try:
        query_string = 'SELECT * FROM "apartments" WHERE 1=1'
        params = {}
        
        if district and district.strip() and district != "Все районы":
            query_string += ' AND "Район" = :district'
            params["district"] = district.strip()
            
        if rooms and rooms.strip() and rooms != "Все":
            rooms_val = rooms.strip()
            if rooms_val.endswith("-комн."):
                rooms_cleaned = rooms_val.replace("-комн.", "")
                query_string += ' AND ("Комнаты/Планировка"::text = :rooms_cl OR "Комнаты/Планировка"::text = :rooms_fl)'
                params["rooms_cl"] = rooms_cleaned
                params["rooms_fl"] = f"{rooms_cleaned}.0"
            else:
                query_string += ' AND "Комнаты/Планировка"::text = :rooms_val'
                params["rooms_val"] = rooms_val
                
        if price_min is not None:
            query_string += ' AND "Цена" >= :price_min'
            params["price_min"] = price_min
            
        if price_max is not None:
            query_string += ' AND "Цена" <= :price_max'
            params["price_max"] = price_max
            
        
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query_string), conn, params=params)
            
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        return df.to_dict(orient="records")
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Внутренняя ошибка бэкенда при фильтрации данных: {str(e)}"
        )

@app.get("/api/v1/apartments/meta", tags=["Данные"])
def get_meta_limits():
    """Возвращает полный набор метаданных для инициализации интерфейса фронтенда"""
    try:
        with engine.connect() as conn:
            districts = pd.read_sql_query('SELECT DISTINCT "Район" FROM "apartments" WHERE "Район" IS NOT NULL AND "Район" != \'\';', conn)['Район'].tolist()
            rooms = pd.read_sql_query('SELECT DISTINCT "Комнаты/Планировка" FROM "apartments" WHERE "Комнаты/Планировка" IS NOT NULL;', conn)['Комнаты/Планировка'].tolist()
            renovations = pd.read_sql_query('SELECT DISTINCT "Ремонт" FROM "apartments" WHERE "Ремонт" IS NOT NULL;', conn)['Ремонт'].tolist()
            heating = pd.read_sql_query('SELECT DISTINCT "Отопление" FROM "apartments" WHERE "Отопление" IS NOT NULL;', conn)['Отопление'].tolist()
            views = pd.read_sql_query('SELECT DISTINCT "Вид из окон" FROM "apartments" WHERE "Вид из окон" IS NOT NULL;', conn)['Вид из окон'].tolist()
            balconies = pd.read_sql_query('SELECT DISTINCT "Балкон/лоджия" FROM "apartments" WHERE "Балкон/лоджия" IS NOT NULL;', conn)['Балкон/лоджия'].tolist()
            bathrooms = pd.read_sql_query('SELECT DISTINCT "Санузел" FROM "apartments" WHERE "Санузел" IS NOT NULL;', conn)['Санузел'].tolist()
            housing_types = pd.read_sql_query('SELECT DISTINCT "Тип жилья" FROM "apartments" WHERE "Тип жилья" IS NOT NULL;', conn)['Тип жилья'].tolist()
            
            limits = pd.read_sql_query('SELECT MIN("Цена") as min_price, MAX("Цена") as max_price FROM "apartments";', conn).to_dict(orient="records")[0]
            
            geo_query = """
                SELECT 
                    "Район", 
                    "Микрорайон",
                    MIN("Расстояние до центра (м)") as min_center,
                    MAX("Расстояние до центра (м)") as max_center,
                    MIN("Расстояние до вокзала (м)") as min_station,
                    MAX("Расстояние до вокзала (м)") as max_station,
                    MIN("Расстояние до аэропорта (м)") as min_airport,
                    MAX("Расстояние до аэропорта (м)") as max_airport
                FROM "apartments"
                WHERE "Район" IS NOT NULL AND "Микрорайон" IS NOT NULL
                GROUP BY "Район", "Микрорайон";
            """
            geo_data = pd.read_sql_query(geo_query, conn).to_dict(orient="records")
            
        return {
            "districts": sorted([str(d).strip() for d in districts if d]),
            "rooms": [str(r).strip() for r in rooms if r],
            "limits": limits,
            "categories": {
                "renovations": sorted([str(x).strip() for x in renovations if x]),
                "types": sorted([str(x).strip() for x in housing_types if x]),
                "heating": sorted([str(x).strip() for x in heating if x]),
                "views": sorted([str(x).strip() for x in views if x]),
                "bathrooms": sorted([str(x).strip() for x in bathrooms if x])
            },
            "heating": sorted([str(x).strip() for x in heating if x]),
            "views": sorted([str(x).strip() for x in views if x]),
            "balconies": sorted([str(x).strip() for x in balconies if x]),
            "bathrooms": sorted([str(x).strip() for x in bathrooms if x]),
            "housing_types": sorted([str(x).strip() for x in housing_types if x]),
            "medians": medians,
            "geo_data": geo_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка метаданных: {str(e)}")

@app.get("/api/v1/analytics/infrastructure-stats", tags=["Аналитика"])
def get_infrastructure_stats():
    """Возвращает глобальные средние по городу для оценки инфраструктуры микрорайонов"""
    try:
        query = """
            SELECT 
                AVG("Кол-во детсадов в радиусе 1 км") as count_kinder,
                AVG("Кол-во школ в радиусе 1 км") as count_schools,
                AVG("Кол-во мед. учреждений в радиусе 1 км") as count_med,
                AVG("Кол-во парков в радиусе 1 км") as count_park
            FROM "apartments";
        """
        df_stats = pd.read_sql_query(query, engine)
        return df_stats.to_dict(orient="records")[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/api/v1/analytics/infrastructure", tags=["Аналитика"])
def get_infrastructure_data(district: str, microdistrict: str):
    """Возвращает инфраструктурные показатели для выбранного микрорайона и глобальные медианы города"""
    try:
        global_query = """
            SELECT 
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Кол-во школ в радиусе 1 км") as count_school,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Кол-во детсадов в радиусе 1 км") as count_kinder,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Кол-во парков в радиусе 1 км") as count_park,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Кол-во мед. учреждений в радиусе 1 км") as count_med
            FROM "apartments";
        """
        df_global = pd.read_sql_query(global_query, engine)
        city_stats = df_global.iloc[0].to_dict()
        
        micro_query = """
            SELECT 
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Расстояние до центра (м)") as dist_center,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Расстояние до вокзала (м)") as dist_station,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Расстояние до аэропорта (м)") as dist_airport,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Расстояние до школы (м)") as dist_school,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Расстояние до парка (м)") as dist_park,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Расстояние до детсада (м)") as dist_kinder,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Расстояние до дет поликлиники (м)") as dist_child_clinic,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Расстояние до взр поликлиники (м)") as dist_adult_clinic,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Кол-во школ в радиусе 1 км") as count_school,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Кол-во детсадов в радиусе 1 км") as count_kinder,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Кол-во парков в радиусе 1 км") as count_park,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "Кол-во мед. учреждений в радиусе 1 км") as count_med
            FROM "apartments"
            WHERE "Район" = %s AND "Микрорайон" = %s;
        """
        df_micro = pd.read_sql_query(micro_query, engine, params=(district, microdistrict))
        
        if df_micro.empty or pd.isna(df_micro.iloc[0]['dist_center']):
            m_stats = {
                'dist_center': 5000, 'dist_station': 6000, 'dist_airport': 15000, 'dist_school': 1200, 'dist_park': 800,
                'dist_kinder': 600, 'dist_child_clinic': 1000, 'dist_adult_clinic': 1500,
                'count_school': 1, 'count_kinder': 2, 'count_park': 1, 'count_med': 1
            }
        else:
            m_stats = df_micro.iloc[0].to_dict()
            
        return {
            "city_stats": city_stats,
            "micro_stats": m_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка БД: {str(e)}")