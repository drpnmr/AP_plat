import streamlit as st
import pandas as pd
import numpy as np
import requests
from components.sidebar import show_sidebar

BACKEND_URL = "https://ap-plat.onrender.com"

def clip_value(val, min_v, max_v):
    """Помогает удержать дефолтное значение в рамках динамических границ слайдера"""
    return max(min_v, min(val, max_v))

st.set_page_config(page_title="Оценка стоимости жилья", layout="wide")
st.title("ОЦЕНКА СТОИМОСТИ НЕДВИЖИМОСТИ")

show_sidebar()

@st.cache_data(ttl=3600)
def load_all_backend_meta():
    """Запрашивает все настройки, списки полей и географические лимиты из БД за один вызов"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/apartments/meta", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Не удалось получить метаданные с бэкэнда: {e}")
    return None

meta_data = load_all_backend_meta()

if not meta_data:
    st.error("Критическая ошибка: Бэкэнд недоступен или база данных пуста! Проверьте работу main.py.")
    st.stop()

# Извлекаем списки селектбоксов напрямую из структуры API бэкэнда

districts = meta_data["districts"]
room_types = meta_data["rooms"]
renovations = meta_data["categories"]["renovations"]
heating_types = meta_data["heating"]
views = meta_data["views"]
balconies = meta_data["balconies"]
bathrooms = meta_data["bathrooms"]
housing_types = meta_data["housing_types"]
medians = meta_data["medians"]

# Перестраиваем словари связей локаций и лимитов слайдеров на основе данных из PostgreSQL
district_to_micro = {}
micro_distance_bounds = {}

for row in meta_data["geo_data"]:
    m = row["Микрорайон"]
    d = row["Район"]
    
    if d not in district_to_micro:
        district_to_micro[d] = []
    if m not in district_to_micro[d]:
        district_to_micro[d].append(m)
        
    min_c = row["min_center"] if row["min_center"] is not None else 0
    max_c = row["max_center"] if row["max_center"] is not None else 25000
    micro_distance_bounds[(m, 'center')] = (max(0, int(min_c) - 500), int(max_c) + 500)
    
    min_v = row["min_station"] if row["min_station"] is not None else 0
    max_v = row["max_station"] if row["max_station"] is not None else 25000
    micro_distance_bounds[(m, 'station')] = (max(0, int(min_v) - 500), int(max_v) + 500)
    
    min_a = row["min_airport"] if row["min_airport"] is not None else 0
    max_a = row["max_airport"] if row["max_airport"] is not None else 40000
    micro_distance_bounds[(m, 'airport')] = (max(0, int(min_a) - 1000), int(max_a) + 1000)

for d in district_to_micro:
    district_to_micro[d] = sorted(district_to_micro[d])


# ОРИГИНАЛЬНАЯ ДВУХКОЛОНОЧНАЯ СТРУКТУРА КЛАССИЧЕСКОГО ДЕШБОРДА
col_inputs, col_sticky_result = st.columns([2, 1], gap="large")

with col_inputs:
    st.markdown("### Параметры объекта")
    
    # Контейнеры объявлены строго в нужном порядке: Дополнительные параметры НАД основными
    expander_layout = st.container()
    main_layout = st.container()

    with main_layout:
        sub_col1, sub_col2 = st.columns(2)
        
        with sub_col1:
            st.markdown("##### Локация")
            district = st.selectbox("Район", districts)
            
            available_micros = district_to_micro.get(district, [])
            if not available_micros:
                available_micros = ["Не указано"]
            microdistrict = st.selectbox("Микрорайон", available_micros)
            
            rooms = st.selectbox("Планировка/Комнаты", room_types)
            renovation = st.selectbox("Ремонт", renovations)
            
            min_center, max_center = micro_distance_bounds.get((microdistrict, 'center'), (0, 25000))
            min_station, max_station = micro_distance_bounds.get((microdistrict, 'station'), (0, 25000))
            
            default_center = clip_value(4500, min_center, max_center)
            default_station = clip_value(4500, min_station, max_station)
            
            dist_center = st.slider("Расстояние до центра (м)", int(min_center), int(max_center), int(default_center), step=100, key=f"c_{microdistrict}")
            dist_train_station = st.slider("Расстояние до вокзала Краснодар-1 (м)", int(min_station), int(max_station), int(default_station), step=100, key=f"s_{microdistrict}")

        with sub_col2:
            st.markdown("##### Параметры квартиры")
            total_area = st.number_input("Общая площадь (м²)", min_value=10.0, max_value=300.0, value=45.0, step=1.0)
            floor = st.number_input("Этаж", min_value=1, max_value=29, value=5, step=1)
            total_floors = st.number_input("Этажность дома", min_value=1, max_value=29, value=9, step=1)
            house_age = st.number_input("Возраст дома (лет)", min_value=0, max_value=110, value=10, step=1)
            dist_adult_clinic = st.slider("Расстояние до взрослой поликлиники (м)", 0, 25000, 4500, step=100)

            if floor == 1: 
                floor_type = "Первый"
            elif floor == total_floors and total_floors > 1: 
                floor_type = "Последний"
            else: 
                floor_type = "Другой"

    with expander_layout:
        with st.expander("Дополнительные параметры", expanded=False):
            geo_col1, geo_col2 = st.columns(2)
            with geo_col1:
                min_airport, max_airport = micro_distance_bounds.get((microdistrict, 'airport'), (0, 40000))
                default_airport = clip_value(16000, min_airport, max_airport)
                
                dist_airport = st.slider("Расстояние до аэропорта (м)", int(min_airport), int(max_airport), int(default_airport), step=500, key=f"a_{microdistrict}")
                dist_park = st.slider("Расстояние до парка (м)", 0, 6000, 800, step=50)
                ceil_height = st.number_input("Высота потолков (м)", min_value=2.2, max_value=6.0, value=float(medians.get('Высота потолков', 2.7)), step=0.1)
                is_furnished = st.selectbox("Продаётся с мебелью", ["Нет", "Да"])
                
                view_from_windows = st.selectbox("Вид из окон", views, index=0)
                balcony = st.selectbox("Балкон/лоджия", balconies, index=0)
                heating = st.selectbox("Отопление", heating_types)
                kitchen_percentage = st.slider("Доля площади кухни (%)", min_value=5, max_value=50, value=20, step=1)
                living_percentage = st.slider("Доля жилой площади (%)", min_value=10, max_value=80, value=55, step=1)
            
            with geo_col2:
                dist_kinder = st.slider("До детского сада (м)", 0, 5000, 400, step=50)
                dist_school_geo = st.slider("До школы (м)", 0, 5000, 500, step=50)
                dist_additional_edu = st.slider("До доп. образования (м)", 0, 6000, 1500, step=50)
                dist_child_clinic = st.slider("До детской поликлиники (м)", 0, 6000, 1200, step=50)
                
                bathroom = st.selectbox("Санузел", bathrooms, index=0)
                housing_type = st.selectbox("Тип жилья", housing_types, index=0)
                
                sub_schools = st.slider("Школ в радиусе 1 км", 0, 7, int(medians.get('Кол-во школ в радиусе 1 км', 1)))
                sub_kinder = st.slider("Садов в радиусе 1 км", 0, 15, int(medians.get('Кол-во детсадов в радиусе 1 км', 2)))
                sub_parks = st.slider("Парков в радиусе 1 км", 0, 8, int(medians.get('Кол-во парков в радиусе 1 км', 1)))
                sub_med = st.slider("Мед. учреждений в 1 км", 0, 5, int(medians.get('Кол-во мед. учреждений в радиусе 1 км', 1)))

if (kitchen_percentage + living_percentage) > 95:
    st.warning("Сумма долей кухни и жилой площади слишком велика (превышает 95%)")
    st.stop()        

data_is_valid = True
warning_message = ""

if floor > total_floors:
    st.warning("Выбранный этаж квартиры не может превышать общую этажность дома")
    st.stop()
    
try:
    rooms_digit = int(''.join(filter(str.isdigit, str(rooms))))
except ValueError:
    rooms_digit = 1  

min_area_for_rooms = {1: 15.0, 2: 32.0, 3: 48.0, 4: 70.0, 5: 95.0, 6: 115.0}
max_area_for_rooms = {1: 90.0, 2: 130.0, 3: 190.0, 4: 250.0, 5: 350.0, 6: 450.0}

min_allowed_area = min_area_for_rooms.get(rooms_digit, 115.0)
max_allowed_area = max_area_for_rooms.get(rooms_digit, 450.0)

if total_area < min_allowed_area:
    data_is_valid = False
    warning_message = f"**Площадь слишком мала!** Для {rooms}-комнатной квартиры указанная площадь ({total_area} м²) является слишком маленькой."
elif total_area > max_allowed_area:
    data_is_valid = False
    warning_message = f"**Площадь слишком велика!** Для {rooms}-комнатной квартиры указанная площадь ({total_area} м²) является огромной."

if data_is_valid:
    kitchen_share_val = kitchen_percentage / 100.0
    living_share_val = living_percentage / 100.0
    
    calculated_kitchen_area = total_area * kitchen_share_val
    calculated_living_area = total_area * living_share_val

    payload = {
        "district": str(district), "microdistrict": str(microdistrict), "renovation": str(renovation),
        "rooms": str(rooms), "heating": str(heating), "is_furnished": str(is_furnished),
        "view_from_windows": str(view_from_windows), "balcony": str(balcony), "bathroom": str(bathroom),
        "floor_type": str(floor_type), "housing_type": str(housing_type),
        "total_area": float(total_area), "calculated_kitchen_area": float(calculated_kitchen_area), "calculated_living_area": float(calculated_living_area),
        "kitchen_share_val": float(kitchen_share_val), "living_share_val": float(living_share_val),
        "ceil_height": float(ceil_height), "house_age": float(house_age), "floor": float(floor), "total_floors": float(total_floors),
        "dist_center": float(dist_center), "dist_train_station": float(dist_train_station), "dist_airport": float(dist_airport),
        "dist_park": float(dist_park), "dist_school_geo": float(dist_school_geo), "dist_kinder": float(dist_kinder), 
        "dist_child_clinic": float(dist_child_clinic), "dist_additional_edu": float(dist_additional_edu), "dist_adult_clinic": float(dist_adult_clinic),
        "sub_kinder": float(sub_kinder), "sub_parks": float(sub_parks), "sub_schools": float(sub_schools), "sub_med": float(sub_med)
    }

    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/predict", json=payload, timeout=10)
        
        if response.status_code == 200:
            res_data = response.json()
            predicted_price = res_data["predicted_price"]
            factors_up = res_data["factors_up"]
            factors_down = res_data["factors_down"]
            
            price_string = f"{int(predicted_price):,} ₽".replace(",", " ")
            price_m2_string = f"{int(predicted_price / total_area):,} ₽/м²".replace(",", " ")

            with col_sticky_result:
                st.markdown("### Результаты оценки")
                st.metric(label="Рыночная стоимость объекта", value=price_string)
                st.metric(label="Цена за квадратный метр", value=price_m2_string)
                
            st.markdown("---")
            st.subheader("Главные факторы, повлиявшие на оценку")

            col_up, col_down = st.columns(2, gap="large")

            with col_up:
                st.markdown("<h4 style='color: #2E7D32; margin-top: 10px;'>Что повысило стоимость</h4>", unsafe_allow_html=True)
                if factors_up:
                    for factor in factors_up[:4]:
                        html_card = f"""
                        <div style="background-color: #E8F5E9; padding: 10px 14px; border-radius: 6px; margin-bottom: 6px; border-left: 4px solid #2E7D32;">
                            <span style="color: #1B5E20; font-size: 14px; font-family: sans-serif;">{factor['text']}</span>
                        </div>
                        """
                        st.html(html_card)
                else:
                    st.markdown("<div style='color: gray; font-style: italic;'>Выраженных факторов повышения цены не обнаружено.</div>", unsafe_allow_html=True)

            with col_down:
                st.markdown("<h4 style='color: #C62828; margin-top: 10px;'>Что снизило стоимость</h4>", unsafe_allow_html=True)
                if factors_down:
                    for factor in factors_down[:4]:
                        html_card = f"""
                        <div style="background-color: #FFEBEE; padding: 10px 14px; border-radius: 6px; margin-bottom: 6px; border-left: 4px solid #C62828;">
                            <span style="color: #B71C1C; font-size: 14px; font-family: sans-serif;">{factor['text']}</span>
                        </div>
                        """
                        st.html(html_card)
                else:
                    st.markdown("<div style='color: gray; font-style: italic;'>Выраженных факторов снижения цены не обнаружено.</div>", unsafe_allow_html=True)
                    
        else:
            st.error(f"Ошибка бэкэнда: {response.json().get('detail')}")
    except Exception as e:
        st.error(f"Не удалось подключиться к бэкэнду: {e}")
else:
    st.warning(warning_message)