import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from components.sidebar import show_sidebar


def clip_value(val, min_v, max_v):
    """Помогает удержать дефолтное значение в рамках динамических границ слайдера"""
    return max(min_v, min(val, max_v))


st.set_page_config(page_title="Оценка стоимости жилья", layout="wide")
st.title("ОЦЕНКА СТОИМОСТИ НЕДВИЖИМОСТИ")

show_sidebar()


@st.cache_data(ttl=3600)
def load_data():

    possible_paths = ["data/Main_dataset.xlsx", "frontend/data/Main_dataset.xlsx", "../data/Main_dataset.xlsx"]
    for path in possible_paths:
        if os.path.exists(path): 
            return pd.read_excel(path)

    raise FileNotFoundError(f"Не удалось найти Main_dataset.xlsx. Текущая директория: {os.getcwd()}")

@st.cache_resource
def load_ml_package():
    possible_paths = ["models/house_price_model.pkl", "frontend/models/house_price_model.pkl", "../models/house_price_model.pkl"]
    for path in possible_paths:
        if os.path.exists(path): 
            return joblib.load(path)
    return None

df = load_data()
package = load_ml_package()

if package is None:
    st.error("Файл обученной модели `models/house_price_model.pkl` не найден!")
    st.stop()

if df.empty:
    st.error("Основной файл датасета `Main_dataset.xlsx` не найден!")
    st.stop()


if isinstance(package, dict):
    model = package.get("model")
    all_features = package.get("features", [])
    medians = package.get("medians", {})
else:
    model = package
    all_features = model.feature_names_ if hasattr(model, 'feature_names_') else []
    medians = {}


districts = sorted(df['Район'].dropna().unique().tolist())
room_types = sorted(df['Комнаты/Планировка'].dropna().astype(str).unique().tolist())
renovations = sorted(df['Ремонт'].dropna().unique().tolist())
heating_types = sorted(df['Отопление'].dropna().unique().tolist())


district_to_micro = {}
micro_distance_bounds = {}  # Словарь для хранения мин/макс расстояний по микрорайонам

for d in districts:
    df_dist = df[df['Район'] == d]
    micros_in_district = df_dist['Микрорайон'].dropna().unique().tolist()
    district_to_micro[d] = sorted(micros_in_district)
    
    # Расчет реальных границ расстояний для каждого микрорайона
    for m in micros_in_district:
        df_micro = df_dist[df_dist['Микрорайон'] == m]
        
        # 1. Исправлено: Границы для расстояния до центра (убраны конфликты названий)
        if 'Расстояние до центра (м)' in df_micro.columns:
            min_c = int(df_micro['Расстояние до центра (м)'].min())
            max_c = int(df_micro['Расстояние до central (м)'].max() if 'Расстояние до central (м)' in df_micro.columns else df_micro['Расстояние до центра (м)'].max())
            micro_distance_bounds[(m, 'center')] = (max(0, min_c - 500), max_c + 500)
        else:
            micro_distance_bounds[(m, 'center')] = (0, 25000)

        # 2. Границы для расстояния до вокзала Краснодар-1
        if 'Расстояние до вокзала Краснодар-1 (м)' in df_micro.columns:
            min_v = int(df_micro['Расстояние до вокзала Краснодар-1 (м)'].min())
            max_v = int(df_micro['Расстояние до вокзала Краснодар-1 (м)'].max())
            micro_distance_bounds[(m, 'station')] = (max(0, min_v - 500), max_v + 500)
        else:
            micro_distance_bounds[(m, 'station')] = (0, 25000)

        # 3. Границы для расстояния до аэропорта
        if 'Расстояние до аэропорта (м)' in df_micro.columns:
            min_a = int(df_micro['Расстояние до аэропорта (м)'].min())
            max_a = int(df_micro['Расстояние до аэропорта (м)'].max())
            micro_distance_bounds[(m, 'airport')] = (max(0, min_a - 1000), max_a + 1000)
        else:
            micro_distance_bounds[(m, 'airport')] = (0, 40000)


result_top_container = st.container()

col_inputs, col_sticky_result = st.columns([2, 1], gap="large")


with col_inputs:
    st.markdown("### Параметры объекта")
    
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
            
            dist_center = st.slider("Расстояние до центра (м)", int(min_center), int(max_center), int(default_center), step=100)
            dist_train_station = st.slider("Расстояние до вокзала Краснодар-1 (м)", int(min_station), int(max_station), int(default_station), step=100)

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
                
                dist_airport = st.slider(
                    "Расстояние до аэропорта (м)", 
                    int(min_airport), int(max_airport), int(default_airport), 
                    step=500
                )
                
                dist_park = st.slider("Расстояние до парка (м)", 0, 6000, 800, step=50)
                ceil_height = st.number_input("Высота потолков (м)", min_value=2.2, max_value=6.0, value=float(medians.get('Высота потолков', 2.7)), step=0.1)
                is_furnished = st.selectbox("Продаётся с мебелью", ["Нет", "Да"])
                
                views = sorted(df['Вид из окон'].dropna().unique().tolist()) if 'Вид из окон' in df.columns else ["Не указано"]
                balconies = sorted(df['Балкон/лоджия'].dropna().unique().tolist()) if 'Балкон/лоджия' in df.columns else ["Не указано"]
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
                
                bathrooms = sorted(df['Санузел'].dropna().unique().tolist()) if 'Санузел' in df.columns else ["Не указано"]
                housing_types = sorted(df['Тип жилья'].dropna().unique().tolist()) if 'Тип жилья' in df.columns else ["Не указано"]
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

    input_dict = {
        'Район': str(district), 'Микрорайон': str(microdistrict), 'Ремонт': str(renovation),
        'Комнаты/Планировка': str(rooms), 'Отопление': str(heating), 'Продаётся с мебелью': str(is_furnished),
        'Вид из окон': str(view_from_windows), 'Балкон/лоджия': str(balcony), 'Санузел': str(bathroom),
        'Тип этажа': str(floor_type), 'Тип жилья': str(housing_type),
        
        'Общая площадь': float(total_area), 
        'Площадь кухни': float(calculated_kitchen_area), 
        'Жилая площадь': float(calculated_living_area),
        'Доля кухни': float(kitchen_share_val), 
        'Доля жилой площади': float(living_share_val), 
        
        'Высота потолков': float(ceil_height), 'Возраст дома': float(house_age), 
        'Этаж': float(floor), 'Этажность дома': float(total_floors), 
        'Расстояние до центра (м)': float(dist_center), 'Расстояние до вокзала Краснодар-1 (м)': float(dist_train_station),
        'Расстояние до аэропорта (м)': float(dist_airport), 'Расстояние до park (м)' if 'Расстояние до park (м)' in all_features else 'Расстояние до парка (м)': float(dist_park), 
        'Расстояние до школы (м)': float(dist_school_geo), 'Расстояние до детсада (м)': float(dist_kinder), 
        'Расстояние до детской поликлиники (м)': float(dist_child_clinic), 'Расстояние до доп. образования (м)': float(dist_additional_edu),
        'Расстояние до взрослой поликлиники (м)': float(dist_adult_clinic), 
        'Кол-во детсадов в радиусе 1 км': float(sub_kinder), 'Кол-во парков в радиусе 1 км': float(sub_parks), 
        'Кол-во школ в радиусе 1 км': float(sub_schools), 'Кол-во школ доп. образования в радиусе 1 км': 1.0,
        'Кол-во мед. учреждений в радиусе 1 км': float(sub_med)
    }

    input_data = pd.DataFrame([input_dict])

    if not all_features:
        all_features = list(input_dict.keys())
        
    for col in all_features:
        if col not in input_data.columns:
            input_data[col] = medians.get(col, 0)

    input_data = input_data[all_features]
    
    predicted_log_res = model.predict(input_data)
    
    if isinstance(predicted_log_res, np.ndarray):
        single_log_price = predicted_log_res[0].item() if hasattr(predicted_log_res[0], 'item') else predicted_log_res[0]
    else:
        single_log_price = predicted_log_res
        
    predicted_price = np.expm1(single_log_price)

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(input_data)
    current_shap_values = shap_explanation[0]



if data_is_valid:
    mae_value = package.get("metrics", {}).get("MAE", 350000) if isinstance(package, dict) else 350000
    r2_value = package.get("metrics", {}).get("R2", 0.88) if isinstance(package, dict) else 0.88

    price_string = f"{int(predicted_price):,} ₽".replace(",", " ")
    price_m2_string = f"{int(predicted_price / total_area):,} ₽/м²".replace(",", " ")

    with col_sticky_result:
        st.markdown("### Результат оценки")
        st.metric(label="Рыночная цена", value=price_string)
        st.metric(label="Цена за м²", value=price_m2_string)
        
    

    st.markdown("---")
    st.subheader("Главные факторы, повлиявшие на оценку")

    factors_up = []
    factors_down = []


    base_log_value = shap_explanation.base_values
    if isinstance(base_log_value, np.ndarray):
        base_log_value = base_log_value[0]
        
    current_accumulated_log = base_log_value

    # Проходим по признакам ровно в том порядке, в котором они идут в датасете
    for feature_name, shap_val in zip(all_features, current_shap_values.values):
        user_val = input_data.iloc[0][feature_name]
        
        # Вычисляем цену в рублях ДО учёта этого признака и ПОСЛЕ
        price_before = np.expm1(current_accumulated_log)
        price_after = np.expm1(current_accumulated_log + shap_val)
        
        # Чистый вклад этого признака в рублях
        rub_contribution = int(price_after - price_before)
        abs_contribution = abs(rub_contribution)
        
        # Обновляем накопительный логарифм для следующего шага
        current_accumulated_log += shap_val
        
        # Форматирование отображения значений признаков
        if isinstance(user_val, float):
            val_str = f"{user_val:.1f}"
        elif isinstance(user_val, (int, np.integer)):
            val_str = f"{user_val:,}".replace(",", " ") if user_val > 1000 else str(user_val)
        else:
            val_str = str(user_val)

        contribution_str = f"{abs_contribution:,} ₽".replace(",", " ")

        # Отсекаем совсем мелкие колебания (менее 5 000 рублей)
        if abs_contribution > 5000:
            if rub_contribution > 0:
                factors_up.append({
                    'text': f"<b>Значение '{user_val}' для параметра {feature_name}</b> увеличивает стоимость на <b>+{contribution_str}</b>",
                    'val': abs_contribution
                })
            else:
                factors_down.append({
                    'text': f"<b>Значение '{user_val}' для параметра{feature_name}</b> снижает стоимость на <b>-{contribution_str}</b>",
                    'val': abs_contribution
                })


    factors_up = sorted(factors_up, key=lambda x: x['val'], reverse=True)
    factors_down = sorted(factors_down, key=lambda x: x['val'], reverse=True)


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
            # Показываем топ-4 главных факторов снижения цены
            for factor in factors_down[:4]:
                html_card = f"""
                <div style="background-color: #FFEBEE; padding: 10px 14px; border-radius: 6px; margin-bottom: 6px; border-left: 4px solid #C62828;">
                    <span style="color: #B71C1C; font-size: 14px; font-family: sans-serif;">{factor['text']}</span>
                </div>
                """
                st.html(html_card)
        else:
            st.markdown("<div style='color: gray; font-style: italic;'>Выраженных факторов снижения цены не обнаружено.</div>", unsafe_allow_html=True)