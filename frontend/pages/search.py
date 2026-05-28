import streamlit as st
import pandas as pd
import numpy as np
import requests
from components.sidebar import show_sidebar

# Настройка страницы
st.set_page_config(page_title="Поиск недвижимости", layout="wide")
st.title("ПОИСК НЕДВИЖИМОСТИ")
show_sidebar()

# URL твоего FastAPI бэкенда
BASE_API_URL = "https://ap-plat.onrender.com"


# Инициализация состояний сессии
if "compare_list" not in st.session_state:
    st.session_state.compare_list = []
if "compare_data" not in st.session_state:
    st.session_state.compare_data = {}

# Загрузка метаданных (районы, комнаты, лимиты цен) из API бэкенда
@st.cache_data(ttl=600)
def fetch_meta_data():
    try:
        response = requests.get(f"{BASE_API_URL}/api/v1/apartments/meta")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Не удалось подключиться к бэкенду для получения метаданных: {e}")
    return None

# Динамический поиск объектов через API бэкенда
def search_apartments_api(district, rooms, price_min, price_max):
    params = {
        "price_min": price_min,
        "price_max": price_max
    }
    if district and district != "Все районы":
        params["district"] = district
    if rooms and rooms != "Все":
        params["rooms"] = rooms

    try:
        response = requests.get(f"{BASE_API_URL}/api/v1/apartments/search", params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Ошибка при выполнении поиска через API: {e}")
    return []

# Интеграция предиктивной аналитики модели ИИ через API бэкенда
def check_price_anomaly_via_api(row):
    # Приведение типов полов квартиры к категориальному признаку бэкенда
    floor = int(row['Этаж']) if pd.notna(row['Этаж']) else 5
    total_floors = int(row['Этажность дома']) if pd.notna(row['Этажность дома']) else 9
    if floor == 1:
        floor_type = "Первый"
    elif floor == total_floors and total_floors > 1:
        floor_type = "Последний"
    else:
        floor_type = "Другой"

    total_area = float(row['Общая площадь'])
    kitchen_area = float(row['Площадь кухни']) if pd.notna(row.get('Площадь кухни')) else total_area * 0.2
    living_area = float(row['Жилая площадь']) if pd.notna(row.get('Жилая площадь')) else total_area * 0.55

    # Формирование строгого JSON Payload для ApartmentPredictRequest на бэкенде
    payload = {
        "district": str(row.get('Район', '')),
        "microdistrict": str(row.get('Микрорайон', 'Не указано')),
        "renovation": str(row.get('Ремонт', 'Косметический')),
        "rooms": str(row.get('Комнаты/Планировка', '1')),
        "heating": str(row.get('Отопление', 'Центральное')),
        "is_furnished": str(row.get('Продаётся с мебелью', 'Нет')),
        "view_from_windows": str(row.get('Вид из окон', 'Во двор')),
        "balcony": str(row.get('Балкон/лоджия', 'Нет')),
        "bathroom": str(row.get('Санузел', 'Совмещенный')),
        "floor_type": floor_type,
        "housing_type": str(row.get('Тип жилья', 'Вторичка')),
        "total_area": total_area,
        "calculated_kitchen_area": kitchen_area,
        "calculated_living_area": living_area,
        "kitchen_share_val": kitchen_area / total_area,
        "living_share_val": living_area / total_area,
        "ceil_height": float(row['Высота потолков']) if pd.notna(row.get('Высота потолков')) else 2.7,
        "house_age": float(row['Возраст дома']) if pd.notna(row.get('Возраст дома')) else 10.0,
        "floor": float(floor),
        "total_floors": float(total_floors),
        "dist_center": float(row.get('Расстояние до центра (м)', 4500)),
        "dist_train_station": float(row.get('Расстояние до вокзала (м)', 4500)),
        "dist_airport": float(row.get('Расстояние до аэропорта (м)', 16000)),
        "dist_park": float(row.get('Расстояние до парка (м)', 800)),
        "dist_school_geo": float(row.get('Расстояние до школы (м)', 500)),
        "dist_kinder": float(row.get('Расстояние до детсада (м)', 400)),
        "dist_child_clinic": float(row.get('Расстояние до дет поликлиники (м)', 1200)),
        "dist_additional_edu": float(row.get('Расстояние до доп. образования (м)', 1500)),
        "dist_adult_clinic": float(row.get('Расстояние до взр поликлиники (м)', 4500)),
        "sub_kinder": float(row.get('Кол-во детсадов в радиусе 1 км', 2)),
        "sub_parks": float(row.get('Кол-во парков в радиусе 1 км', 1)),
        "sub_schools": float(row.get('Кол-во школ в радиусе 1 км', 1)),
        "sub_med": float(row.get('Кол-во мед. учреждений в радиусе 1 км', 1))
    }

    try:
        response = requests.post(f"{BASE_API_URL}/api/v1/predict", json=payload)
        if response.status_code == 200:
            pred_data = response.json()
            predicted_price = pred_data["predicted_price"]
        else:
            return None, 0, 0, None
    except:
        return None, 0, 0, None

    # Оценка потенциальной выгоды улучшения ремонта (Второй быстрый запрос к API ИИ)
    remont_upgrade_info = None
    current_remont = payload["renovation"]
    target_remont = "Евроремонт" if current_remont != "Евроремонт" else "Дизайнерский"
    
    if current_remont in ["Без ремонта", "Черновая отделка", "Косметический", "Евроремонт"]:
        payload_up = payload.copy()
        payload_up["renovation"] = target_remont
        try:
            response_up = requests.post(f"{BASE_API_URL}/api/v1/predict", json=payload_up)
            if response_up.status_code == 200:
                predicted_upgrade_price = response_up.json()["predicted_price"]
                added_value = predicted_upgrade_price - predicted_price
                if added_value > 50000:
                    remont_upgrade_info = {
                        "target": target_remont,
                        "added_value": added_value
                    }
        except:
            pass

    real_price = float(row['Цена'])
    price_diff = real_price - predicted_price
    threshold = 1.5 * 350000 # 1.5 * MAE
    
    if price_diff < -threshold:
        status, color = "Цена ниже рынка", "green"
    elif price_diff > threshold:
        status, color = "Цена выше рынка", "red"
    else:
        status, color = "Рыночная цена", "gray"
        
    return status, color, predicted_price, remont_upgrade_info

# Рендеринг HTML-бейджa расстояния до инфраструктуры
def format_distance_badge(meters_val):
    try:
        meters = int(meters_val)
        if meters < 500:
            bg_color, text_color = "#E8F5E9", "#2E7D32"
        elif meters > 3000:
            bg_color, text_color = "#FFEBEE", "#C62828"
        else:
            bg_color, text_color = "#F0F2F6", "#31333F"

        dist_str = f"{meters} м" if meters < 1000 else f"{meters / 1000:.2f} км".replace('.', ',')
        return f"""
        <span style="background-color: {bg_color}; color: {text_color}; padding: 4px 10px; 
              border-radius: 6px; font-size: 13px; font-weight: bold; font-family: sans-serif; white-space: nowrap;">
            {dist_str}
        </span>
        """
    except:
        return '<span style="color: gray; font-size: 13px;">—</span>'

# Получение метаданных
meta = fetch_meta_data()

if meta:
    districts = meta["districts"]
    if "Все районы" not in districts:
        districts.insert(0, "Все районы")

    raw_rooms = meta["rooms"]
    numeric_rooms = []
    string_rooms = []
    for r in raw_rooms:
        if str(r).endswith('.0'):
            r = str(r)[:-2]
        if str(r).isdigit():
            numeric_rooms.append(int(r))
        elif r:  
            string_rooms.append(str(r))

    rooms_options = [f"{r}-комн." for r in sorted(numeric_rooms)] + sorted(string_rooms)
    if "Все" not in rooms_options:
        rooms_options.insert(0, "Все")

    min_price = int(meta["limits"].get("min_price", 0))
    max_price = int(meta["limits"].get("max_price", 100000000))
else:
    districts = ["Все районы"]
    rooms_options = ["Все"]
    min_price, max_price = 0, 100000000

# Логика всплывающего Toast-уведомления
if "toast_shown" not in st.session_state:
    st.session_state.toast_shown = False

if len(st.session_state.compare_list) == 2 and not st.session_state.toast_shown:
    st.toast("Оба объекта выбраны! Пролистайте страницу вниз для сравнения.")
    st.session_state.toast_shown = True

if len(st.session_state.compare_list) < 2:
    st.session_state.toast_shown = False

# Модальное окно с детальной карточкой объекта недвижимости
@st.dialog("Полная информация об объекте", width="middle")
def show_object_details(row):
    price_formatted = f"{int(row['Цена']):,.0f} ₽".replace(",", " ")
    price_m2_formatted = f"{int(row['Цена за кв.м']):,.0f} ₽/м²".replace(",", " ")
    
    st.markdown(f"## {row['Название']}")
    st.markdown(f"# {price_formatted}")
    st.markdown(f"<p style='color: gray; font-size: 16px; margin-top: -15px;'>Цена за квадратный метр: {price_m2_formatted}</p>", unsafe_allow_html=True)
    
    status, color, pred_p, remont_info = check_price_anomaly_via_api(row)
    if status:
        pred_p_formatted = f"{int(pred_p):,.0f} ₽".replace(",", " ")
        st.markdown(
            f"<div style='padding: 10px; border-radius: 5px; background-color: rgba(0,0,0,0.05); margin-bottom: 15px;'>\n"
            f"<b>Оценка:</b> <span style='color:{color}; font-weight:bold;'>{status}</span><br>\n"
            f"<span style='font-size:14px; color:dimgray;'>Прогнозная рыночная стоимость: {pred_p_formatted}</span>\n"
            f"</div>", 
            unsafe_allow_html=True
        )

    if pd.notna(row.get('Микрорайон')) and str(row['Микрорайон']).strip() != "":
        st.markdown(f"**Район:** {row['Район']}")
        st.markdown(f"**Микрорайон:** {row['Микрорайон']}")
    else:
        st.markdown(f"**Район:** {row['Район']}")
    st.markdown(f"**Адрес:** {row['Адрес']}")
    
    if pd.notna(row.get('Ссылка')):
        st.link_button("Открыть объявление на ЦИАН", row['Ссылка'], use_container_width=True, type="primary")
            
    st.markdown("---")
    
    with st.container(border=True):
        infra_items = [
            {"label": "Школа", "name_col": "Ближайшая школа", "dist_col": "Расстояние до школы (м)"},
            {"label": "Детский сад", "name_col": "Ближайший детский сад", "dist_col": "Расстояние до детсада (м)"},
            {"label": "Взрослая поликлиника", "name_col": "Ближайшая взрослая поликлиника", "dist_col": "Расстояние до взр поликлиники (м)"},
            {"label": "Детская поликлиника", "name_col": "Ближайшая детская поликлиника", "dist_col": "Расстояние до дет поликлиники (м)"},
            {"label": "Парк/сквер", "name_col": "Ближайший парк", "dist_col": "Расстояние до park (м)" if 'Расстояние до park (м)' in row else "Расстояние до парка (м)"},
        ]
        st.markdown("### БЛИЖАЙШАЯ ИНФРАСТРУКТУРА")
        html_rows = ""
        for item in infra_items:
            obj_name = row.get(item["name_col"])
            obj_dist = row.get(item["dist_col"])
            obj_name_str = str(obj_name).strip() if pd.notna(obj_name) and str(obj_name).strip() != "" else "Не указано"
            badge_html = format_distance_badge(obj_dist)
            
            html_rows += f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #EDEDED; font-family: sans-serif;">
                <div style="font-weight: 600; color: #31333F; width: 30%; font-size: 14px; flex-shrink: 0;">{item["label"]}</div>
                <div style="color: #444444; width: 50%; padding: 0 10px; font-size: 13px; line-height: 1.3; word-wrap: break-word;">{obj_name_str}</div>
                <div style="width: 20%; flex-shrink: 0; text-align: right;">{badge_html}</div>
            </div>
            """
        st.html(f'<div style="background: #FFFFFF; border-radius: 8px; padding: 5px 0;">{html_rows}</div>')

    with st.container(border=True):
        st.markdown("### ПЛОЩАДЬ")
        st.write(f"**Общая площадь:** {row['Общая площадь']} м²")
        if pd.notna(row.get('Жилая площадь')): st.write(f"**Жилая площадь:** {row['Жилая площадь']} м²")
        if pd.notna(row.get('Площадь кухни')): st.write(f"**Площадь кухни:** {row['Площадь кухни']} м²")
        if pd.notna(row.get('Высота потолков')): st.write(f"**Высота потолков:** {row['Высота потолков']} м")
            
    with st.container(border=True):
        st.markdown("### СОСТОЯНИЕ И ОБУСТРОЙСТВО")
        if pd.notna(row.get('Санузел')): st.write(f"**Санузел:** {row['Санузел']}")
        if pd.notna(row.get('Балкон/лоджия')): st.write(f"**Балкон/лоджия:** {row['Балкон/лоджия']}")
        if pd.notna(row.get('Вид из окон')): st.write(f"**Вид из окон:** {row['Вид из окон']}")
        if pd.notna(row.get('Продаётся с мебелью')): st.write(f"**Продажа с мебелью:** {row['Продаётся с мебелью']}")
        if pd.notna(row.get('Ремонт')): 
            st.write(f"**Ремонт:** {row['Ремонт']}")
            if remont_info:
                added_val_formatted = f"{int(remont_info['added_value']):,.0f} ₽".replace(",", " ")
                st.markdown(
                    f"<div style='margin-top: 10px; color: #0068c9; font-size: 14px; background-color: rgba(0,104,201,0.1); padding: 6px 10px; border-radius: 4px; font-weight: 500;'>"
                    f"Сделав «{remont_info['target']}», стоимость квартиры может увеличиться на <b>{added_val_formatted}</b>"
                    f"</div>", unsafe_allow_html=True
                )

    with st.container(border=True):
        st.markdown("### О ДОМЕ")
        floor_val = int(row['Этаж']) if pd.notna(row['Этаж']) else "?"
        max_floor = int(row['Этажность дома']) if pd.notna(row['Этажность дома']) else "?"
        st.write(f"**Этаж:** {floor_val} из {max_floor}")
        if pd.notna(row.get('Год постройки')): st.write(f"**Год постройки:** {int(row['Год постройки'])} г.")
        if pd.notna(row.get('Тип жилья')): st.write(f"**Тип жилья:** {row['Тип жилья']}")
        if pd.notna(row.get('Отопление')): st.write(f"**Отопление:** {row['Отопление']}")

    with st.container(border=True):
        st.markdown("### ДОСТУПНОСТЬ")
        if pd.notna(row.get('Расстояние до центра (м)')): st.write(f"**До центра города:** {row['Расстояние до center (м)']/1000:.2f} км" if 'Расстояние до center (м)' in row else f"**До центра города:** {row['Расстояние до центра (м)']/1000:.2f} км")
        if pd.notna(row.get('Расстояние до вокзала (м)')): st.write(f"**До вокзала Краснодар-1:** {row['Расстояние до вокзала (м)']/1000:.2f} км")
        if pd.notna(row.get('Расстояние до аэропорта (м)')): st.write(f"**До аэропорта Пашковский:** {row['Расстояние до аэропорта (м)']/1000:.2f} км")

# Фильтры поиска
col_rooms, col_price, col_district = st.columns([1.5, 2.5, 2.5], gap="small")

with col_rooms:
    selected_rooms = st.selectbox("Количество комнат", rooms_options)
with col_price:
    selected_price_range = st.slider("Цена (руб.)", min_value=min_price, max_value=max_price, value=(min_price, max_price), step=500000)
with col_district:
    selected_district = st.selectbox("Район города", districts)

# Запрос данных из API бэкенда вместо локального DataFrame
search_results = search_apartments_api(selected_district, selected_rooms, selected_price_range[0], selected_price_range[1])

if search_results:
    filtered_df = pd.DataFrame(search_results)
    st.markdown("---")
    st.subheader(f"Найдено объектов: {len(filtered_df)}")

    # Постраничный вывод (Пагинация по 10 элементов)
    items_per_page = 10
    total_pages = max(1, int(len(filtered_df) / items_per_page) + (1 if len(filtered_df) % items_per_page > 0 else 0))
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = 1

    col_prev, col_page_num, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("Назад", disabled=(st.session_state.current_page == 1), use_container_width=True):
            st.session_state.current_page -= 1
            st.rerun()
    with col_page_num:
        st.markdown(f"<p style='text-align: center; font-size: 16px; font-weight: bold; padding-top: 5px;'>Страница {st.session_state.current_page} из {total_pages}</p>", unsafe_allow_html=True)
    with col_next:
        if st.button("Вперед", disabled=(st.session_state.current_page == total_pages), use_container_width=True):
            st.session_state.current_page += 1
            st.rerun()

    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_df = filtered_df.iloc[start_idx:end_idx]

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Отрисовка карточек объявлений
    for idx, row in page_df.iterrows():
        price_formatted = f"{int(row['Цена']):,.0f} ₽".replace(",", " ")
        price_m2_formatted = f"{int(row['Цена за кв.м']):,.0f} ₽/м²".replace(",", " ")
        
        with st.container(border=True):
            col_main, col_geo, col_action = st.columns([4.5, 3, 2.5], gap="medium")
            
            with col_main:
                st.markdown(f"##### {row['Название']}")
                st.markdown(f"### {price_formatted} <span style='font-size:13px; font-weight:normal; color:gray;'>({price_m2_formatted})</span>", unsafe_allow_html=True)
                
                quick_features = []
                if pd.notna(row.get('Балкон/лоджия')): quick_features.append(f"В квартире: {row['Балкон/лоджия']}")
                if pd.notna(row.get('Ремонт')): quick_features.append(f"Ремонт: {row['Ремонт']}")
                st.caption(" | ".join(quick_features))
            
            with col_geo:
                st.markdown("<div style='padding-top: 5px;'></div>", unsafe_allow_html=True)
                if pd.notna(row.get('Микрорайон')) and str(row['Микрорайон']).strip() != "":
                    st.markdown(f"**{row['Микрорайон']}**")
                else:
                    st.markdown(f"**{row['Район']} район**")
                st.write(f"<span style='font-size:14px; color:dimgray;'>{row['Адрес']}</span>", unsafe_allow_html=True)
                
                if pd.notna(row.get('Расстояние до центра (м)')):
                    st.caption(f"Расстояние до центра: {row['Расстояние до center (м)']/1000:.1f} км" if 'Расстояние до center (м)' in row else f"Расстояние до центра: {row['Расстояние до центра (м)']/1000:.1f} км")
            
            with col_action:
                obj_id = f"{row['Название']}_{idx}"
                is_checked = obj_id in st.session_state.compare_list
                chosen = st.checkbox("Выбрать для сравнения", key=f"chk_{idx}", value=is_checked)
                
                if chosen and obj_id not in st.session_state.compare_list:
                    if len(st.session_state.compare_list) >= 2:
                        st.warning("Можно сравнивать только 2 объекта одновременно!")
                    else:
                        st.session_state.compare_list.append(obj_id)
                        st.session_state.compare_data[obj_id] = row.to_dict()  # Сохраняем данные во внутренний кэш сессии
                        st.rerun()
                elif not chosen and obj_id in st.session_state.compare_list:
                    st.session_state.compare_list.remove(obj_id)
                    st.session_state.compare_data.pop(obj_id, None)
                    st.rerun()

                if st.button("Подробнее", key=f"btn_search_details_{idx}", use_container_width=True):
                    show_object_details(row)

            st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
          
    # Нижний блок пагинации
    st.markdown("---")
    col_prev_b, col_page_num_b, col_next_b = st.columns([1, 2, 1])
    with col_prev_b:
        if st.button("Назад", disabled=(st.session_state.current_page == 1), use_container_width=True, key="prev_bottom"):
            st.session_state.current_page -= 1
            st.rerun()
    with col_page_num_b:
        st.markdown(f"<p style='text-align: center; font-size: 14px; color: gray; padding-top: 5px;'>Страница {st.session_state.current_page} из {total_pages}</p>", unsafe_allow_html=True)
    with col_next_b:
        if st.button("Вперед", disabled=(st.session_state.current_page == total_pages), use_container_width=True, key="next_bottom"):
            st.session_state.current_page += 1
            st.rerun()
            
else:
    st.warning("Объектов с такими параметрами не найдено. Попробуйте изменить критерии поиска.")


# =========================================================
# БЛОК СРАВНЕНИЯ ОБЪЕКТОВ (ПОЛНОСТЬЮ СОХРАНЕННАЯ ТАБЛИЦА)
# =========================================================
if len(st.session_state.compare_list) == 2:
    st.markdown("---")
    with st.expander("Сравнение выбранных объявлений", expanded=True):
        
        selected_rows = [st.session_state.compare_data[obj_id] for obj_id in st.session_state.compare_list]
        obj1, obj2 = selected_rows[0], selected_rows[1]
        
        def clear_comparison_callback():
            for obj_id in st.session_state.compare_list:
                orig_idx = obj_id.split("_")[-1]
                chk_key = f"chk_{orig_idx}"
                if chk_key in st.session_state:
                    st.session_state[chk_key] = False
            st.session_state.compare_list = []
            st.session_state.compare_data = {}

        st.button("Очистить сравнение", type="secondary", on_click=clear_comparison_callback)

        price1 = f"{int(obj1['Цена']):,}".replace(",", " ") + " ₽"
        price2 = f"{int(obj2['Цена']):,}".replace(",", " ") + " ₽"
        price_m2_1 = f"{int(obj1['Цена за кв.м']):,}".replace(",", " ") + " ₽/м²"
        price_m2_2 = f"{int(obj2['Цена за кв.м']):,}".replace(",", " ") + " ₽/м²"

        def fmt_m(val):
            try: return f"{int(val)} м" if int(val) < 1000 else f"{int(val)/1000:.2f} км".replace('.', ',')
            except: return "—"

        def fmt_h(val):
            return f"{val} м" if pd.notna(val) else "—"

        def fmt_age(val):
            if pd.isna(val): return "—"
            v = int(val)
            if v % 10 == 1 and v % 100 != 11: return f"{v} год"
            elif v % 10 in [2, 3, 4] and v % 100 not in [12, 13, 14]: return f"{v} года"
            else: return f"{v} лет"

        st.html(f"""
        <table style="width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 14px;">
            <thead>
                <tr style="text-align: left;">
                    <th style="padding: 12px; width: 34%; border-bottom: 2px solid #D1D5DB;">Характеристика</th>
                    <th style="padding: 12px; width: 33%; border-bottom: 2px solid #D1D5DB;">Объект №1</th>
                    <th style="padding: 12px; width: 33%; border-bottom: 2px solid #D1D5DB;">Объект №2</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">Название</td>
                    <td style="padding: 10px; font-weight: 500;">{obj1['Название']}</td>
                    <td style="padding: 10px; font-weight: 500;">{obj2['Название']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">Стоимость</td>
                    <td style="padding: 10px;">{price1}</td>
                    <td style="padding: 10px;">{price2}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">Цена за м²</td>
                    <td style="padding: 10px;">{price_m2_1}</td>
                    <td style="padding: 10px;">{price_m2_2}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">Район</td>
                    <td style="padding: 10px;">{obj1.get('Район', '—')}</td>
                    <td style="padding: 10px;">{obj2.get('Район', '—')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">Микрорайон</td>
                    <td style="padding: 10px;">{obj1.get('Микрорайон', '—')}</td>
                    <td style="padding: 10px;">{obj2.get('Микрорайон', '—')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">Ремонт</td>
                    <td style="padding: 10px;">{obj1.get('Ремонт', '—')}</td>
                    <td style="padding: 10px;">{obj2.get('Ремонт', '—')}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">Возраст дома</td>
                    <td style="padding: 10px;">{fmt_age(obj1.get('Возраст дома'))}</td>
                    <td style="padding: 10px;">{fmt_age(obj2.get('Возраст дома'))}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">Высота потолков</td>
                    <td style="padding: 10px;">{fmt_h(obj1.get('Высота потолков'))}</td>
                    <td style="padding: 10px;">{fmt_h(obj2.get('Высота потолков'))}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">До центра города</td>
                    <td style="padding: 10px;">{fmt_m(obj1.get('Расстояние до центра (м)', 0))}</td>
                    <td style="padding: 10px;">{fmt_m(obj2.get('Расстояние до центра (м)', 0))}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">До вокзала Краснодар-1</td>
                    <td style="padding: 10px;">{fmt_m(obj1.get('Расстояние до вокзала (м)', 0))}</td>
                    <td style="padding: 10px;">{fmt_m(obj2.get('Расстояние до вокзала (м)', 0))}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">До аэропорта</td>
                    <td style="padding: 10px;">{fmt_m(obj1.get('Расстояние до аэропорта (м)', 0))}</td>
                    <td style="padding: 10px;">{fmt_m(obj2.get('Расстояние до аэропорта (м)', 0))}</td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">До школы</td>
                    <td style="padding: 10px;">{fmt_m(obj1.get('Расстояние до школы (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj1.get('Ближайшая школа', '')}</span></td>
                    <td style="padding: 10px;">{fmt_m(obj2.get('Расстояние до школы (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj2.get('Ближайшая школа', '')}</span></td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">До детского сада</td>
                    <td style="padding: 10px;">{fmt_m(obj1.get('Расстояние до детсада (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj1.get('Ближайший детский сад', '')}</span></td>
                    <td style="padding: 10px;">{fmt_m(obj2.get('Расстояние до детсада (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj2.get('Ближайший детский сад', '')}</span></td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">До взрослой поликлиники</td>
                    <td style="padding: 10px;">{fmt_m(obj1.get('Расстояние до взр поликлиники (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj1.get('Ближайшая взрослая поликлиника', '')}</span></td>
                    <td style="padding: 10px;">{fmt_m(obj2.get('Расстояние до взр поликлиники (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj2.get('Ближайшая взрослая поликлиника', '')}</span></td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">До детской поликлиники</td>
                    <td style="padding: 10px;">{fmt_m(obj1.get('Расстояние до дет поликлиники (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj1.get('Ближайшая детская поликлиника', '')}</span></td>
                    <td style="padding: 10px;">{fmt_m(obj2.get('Расстояние до дет поликлиники (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj2.get('Ближайшая детская поликлиника', '')}</span></td>
                </tr>
                <tr style="border-bottom: 1px solid #EDEDED;">
                    <td style="padding: 10px; font-weight: bold;">До парка / сквера</td>
                    <td style="padding: 10px;">{fmt_m(obj1.get('Расстояние до парка (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj1.get('Ближайший парк', '')}</span></td>
                    <td style="padding: 10px;">{fmt_m(obj2.get('Расстояние до park (м)', 0) if pd.isna(obj1.get('Расстояние до парка (м)')) else obj2.get('Расстояние до парка (м)', 0))} <br><span style="font-size:11px; color:gray;">{obj2.get('Ближайший парк', '')}</span></td>
                </tr>
            </tbody>
        </table>
        """)