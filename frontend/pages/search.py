import streamlit as st
import pandas as pd
from components.sidebar import show_sidebar
import numpy as np
import joblib
import os


st.set_page_config(page_title="Поиск недвижимости", layout="wide")
st.title("ПОИСК НЕДВИЖИМОСТИ")
show_sidebar()

@st.cache_resource
def load_ml_package():
    possible_paths = ["models/house_price_model.pkl", "frontend/models/house_price_model.pkl", "../models/house_price_model.pkl"]
    for path in possible_paths:
        if os.path.exists(path): 
            return joblib.load(path)
    return None

ml_package = load_ml_package()


@st.cache_data(ttl=3600)
def load_data():

    possible_paths = ["data/Main_dataset.xlsx", "frontend/data/Main_dataset.xlsx", "../data/Main_dataset.xlsx"]
    for path in possible_paths:
        if os.path.exists(path): 
            return pd.read_excel(path)

    raise FileNotFoundError(f"Не удалось найти Main_dataset.xlsx. Текущая директория: {os.getcwd()}")

df = load_data()

def format_distance(meters_val):
        try:
            meters = int(meters_val)
            if meters < 1000:
                return f"{meters} м"
            else:
                return f"{meters / 1000:.2f} км"
        except (ValueError, TypeError):
            return "Не указано"
        
def check_price_anomaly(row, package):
    if package is None:
        return None, 0, 0, None
    

    if isinstance(package, dict):
        model = package.get("model")
        all_features = package.get("features", [])
        medians = package.get("medians", {})
        mae = package.get("metrics", {}).get("MAE", 350000)
    else:
        model = package
        all_features = model.feature_names_ if hasattr(model, 'feature_names_') else []
        medians = {}
        mae = 350000
    

    total_area = float(row['Общая площадь'])
    kitchen_area = float(row['Площадь кухни']) if pd.notna(row.get('Площадь кухни')) else total_area * 0.2
    living_area = float(row['Жилая площадь']) if pd.notna(row.get('Жилая площадь')) else total_area * 0.55
    

    floor = int(row['Этаж']) if pd.notna(row['Этаж']) else 5
    total_floors = int(row['Этажность дома']) if pd.notna(row['Этажность дома']) else 9
    if floor == 1:
        floor_type = "Первый"
    elif floor == total_floors and total_floors > 1:
        floor_type = "Последний"
    else:
        floor_type = "Другой"

    current_remont = str(row.get('Ремонт', 'Косметический'))


    input_dict = {
        'Район': str(row.get('Район', '')),
        'Микрорайон': str(row.get('Микрорайон', 'Не указано')),
        'Ремонт': current_remont,
        'Комнаты/Планировка': str(row.get('Комнаты/Планировка', '1')),
        'Отопление': str(row.get('Отопление', 'Центральное')),
        'Продаётся с мебелью': str(row.get('Продаётся с мебелью', 'Нет')),
        'Вид из окон': str(row.get('Вид из окон', 'Во двор')),
        'Балкон/лоджия': str(row.get('Балкон/лоджия', 'Нет')),
        'Санузел': str(row.get('Санузел', 'Совмещенный')),
        'Тип этажа': floor_type,
        'Тип жилья': str(row.get('Тип жилья', 'Вторичка')),
        'Общая площадь': total_area,
        'Площадь кухни': kitchen_area,
        'Жилая площадь': living_area,
        'Доля кухни': kitchen_area / total_area,
        'Доля жилой площади': living_area / total_area,
        'Высота потолков': float(row['Высота потолков']) if pd.notna(row.get('Высота потолков')) else float(medians.get('Высота потолков', 2.7)),
        'Возраст дома': float(row['Возраст дома']) if pd.notna(row.get('Возраст дома')) else float(row.get('Возраст дома', 10)),
        'Этаж': float(floor),
        'Этажность дома': float(total_floors),
        'Расстояние до центра (м)': float(row.get('Расстояние до центра (м)', 4500)),
        'Расстояние до вокзала Краснодар-1 (м)': float(row.get('Расстояние до вокзала Краснодар-1 (м)', 4500)),
        'Расстояние до аэропорта (м)': float(row.get('Расстояние до аэропорта (м)', 16000)),
        'Расстояние до парка (м)': float(row.get('Расстояние до парка (м)', 800)),
        'Расстояние до школы (м)': float(row.get('Расстояние до школы (м)', 500)),
        'Расстояние до детсада (м)': float(row.get('Расстояние до детсада (м)', 400)),
        'Расстояние до детской поликлиники (м)': float(row.get('Расстояние до детской поликлиники (м)', 1200)),
        'Расстояние до доп. образования (м)': float(row.get('Расстояние до доп. образования (м)', 1500)),
        'Расстояние до взрослой поликлиники (м)': float(row.get('Расстояние до взрослой поликлиники (м)', 4500)),
        'Кол-во детсадов в радиусе 1 км': float(row.get('Кол-во детсадов в радиусе 1 км', medians.get('Кол-во детсадов в радиусе 1 км', 2))),
        'Кол-во парков в радиусе 1 км': float(row.get('Кол-во парков в радиусе 1 км', medians.get('Кол-во парков в радиусе 1 км', 1))),
        'Кол-во школ в радиусе 1 км': float(row.get('Кол-во школ в радиусе 1 км', medians.get('Кол-во школ в радиусе 1 км', 1))),
        'Кол-во школ доп. образования в радиусе 1 км': 1.0,
        'Кол-во мед. учреждений в радиусе 1 км': float(row.get('Кол-во мед. учреждений в радиусе 1 км', medians.get('Кол-во мед. учреждений в радиусе 1 км', 1)))
    }
    

    if not all_features:
        all_features = list(input_dict.keys())
        

    df_current = pd.DataFrame([input_dict])
    for col in all_features:
        if col not in df_current.columns:
            df_current[col] = medians.get(col, 0)
    df_current = df_current[all_features]
    

    pred_log_curr = model.predict(df_current)
    log_curr = pred_log_curr[0] if isinstance(pred_log_curr, np.ndarray) else pred_log_curr
    predicted_price = np.expm1(log_curr)
    

    remont_upgrade_info = None
    target_remont = "Евроремонт" if current_remont != "Евроремонт" else "Дизайнерский"
    
    if current_remont in ["Без ремонта", "Черновая отделка", "Косметический", "Евроремонт"]:
        input_dict_up = input_dict.copy()
        input_dict_up['Ремонт'] = target_remont
        
        df_upgrade = pd.DataFrame([input_dict_up])
        for col in all_features:
            if col not in df_upgrade.columns:
                df_upgrade[col] = medians.get(col, 0)
        df_upgrade = df_upgrade[all_features]
        
        pred_log_up = model.predict(df_upgrade)
        log_up = pred_log_up[0] if isinstance(pred_log_up, np.ndarray) else pred_log_up
        predicted_upgrade_price = np.expm1(log_up)
        
   
        added_value = predicted_upgrade_price - predicted_price
        if added_value > 50000:  # показываем, только если прирост значительный
            remont_upgrade_info = {
                "target": target_remont,
                "added_value": added_value
            }

  
    real_price = float(row['Цена'])
    price_diff = real_price - predicted_price
    threshold = 1.5 * mae
    
    if price_diff < -threshold:
        status, color = "Цена ниже рынка", "green"
    elif price_diff > threshold:
        status, color = "Цена выше рынка", "red"
    else:
        status, color = "Рыночная цена", "gray"
        
    return status, color, predicted_price, remont_upgrade_info


if not df.empty:
    districts = sorted(df['Район'].dropna().unique().tolist())
    districts.insert(0, "Все районы")

    raw_rooms = df['Комнаты/Планировка'].dropna().astype(str).unique().tolist()
    numeric_rooms = []
    string_rooms = []

    for r in raw_rooms:
        if r.endswith('.0'):
            r = r[:-2]
        if r.isdigit():
            numeric_rooms.append(int(r))
        elif r:  
            string_rooms.append(r)

    rooms_options = [f"{r}-комн." for r in sorted(numeric_rooms)] + sorted(string_rooms)
    rooms_options.insert(0, "Все")

    min_price = int(df['Цена'].min())
    max_price = int(df['Цена'].max())
else:
    districts = ["Все районы"]
    rooms_options = ["Все"]
    min_price, max_price = 0, 100000000

if 'search_clicked' not in st.session_state:
    st.session_state.search_clicked = False


@st.dialog("Полная информация об объекте", width="middle")
def show_object_details(row):
    price_formatted = f"{int(row['Цена']):,.0f} ₽".replace(",", " ")
    price_m2_formatted = f"{int(row['Цена за кв.м']):,.0f} ₽/м²".replace(",", " ")
    
    st.markdown(f"## {row['Название']}")
    st.markdown(f"# {price_formatted}")
    st.markdown(f"<p style='color: gray; font-size: 16px; margin-top: -15px;'>Цена за квадратный метр: {price_m2_formatted}</p>", unsafe_allow_html=True)
    
    # Вызов аналитики модели ИИ
    status, color, pred_p, remont_info = check_price_anomaly(row, ml_package)
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
    
    col_dialog_fav = st.columns(1)[0]
    with col_dialog_fav:
        if st.button("Добавить в избранное", key=f"fav_modal_{row.name}", use_container_width=True, type="primary"):
            pass
            
    st.markdown("---")
    
    with st.container(border=True):
        st.markdown("### ИНФРАСТРУКТУРА")
        
        if pd.notna(row.get('Расстояние до школы (м)')):
            st.write(f"**Ближайшая школа:** {format_distance(row['Расстояние до школы (м)'])}")
            
        if pd.notna(row.get('Расстояние до детсада (м)')):
            st.write(f"**Ближайший детский сад:** {format_distance(row['Расстояние до детсада (м)'])}")
            
        if pd.notna(row.get('Расстояние до взрослой поликлиники (м)')):
            st.write(f"**Ближайшая взрослая поликлиника:** {format_distance(row['Расстояние до взрослой поликлиники (м)'])}")
            
        if pd.notna(row.get('Расстояние до детской поликлиники (м)')):
            st.write(f"**Ближайшая детская поликлиника:** {format_distance(row['Расстояние до детской поликлиники (м)'])}")
            
        if pd.notna(row.get('Расстояние до парка (м)')):
            st.write(f"**Ближайший парк:** {format_distance(row['Расстояние до парка (м)'])}")

    with st.container(border=True):
        st.markdown("### ПЛОЩАДЬ")
        st.write(f"**Общая площадь:** {row['Общая площадь']} м²")
        if pd.notna(row.get('Жилая площадь')): 
            st.write(f"**Жилая площадь:** {row['Жилая площадь']} м²")
        if pd.notna(row.get('Площадь кухни')): 
            st.write(f"**Площадь кухни:** {row['Площадь кухни']} м²")
        if pd.notna(row.get('Высота потолков')): 
            st.write(f"**Высота потолков:** {row['Высота потолков']} м")
            
            
    with st.container(border=True):
        st.markdown("### СОСТОЯНИЕ И ОБУСТРОЙСТВО")
        if pd.notna(row.get('Санузел')): 
            st.write(f"**Санузел:** {row['Санузел']}")
        if pd.notna(row.get('Балкон/лоджия')): 
            st.write(f"**Балкон/лоджия:** {row['Балкон/лоджия']}")
        if pd.notna(row.get('Вид из окон')): 
            st.write(f"**Вид из окон:** {row['Вид из окон']}")
        if pd.notna(row.get('Продаётся с мебелью')): 
            st.write(f"**Продажа с мебелью:** {row['Продаётся с мебелью']}")
        if pd.notna(row.get('Ремонт')): 
            st.write(f"**Ремонт:** {row['Ремонт']}")
            
            if remont_info:
                added_val_formatted = f"{int(remont_info['added_value']):,.0f} ₽".replace(",", " ")
                st.markdown(
                    f"<div style='margin-top: 10px; color: #0068c9; font-size: 14px; "
                    f"background-color: rgba(0,104,201,0.1); padding: 6px 10px; border-radius: 4px; font-weight: 500;'>"
                    f"Сделав «{remont_info['target']}», стоимость квартиры может увеличиться на <b>{added_val_formatted}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown("")

    with st.container(border=True):
        st.markdown("### О ДОМЕ")
        floor = int(row['Этаж']) if pd.notna(row['Этаж']) else "?"
        max_floor = int(row['Этажность дома']) if pd.notna(row['Этажность дома']) else "?"
        st.write(f"**Этаж:** {floor} из {max_floor}")
        if pd.notna(row.get('Год постройки')): 
            st.write(f"**Год постройки:** {int(row['Год постройки'])} г.")
        if pd.notna(row.get('Тип жилья')): 
            st.write(f"**Тип жилья:** {row['Тип жилья']}")
        if pd.notna(row.get('Отопление')): 
            st.write(f"**Отопление:** {row['Отопление']}")



    with st.container(border=True):
        st.markdown("### ДОСТУПНОСТЬ")
        if pd.notna(row.get('Расстояние до центра (м)')): 
            st.write(f"**До центра города:** {row['Расстояние до центра (м)']/1000:.2f} км")
        if pd.notna(row.get('Расстояние до вокзала Краснодар-1 (м)')): 
            st.write(f"**До вокзала Краснодар-1:** {row['Расстояние до вокзала Краснодар-1 (м)']/1000:.2f} км")
        if pd.notna(row.get('Расстояние до аэропорта (м)')): 
            st.write(f"**До аэропорта Пашковский:** {row['Расстояние до аэропорта (м)']/1000:.2f} км")


        
    st.markdown("---")
    if pd.notna(row.get('Ссылка')):
        st.link_button("Открыть объявление на ЦИАН", row['Ссылка'], use_container_width=True, type="primary")


col_rooms, col_price, col_district = st.columns([1.5, 2.5, 2.5], gap="small")

with col_rooms:
    selected_rooms = st.selectbox("Количество комнат", rooms_options)

with col_price:
    selected_price_range = st.slider(
        "Цена (руб.)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=500000
    )

with col_district:
    selected_district = st.selectbox("Район города", districts)

st.session_state.search_clicked = True


if st.session_state.search_clicked and not df.empty:
    filtered_df = df.copy()

    if selected_rooms != "Все":
        filtered_df['rooms_str'] = filtered_df['Комнаты/Планировка'].dropna().astype(str).apply(
            lambda x: x[:-2] if x.endswith('.0') else x
        )
        if "-комн." in selected_rooms:
            rooms_val = selected_rooms.split("-")[0]
            filtered_df = filtered_df[filtered_df['rooms_str'] == rooms_val]
        else:
            filtered_df = filtered_df[filtered_df['rooms_str'] == selected_rooms]
            
        filtered_df = filtered_df.drop(columns=['rooms_str'])

    filtered_df = filtered_df[
        (filtered_df['Цена'] >= selected_price_range[0]) & 
        (filtered_df['Цена'] <= selected_price_range[1])
    ]

    if selected_district != "Все районы":
        filtered_df = filtered_df[filtered_df['Район'] == selected_district]

    st.markdown("---")
    st.subheader(f"Найдено объектов: {len(filtered_df)}")

    if not filtered_df.empty:
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
        
        for idx, row in page_df.iterrows():
            price_formatted = f"{int(row['Цена']):,.0f} ₽".replace(",", " ")
            price_m2_formatted = f"{int(row['Цена за кв.м']):,.0f} ₽/м²".replace(",", " ")
            
            with st.container(border=True):
                col_main, col_geo, col_action = st.columns([4.5, 3, 2.5], gap="medium")
                
                with col_main:
                    st.markdown(f"##### {row['Название']}")
                    st.markdown(
                        f"### {price_formatted} <span style='font-size:13px; font-weight:normal; color:gray;'>({price_m2_formatted})</span>", 
                        unsafe_allow_html=True
                    )
                    
                    quick_features = []
                    if pd.notna(row.get('Балкон/лоджия')): 
                        quick_features.append(f"В квартире: {row['Балкон/лоджия']}")
                    if pd.notna(row.get('Ремонт')): 
                        quick_features.append(f"Ремонт: {row['Ремонт']}")
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
                    st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)
                    if st.button("Подробнее", key=f"btn_details_{idx}", use_container_width=True):
                        show_object_details(row)
                    
                    if st.button("В избранное", key=f"fav_card_{idx}", use_container_width=True, type="primary"):
                        pass
                        
                st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
              
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
else:
    st.info("Данные недоступны или пустой датасет.")