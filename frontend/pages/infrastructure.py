import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from components.sidebar import show_sidebar

st.set_page_config(page_title="Оценка инфраструктуры", layout="wide")
st.title("ОЦЕНКА ИНФРАСТРУКТУРЫ ДЛЯ МИКРОРАЙОНОВ")

show_sidebar()

def score_distance(distance, ideal, max_acceptable):
    if distance <= ideal: return 100
    if distance >= max_acceptable: return 0
    return int(100 * (1 - (distance - ideal) / (max_acceptable - ideal)))

def score_density_vs_city(micro_count, city_avg):
    if micro_count == 0: return 15
    ratio = micro_count / max(1, city_avg)
    if ratio >= 1.5: return 100  
    if ratio >= 1.0: return 85   
    if ratio >= 0.5: return 50   
    return 30


@st.cache_data(ttl=3600)
def load_and_process_data():
    possible_paths = ["data/Main_dataset.xlsx", "frontend/data/Main_dataset.xlsx", "../data/Main_dataset.xlsx"]
    
    df_loaded = pd.DataFrame()
    for path in possible_paths:
        if os.path.exists(path): 
            df_loaded = pd.read_excel(path)
            break
            
    if df_loaded.empty:
        return None, {}, {}, {}

    city_stats = {
        'count_school': df_loaded['Кол-во школ в радиусе 1 км'].median() if 'Кол-во школ в радиусе 1 км' in df_loaded.columns else 1.0,
        'count_kinder': df_loaded['Кол-во детсадов в радиусе 1 км'].median() if 'Кол-во детсадов в радиусе 1 км' in df_loaded.columns else 2.0,
        'count_park': df_loaded['Кол-во парков в радиусе 1 км'].median() if 'Кол-во парков в радиусе 1 км' in df_loaded.columns else 1.0,
        'count_med': df_loaded['Кол-во мед. учреждений в радиусе 1 км'].median() if 'Кол-во мед. учреждений в радиусе 1 км' in df_loaded.columns else 1.0
    }

    districts_list = sorted(df_loaded['Район'].dropna().unique().tolist())
    dist_to_micro = {}
    micro_stats = {}

    for d in districts_list:
        df_dist = df_loaded[df_loaded['Район'] == d]
        micros_in_district = df_dist['Микрорайон'].dropna().unique().tolist()
        dist_to_micro[d] = sorted(micros_in_district)
        
        for m in micros_in_district:
            df_micro = df_dist[df_dist['Микрорайон'] == m]
            
            stats = {
                'dist_center': df_micro['Расстояние до центра (м)'].median() if 'Расстояние до центра (м)' in df_micro.columns else 5000,
                'dist_station': df_micro['Расстояние до вокзала Краснодар-1 (м)'].median() if 'Расстояние до вокзала Краснодар-1 (м)' in df_micro.columns else 6000,
                'dist_airport': df_micro['Расстояние до аэропорта (м)'].median() if 'Расстояние до аэропорта (м)' in df_micro.columns else 15000,
                'dist_school': df_micro['Расстояние до школы (м)'].median() if 'Расстояние до школы (м)' in df_micro.columns else 1200,
                'dist_park': df_micro['Расстояние до парка (м)'].median() if 'Расстояние до парка (м)' in df_micro.columns else 800,
                'dist_kinder': df_micro['Расстояние до детсада (м)'].median() if 'Расстояние до детсада (м)' in df_micro.columns else 600,
                'dist_child_clinic': df_micro['Расстояние до детской поликлиники (м)'].median() if 'Расстояние до детской поликлиники (м)' in df_micro.columns else 1200,
                'dist_adult_clinic': df_micro['Расстояние до взрослой поликлиники (м)'].median() if 'Расстояние до взрослой поликлиники (м)' in df_micro.columns else 1500,
                
                'count_school': df_micro['Кол-во школ в радиусе 1 км'].median() if 'Кол-во школ в радиусе 1 км' in df_loaded.columns else 1,
                'count_kinder': df_micro['Кол-во детсадов в радиусе 1 км'].median() if 'Кол-во детсадов в радиусе 1 км' in df_loaded.columns else 2,
                'count_park': df_micro['Кол-во парков в радиусе 1 км'].median() if 'Кол-во парков в радиусе 1 км' in df_loaded.columns else 1,
                'count_med': df_micro['Кол-во мед. учреждений в радиусе 1 км'].median() if 'Кол-во мед. учреждений в радиусе 1 км' in df_loaded.columns else 1,
            }
            micro_stats[m] = stats


    return districts_list, dist_to_micro, micro_stats, city_stats


districts, district_to_micro, all_micro_stats, global_city_stats = load_and_process_data()


if not districts:
    st.error("Не удалось загрузить датасет Main_dataset.xlsx. Проверьте путь к файлу.")
    st.stop()


col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:
    st.subheader("Выбор параметров")
    
    geo_col1, geo_col2 = st.columns(2)
    with geo_col1:
        district = st.selectbox("Район города", districts, key="input_district")
    with geo_col2:
        available_micros = district_to_micro.get(district, [])
        microdistrict = st.selectbox("Микрорайон", available_micros if available_micros else ["Не указано"], key="input_micro")

    m_stats = all_micro_stats.get(microdistrict, {
        'dist_center': 5000, 'dist_station': 6000, 'dist_airport': 15000, 'dist_school': 1200, 'dist_park': 800,
        'dist_kinder': 600, 'dist_child_clinic': 1000, 'dist_adult_clinic': 1500,
        'count_school': 1, 'count_kinder': 2, 'count_park': 1, 'count_med': 1
    })

    dist_kinder = int(m_stats['dist_kinder'])
    dist_school = int(m_stats['dist_school'])
    count_kinder = float(m_stats['count_kinder'])
    count_school = float(m_stats['count_school'])
    dist_child_clinic = int(m_stats['dist_child_clinic'])
    dist_adult_clinic = int(m_stats['dist_adult_clinic'])
    count_med = float(m_stats['count_med'])
    dist_park = int(m_stats['dist_park'])
    count_park = float(m_stats['count_park'])
    dist_center = int(m_stats['dist_center'])
    dist_station = int(m_stats['dist_station'])
    dist_airport = int(m_stats['dist_airport'])

    st.markdown("---")
    st.subheader("Особенности")
    st.caption("По сравнению со средними показателями по Краснодару  \n(в критериях за показатель 'вокруг домов' берется радиус 1 км)")
    
    m_school = math.floor(count_school)
    c_school = math.floor(global_city_stats['count_school'])
    
    m_kinder = math.floor(count_kinder)
    c_kinder = math.floor(global_city_stats['count_kinder'])
    
    m_park = math.floor(count_park)
    c_park = math.floor(global_city_stats['count_park'])
    
    m_med = math.floor(count_med)
    c_med = math.floor(global_city_stats['count_med'])

    advantages = []
    disadvantages = []

    # 1. Школы
    if m_school > c_school:
        diff = m_school - c_school
        word = "школа" if m_school == 1 else ("школы" if 1 < m_school < 5 else "школ")
        advantages.append(f"<b>Обеспеченность школами:</b> В среднем {m_school} {word} около домов, что на {diff} больше, чем обычно по городу.")
    elif m_school < c_school:
        diff = c_school - m_school
        word = "школа" if m_school == 1 else ("школы" if 1 < m_school < 5 else "школ")
        disadvantages.append(f"<b>Дефицит школ:</b> В среднем {m_school} {word} около домов, что на {diff} меньше общегородского уровня.")

    # 2. Детские сады
    if m_kinder > c_kinder:
        diff = m_kinder - c_kinder
        word = "садик" if m_kinder % 10 == 1 and m_kinder % 100 != 11 else ("садика" if 1 < m_kinder % 10 < 5 and not (11 <= m_kinder % 100 <= 14) else "садиков")
        advantages.append(f"<b>Обилие детских садов:</b> Вокруг домов в среднем {m_kinder} {word}, что превышает средний уровень по городу на {diff}.")
    elif m_kinder < c_kinder:
        diff = c_kinder - m_kinder
        word = "садик" if m_kinder % 10 == 1 and m_kinder % 100 != 11 else ("садика" if 1 < m_kinder % 10 < 5 and not (11 <= m_kinder % 100 <= 14) else "садиков")
        disadvantages.append(f"<b>Нехватка детских садов:</b>В среднем {m_kinder} {word}, что меньше на {diff} общегородского уровня.")

    # 3. Парки
    if m_park > c_park:
        advantages.append(f"<b>Наличие парков:</b> Возле домов среднее число прогулочных зон равно {m_park}.")
    elif m_park < c_park:
        disadvantages.append(f"<b>Недостаток зеленых зон:</b> В микрорайоне практически нет парков.")

    # 4. Медицина
    if m_med > c_med:
        diff = m_med - c_med
        advantages.append(f"<b>Развитая медицинская сеть:</b> Плотность поликлиник выше общегородского значения на {diff}.")
    elif m_med < c_med:
        diff = c_med - m_med
        disadvantages.append(f"<b>Слабое медицинское обеспечение:</b> Количество медицинских учреждений в шаговой доступности уступает среднему по городу на {diff}.")

    if advantages or disadvantages:
        if advantages:
            st.markdown("<h4 style='color: #2E7D32; margin-top: 15px;'>Преимущества микрорайона</h4>", unsafe_allow_html=True)
            for adv in advantages:
                html_card = f"""
                <div style="background-color: #E8F5E9; padding: 12px 16px; border-radius: 6px; margin-bottom: 8px; border-left: 4px solid #2E7D32;">
                    <span style="color: #1B5E20; font-size: 14px; font-family: sans-serif;">{adv}</span>
                </div>
                """
                st.html(html_card)
        
        if disadvantages:
            st.markdown("<h4 style='color: #C62828; margin-top: 20px;'>Недостатки микрорайона</h4>", unsafe_allow_html=True)
            for disadv in disadvantages:
                html_card = f"""
                <div style="background-color: #FFEBEE; padding: 12px 16px; border-radius: 6px; margin-bottom: 8px; border-left: 4px solid #C62828;">
                    <span style="color: #B71C1C; font-size: 14px; font-family: sans-serif;">{disadv}</span>
                </div>
                """
                st.html(html_card)
    else:
        st.markdown("<div style='color: #555; font-style: italic; padding-top: 10px;'>Плотность социальной инфраструктуры в данном микрорайоне полностью сбалансирована и соответствует средним общегородским значениям Краснодара.</div>", unsafe_allow_html=True)


with col_results:
    st.subheader("Рейтинг")
    
    # Расчет суб-индексов
    edu_dist_score = (score_distance(dist_kinder, 400, 2000) + score_distance(dist_school, 500, 2500)) / 2
    edu_density_score = (score_density_vs_city(count_kinder, global_city_stats['count_kinder']) + 
                         score_density_vs_city(count_school, global_city_stats['count_school'])) / 2
    education_index = int(edu_dist_score * 0.4 + edu_density_score * 0.6)
    
    health_dist_score = (score_distance(dist_child_clinic, 800, 3500) + score_distance(dist_adult_clinic, 1000, 4000)) / 2
    health_density_score = score_density_vs_city(count_med, global_city_stats['count_med'])
    health_index = int(health_dist_score * 0.4 + health_density_score * 0.6)
    
    leisure_score = (score_distance(dist_park, 500, 2500) * 0.4 + 
                     score_density_vs_city(count_park, global_city_stats['count_park']) * 0.6)
    transport_score = (score_distance(dist_center, 2000, 16000) + score_distance(dist_station, 3000, 18000)) / 2
    leisure_transport_index = int(leisure_score * 0.5 + transport_score * 0.5)
    
    # Итоговый рейтинг
    total_infrastructure_index = int(education_index * 0.40 + health_index * 0.35 + leisure_transport_index * 0.25)
    
    if total_infrastructure_index >= 75:
        st.success(f"### Общий рейтинг: {total_infrastructure_index} / 100  \nРазвитая инфраструктура")
    elif total_infrastructure_index >= 45:
        st.info(f"### Общий рейтинг: {total_infrastructure_index} / 100  \nСредний уровень комфорта")
    else:
        st.warning(f"### Общий рейтинг: {total_infrastructure_index} / 100  \nДефицит социальных объектов")
        
    st.progress(total_infrastructure_index / 100)
    st.markdown("---")
    
    st.markdown(f"**Индекс образования: {education_index} / 100**")
    st.progress(education_index / 100)
    
    st.markdown(f"**Индекс здравоохранения: {health_index} / 100**")
    st.progress(health_index / 100)
    
    st.markdown(f"**Индекс экологии и досуга: {leisure_transport_index} / 100**")
    st.progress(leisure_transport_index / 100)