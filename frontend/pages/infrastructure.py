import streamlit as st
import requests
import math
from components.sidebar import show_sidebar

BACKEND_URL = "http://127.0.0.1:8000"

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
def load_geo_meta():
    """Загружает структуру распределения районов и микрорайонов из единой мета-информации бэкенда"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/apartments/meta", timeout=5)
        if response.status_code == 200:
            data = response.json()
            geo_list = data.get("geo_data", [])
            
            districts_set = set()
            dist_to_micro_map = {}
            
            for item in geo_list:
                d = item.get("Район")
                m = item.get("Микрорайон")
                if d and m:
                    d_str = str(d).strip()
                    m_str = str(m).strip()
                    districts_set.add(d_str)
                    if d_str not in dist_to_micro_map:
                        dist_to_micro_map[d_str] = []
                    if m_str not in dist_to_micro_map[d_str]:
                        dist_to_micro_map[d_str].append(m_str)
                        
            return sorted(list(districts_set)), {k: sorted(v) for k, v in dist_to_micro_map.items()}
    except Exception as e:
        st.error(f"Ошибка загрузки географических метаданных: {e}")
    return [], {}

districts, district_to_micro = load_geo_meta()

if not districts:
    st.error("Критическая ошибка: не удалось получить список районов с бэкенда. Проверьте работу сервиса.")
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

    m_stats = {}
    global_city_stats = {}
    
    try:
        res = requests.get(f"{BACKEND_URL}/api/v1/analytics/infrastructure", params={"district": district, "microdistrict": microdistrict}, timeout=5)
        if res.status_code == 200:
            res_data = res.json()
            m_stats = res_data.get("micro_stats", {})
            global_city_stats = res_data.get("city_stats", {})
    except Exception as e:
        st.error(f"Ошибка получения инфраструктурных данных: {e}")

    if not m_stats or not global_city_stats:
        m_stats = {
            'dist_center': 5000, 'dist_station': 6000, 'dist_airport': 15000, 'dist_school': 1200, 'dist_park': 800,
            'dist_kinder': 600, 'dist_child_clinic': 1000, 'dist_adult_clinic': 1500,
            'count_school': 1, 'count_kinder': 2, 'count_park': 1, 'count_med': 1
        }
        global_city_stats = {'count_school': 1, 'count_kinder': 2, 'count_park': 1, 'count_med': 1}

    dist_kinder = int(m_stats.get('dist_kinder', 600))
    dist_school = int(m_stats.get('dist_school', 1200))
    count_kinder = float(m_stats.get('count_kinder', 2))
    count_school = float(m_stats.get('count_school', 1))
    dist_child_clinic = int(m_stats.get('dist_child_clinic', 1000))
    dist_adult_clinic = int(m_stats.get('dist_adult_clinic', 1500))
    count_med = float(m_stats.get('count_med', 1))
    dist_park = int(m_stats.get('dist_park', 800))
    count_park = float(m_stats.get('count_park', 1))
    dist_center = int(m_stats.get('dist_center', 5000))
    dist_station = int(m_stats.get('dist_station', 6000))
    dist_airport = int(m_stats.get('dist_airport', 15000))

    st.markdown("---")
    st.subheader("Особенности")
    st.caption("По сравнению со средними показателями по Краснодару  \n(в критериях за показатель 'вокруг домов' берется радиус 1 км)")
    
    m_school = math.floor(count_school)
    c_school = math.floor(global_city_stats.get('count_school', 1))
    
    m_kinder = math.floor(count_kinder)
    c_kinder = math.floor(global_city_stats.get('count_kinder', 2))
    
    m_park = math.floor(count_park)
    c_park = math.floor(global_city_stats.get('count_park', 1))
    
    m_med = math.floor(count_med)
    c_med = math.floor(global_city_stats.get('count_med', 1))

    advantages = []
    disadvantages = []

    if m_school > c_school:
        diff = m_school - c_school
        word = "школа" if m_school == 1 else ("школы" if 1 < m_school < 5 else "школ")
        advantages.append(f"<b>Обеспеченность школами:</b> В среднем {m_school} {word} около домов, что на {diff} больше, чем обычно по городу.")
    elif m_school < c_school:
        diff = c_school - m_school
        word = "школа" if m_school == 1 else ("школы" if 1 < m_school < 5 else "школ")
        disadvantages.append(f"<b>Дефицит школ:</b> В среднем {m_school} {word} около домов, что на {diff} меньше общегородского уровня.")

    if m_kinder > c_kinder:
        diff = m_kinder - c_kinder
        word = "садик" if m_kinder % 10 == 1 and m_kinder % 100 != 11 else ("садика" if 1 < m_kinder % 10 < 5 and not (11 <= m_kinder % 100 <= 14) else "садиков")
        advantages.append(f"<b>Обилие детских садов:</b> Вокруг домов в среднем {m_kinder} {word}, что превышает средний уровень по городу на {diff}.")
    elif m_kinder < c_kinder:
        diff = c_kinder - m_kinder
        word = "садик" if m_kinder % 10 == 1 and m_kinder % 100 != 11 else ("садика" if 1 < m_kinder % 10 < 5 and not (11 <= m_kinder % 100 <= 14) else "садиков")
        disadvantages.append(f"<b>Нехватка детских садов:</b> В среднем {m_kinder} {word}, что меньше на {diff} общегородского уровня.")

    if m_park > c_park:
        advantages.append(f"<b>Наличие парков:</b> Возле домов среднее число прогулочных зон равно {m_park}.")
    elif m_park < c_park:
        disadvantages.append(f"<b>Недостаток зеленых зон:</b> В микрорайоне практически нет парков.")

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
    
    edu_dist_score = (score_distance(dist_kinder, 400, 2000) + score_distance(dist_school, 500, 2500)) / 2
    edu_density_score = (score_density_vs_city(count_kinder, global_city_stats.get('count_kinder', 2)) + 
                         score_density_vs_city(count_school, global_city_stats.get('count_school', 1))) / 2
    education_index = int(edu_dist_score * 0.4 + edu_density_score * 0.6)
    
    health_dist_score = (score_distance(dist_child_clinic, 800, 3500) + score_distance(dist_adult_clinic, 1000, 4000)) / 2
    health_density_score = score_density_vs_city(count_med, global_city_stats.get('count_med', 1))
    health_index = int(health_dist_score * 0.4 + health_density_score * 0.6)
    
    leisure_score = (score_distance(dist_park, 500, 2500) * 0.4 + 
                     score_density_vs_city(count_park, global_city_stats.get('count_park', 1)) * 0.6)
    transport_score = (score_distance(dist_center, 2000, 16000) + score_distance(dist_station, 3000, 18000)) / 2
    leisure_transport_index = int(leisure_score * 0.5 + transport_score * 0.5)
    
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