import streamlit as st
import pandas as pd
from datetime import datetime
from components.sidebar import show_sidebar
import requests
import plotly.express as px

BACKEND_URL = "https://ap-plat.onrender.com"

st.set_page_config(
    page_title="Недвижимость Краснодара",
    layout="wide",
    initial_sidebar_state="expanded"
)

show_sidebar()

st.title("АНАЛИТИКА ПО РАЙОНАМ")
st.subheader("Средняя цена за м² по районам")

col1, col2, col3, col4 = st.columns(4)

try:
    response = requests.get(f"{BACKEND_URL}/api/v1/analytics/top-districts")
    # Проверяем, что бэкенд вернул статус 200, иначе вызовется исключение
    response.raise_for_status() 
    top_districts_data = response.json()
        
    for col, data in zip([col1, col2, col3, col4], top_districts_data):
        with col:
            st.metric(
                label=f"**{data['district']}**",
                value=f"{data['avg_price_m2']:,.0f} ₽/м²",
                delta=f"{data['count']} объявл."
            )
except Exception as e:
    st.error(f"Ошибка получения топ-районов с бэкенда: {e}")
    # Если бэкенд ответил, но вернул ошибку (например, 500), покажем её текст
    if 'response' in locals():
        st.error("Детали ответа бэкенда:")
        st.code(response.text)

st.markdown("---")
st.header("Стоимость недвижимости по районам")

try:
    districts_resp = requests.get(f"{BACKEND_URL}/api/v1/analytics/districts-list")
    available_districts = districts_resp.json()
    
    selected_analytic_district = st.selectbox("Выберите район для детального анализа", available_districts)
    
    info_resp = requests.get(f"{BACKEND_URL}/api/v1/analytics/district-info", params={"district": selected_analytic_district})
    district_info = info_resp.json()
    

    bar_data = pd.DataFrame(district_info["bar_data"])
    hist_prices = pd.DataFrame(district_info["hist_prices"], columns=["Цена"])
    table_data = pd.DataFrame(district_info["table_data"])
    
    col_graph1, col_graph2 = st.columns(2, gap="large")
    
    with col_graph1:
        st.subheader("Средняя цена за м² в микрорайонах")
        fig_bar = px.bar(
            bar_data, x='Цена за кв.м', y='Микрорайон', orientation='h', text_auto=',.0f',
            labels={'Цена за кв.м': 'Цена за м² (₽)', 'Микрорайон': ''},
            color='Цена за кв.м', color_continuous_scale='Blues'
        )
        fig_bar.update_layout(showlegend=False, height=450, margin=dict(l=20, r=20, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_graph2:
        st.subheader(f"Распределение стоимости в районе {selected_analytic_district}")
        fig_hist = px.histogram(
            hist_prices, x='Цена', color_discrete_sequence=["#0054A3"],
            labels={'Цена': 'Полная стоимость квартиры (₽)', 'count': 'Количество объектов'},
        )
        fig_hist.update_layout(showlegend=False, height=450, margin=dict(l=20, r=20, t=10, b=10), yaxis_title="Количество объявлений", coloraxis_showscale=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with st.expander("Посмотреть подробную информацию по микрорайонам"):

        table_data['Средняя_цена'] = table_data['Средняя_цена'].map('{:,.0f} ₽'.format).str.replace(',', ' ')
        table_data['Медианная_цена'] = table_data['Медианная_цена'].map('{:,.0f} ₽'.format).str.replace(',', ' ')
        table_data['Средняя_цена_м2'] = table_data['Средняя_цена_м2'].map('{:,.0f} ₽/м²'.format).str.replace(',', ' ')
        
        table_data = table_data.rename(columns={
            'Количество_объектов': 'Количество предложений',
            'Средняя_цена': 'Средняя цена',
            'Медианная_цена': 'Медианная цена',
            'Средняя_цена_м2': 'Средняя цена м²'
        })
        
        st.dataframe(
            table_data[['Микрорайон', 'Количество предложений', 'Средняя цена', 'Медианная цена', 'Средняя цена м²']], 
            use_container_width=True, hide_index=True
        )
except Exception as e:
    st.error(f"Не удалось построить аналитические графики: {e}")