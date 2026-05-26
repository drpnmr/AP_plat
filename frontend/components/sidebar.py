import streamlit as st
from datetime import datetime

def show_sidebar():
    # Скрываем стандартное меню страниц
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("АНАЛИТИКА НЕДВИЖИМОСТИ КРАСНОДАРА")
        st.markdown("---")

        if st.button("Аналитика по районам", use_container_width=True):
            st.switch_page("app.py")

        if st.button("Просмотр объявлений", use_container_width=True):
            st.switch_page("pages/search.py")


        if st.button("Предсказание стоимости", use_container_width=True):
            st.switch_page("pages/predict.py")

        if st.button("Оценка инфраструктуры", use_container_width=True):
            st.switch_page("pages/infrastructure.py")

