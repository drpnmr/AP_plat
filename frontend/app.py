import streamlit as st
import pandas as pd
from datetime import datetime
from components.sidebar import show_sidebar
import os


st.set_page_config(
    page_title="Недвижимость Краснодара",
    layout="wide",
    initial_sidebar_state="expanded"
)

show_sidebar()


st.title("АНАЛИТИКА ПО РАЙОНАМ")


@st.cache_data(ttl=3600)
def load_data():
    possible_paths = ["data/Main_dataset.xlsx", "frontend/data/Main_dataset.xlsx", "../data/Main_dataset.xlsx"]
    for path in possible_paths:
        if os.path.exists(path): 
            return pd.read_excel(path)
    raise FileNotFoundError(f"Не удалось найти Main_dataset.xlsx. Текущая директория: {os.getcwd()}")

df = load_data()

st.subheader("Средняя цена за м² по районам")

col1, col2, col3, col4 = st.columns(4)

districts = df['Район'].dropna().unique()[:4] 

for col, district in zip([col1, col2, col3, col4], districts):
    with col:
        district_df = df[df['Район'] == district]
        if not district_df.empty:
            avg_price_m2 = district_df['Цена за кв.м'].mean()
            count = len(district_df)
            
            st.metric(
                label=f"**{district}**",
                value=f"{avg_price_m2:,.0f} ₽/м²",
                delta=f"{count} объявл."
            )
        else:
            st.metric(label=district, value="—")


st.markdown("---")
st.header("Стоимость недвижимости по районам")


sub_df = df[df['Район'].notna() & (df['Район'].str.strip() != "")]

if not sub_df.empty:

    available_districts = sorted(sub_df['Район'].unique().tolist())
    selected_analytic_district = st.selectbox(
        "", 
        available_districts
    )
    
    district_sub_df = sub_df[sub_df['Район'] == selected_analytic_district]
    
    avg_price_microraion = (
        district_sub_df.groupby('Microdistrict')['Цена за кв.м'] if 'Microdistrict' in district_sub_df else district_sub_df.groupby('Микрорайон')['Цена за кв.м']
    ).mean().reset_index()
    
    if 'Microdistrict' in avg_price_microraion.columns:
        avg_price_microraion = avg_price_microraion.rename(columns={'Microdistrict': 'Микрорайон'})
        
    avg_price_microraion = avg_price_microraion.sort_values(by='Цена за кв.м', ascending=False)
    
    col_graph1, col_graph2 = st.columns(2, gap="large")
    
    with col_graph1:
        st.subheader("Средняя цена за м² в микрорайонах")
        
        import plotly.express as px
        
        fig_bar = px.bar(
            avg_price_microraion,
            x='Цена за кв.м',
            y='Микрорайон',
            orientation='h',
            text_auto=',.0f',
            labels={'Цена за кв.м': 'Цена за м² (₽)', 'Микрорайон': ''},
            color='Цена за кв.м',
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(
            showlegend=False, 
            height=450, 
            margin=dict(l=20, r=20, t=10, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_graph2:
        st.subheader(f"Распределение стоимости в районе {selected_analytic_district}")
        
        fig_hist = px.histogram(
            district_sub_df,
            x='Цена',
            color_discrete_sequence=["#002950"],
            labels={'Цена': 'Полная стоимость квартиры (₽)', 'count': 'Количество объектов'},
        )
        fig_hist.update_layout(
            showlegend=False, 
            height=450, 
            margin=dict(l=20, r=20, t=10, b=10),
            yaxis_title="Количество объявлений",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with st.expander(f"Посмотреть подробную информацию по микрорайонам"):
        group_col = 'Microdistrict' if 'Microdistrict' in district_sub_df.columns else 'Микрорайон'
        
        stats_df = district_sub_df.groupby(['Район', group_col]).agg(
            Количество_объектов=('Цена', 'count'),
            Средняя_цена=('Цена', 'mean'),
            Медианная_цена=('Цена', 'median'),
            Средняя_цена_м2=('Цена за кв.м', 'mean')
        ).reset_index()
        
        if group_col == 'Microdistrict':
            stats_df = stats_df.rename(columns={'Microdistrict': 'Микрорайон'})
            
        stats_df = stats_df.sort_values(by='Средняя_цена_м2', ascending=False)
        
        stats_df['Средняя_цена'] = stats_df['Средняя_цена'].map('{:,.0f} ₽'.format).str.replace(',', ' ')
        stats_df['Медианная_цена'] = stats_df['Медианная_цена'].map('{:,.0f} ₽'.format).str.replace(',', ' ')
        stats_df['Средняя_цена_м2'] = stats_df['Средняя_цена_м2'].map('{:,.0f} ₽/м²'.format).str.replace(',', ' ')
        
        stats_df = stats_df.rename(columns={
            'Количество_объектов': 'Количество предложений',
            'Средняя_цена': 'Средняя цена',
            'Медианная_цена': 'Медианная цена',
            'Средняя_цена_м2': 'Средняя цена м²'
        })
        
        st.dataframe(
            stats_df[['Микрорайон', 'Количество предложений', 'Средняя цена', 'Медианная цена', 'Средняя цена м²']], 
            use_container_width=True,
            hide_index=True
        )
else:
    st.warning("В датасете отсутствуют необходимые данные для построения аналитики.")
   

