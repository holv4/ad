import os
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Зчитування та очищення CSV-файлів
def read_vhi_from_csv(directory):
    data_frames = []
    for filename in os.listdir(directory):
        if filename.startswith('vhi_id_') and filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            try:
                area_id = int(filename.split('_')[2])
            except (IndexError, ValueError):
                os.remove(filepath)
                continue
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = [re.sub(r'<.*?>', '', line) for line in f if 'N/A' not in line]
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                df = pd.read_csv(filepath, index_col=False, header=1)
                df.columns = df.columns.str.strip()
                df['area_ID'] = area_id

                df['VHI'] = pd.to_numeric(df['VHI'], errors='coerce')
                df = df[df['VHI'].notna() & (df['VHI'] >= 0)]

                df['year'] = df['year'].astype(str).str.extract(r'(\d+)')
                df = df.dropna(subset=['year']).copy()
                df['year'] = df['year'].astype(int)

                data_frames.append(df)
            except Exception as e:
                print(f"[!] Error reading {filename}: {e}")
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

# Завантаження даних
data = read_vhi_from_csv("data")
area_names = {1: "Вінницька", 2: "Волинська", 3: "Дніпропетровська", 4: "Донецька", 5: "Житомирська", 
              6: "Закарпатська", 7: "Запорізька", 8: "Івано-Франківська", 9: "Київська", 10: "Кіровоградська",
              11: "Луганська", 12: "Львівська", 13: "Миколаївська", 14: "Одеська", 15: "Полтавська",
              16: "Рівненська", 17: "Сумська", 18: "Тернопільська", 19: "Харківська", 20: "Херсонська",
              21: "Хмельницька", 22: "Черкаська", 23: "Чернівецька", 24: "Чернігівська", 25: "Республіка Крим"}

data['Область'] = data['area_ID'].map(area_names)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Аналіз індексів VHI, VCI, TCI по регіонах України")

default_state = {
    "selected_index": "VCI",
    "selected_area": "Вінницька",
    "week_range": (1, 52),
    "year_range": (1982, 2024),
    "ascending": False,
    "descending": False,
}

if st.session_state.get("reset_filters", False):
    for key, value in default_state.items():
        st.session_state[key] = value
    st.session_state.reset_filters = False
    st.rerun()

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Layout
col1, col2 = st.columns([1, 3])

# Інтерактивні елементи
with col1:
    indicator = st.selectbox("Оберіть індекс:", ['VCI', 'TCI', 'VHI'], key="selected_index")
    selected_region = st.selectbox("Оберіть область:", sorted(data['Область'].unique()), key="selected_area")
    week_range = st.slider("Інтервал тижнів:", 1, 52, key="week_range")
    year_range = st.slider("Інтервал років:", 1982, 2024, key="year_range")
    
    sort_asc = st.checkbox("Сортувати за зростанням", key="ascending")
    sort_desc = st.checkbox("Сортувати за спаданням", key="descending")

    if sort_asc and sort_desc:
        st.warning("Не можна одночасно обрати сортування за зростанням і спаданням.")
        sort_asc = sort_desc = False


    if st.button("Скинути фільтри"):
        st.session_state["reset_filters"] = True
        st.rerun()


# Фільтрація даних
filtered_data = data[
    (data['Область'] == selected_region) &
    (data['week'].between(week_range[0], week_range[1])) &
    (data['year'].between(year_range[0], year_range[1]))
].copy()

# Сортування
if sort_asc:
    filtered_data = filtered_data.sort_values(by=indicator)
elif sort_desc:
    filtered_data = filtered_data.sort_values(by=indicator, ascending=False)


# Вкладки
with col2:
    tabs = st.tabs(["Таблиця", "Графік", "Порівняння"])

    with tabs[0]:
        st.dataframe(filtered_data)

    with tabs[1]:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=filtered_data, x='week', y=indicator, hue='year', ax=ax)
        ax.set_title(f"{indicator} у {selected_region} по тижнях")
        ax.set_ylabel(indicator)
        ax.set_xlabel("Тиждень")
        st.pyplot(fig)

    with tabs[2]:
        comparison_data = data[
            (data['week'].between(week_range[0], week_range[1])) &
            (data['year'].between(year_range[0], year_range[1]))
        ]
        mean_values = comparison_data.groupby('Область')[indicator].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=mean_values, x='Область', y=indicator, palette="coolwarm", ax=ax2)
        ax2.set_title(f"Порівняння середніх значень {indicator} по областях")
        ax2.tick_params(axis='x', rotation=90)
        st.pyplot(fig2)
