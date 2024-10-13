import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import encoder as en
from fuzzywuzzy import process

# โหลดโมเดลที่ได้รับการฝึกมา
loaded_rf_classifier = joblib.load('model_now.joblib')
loaded_predict_cheese_model = joblib.load('predict_cheese_model.joblib')

df = pd.read_csv('cheese_encode_input.csv')
cheese_mapping_df = pd.read_csv('convert_cheese.csv')

# Function to fill NaN in the sample data based on nearest neighbors
def fill_missing_values(sample_data, df):
    known_columns = ~np.isnan(sample_data)
    known_values = sample_data[known_columns]
    columns_to_use = np.where(known_columns)[0]
    df_relevant = df.iloc[:, columns_to_use]
    nearest_neighbors = NearestNeighbors(n_neighbors=1)
    nearest_neighbors.fit(df_relevant)
    _, indices = nearest_neighbors.kneighbors([known_values])
    closest_row = df.iloc[indices[0][0]].values
    filled_sample = sample_data.copy()
    nan_columns = np.isnan(filled_sample)
    filled_sample[nan_columns] = closest_row[nan_columns]
    return filled_sample

# ฟังก์ชัน fuzzy matching
def get_closest_match(value, valid_values):
    if isinstance(value, str):
        if value in valid_values:
            return value
        for valid_value in valid_values:
            if set(valid_value.split(', ')) == set(value.split(', ')):
                return valid_value
        matches = process.extract(value, valid_values, limit=None)
        best_match = None
        best_score = 0
        for match, score in matches:
            if score > best_score:
                best_score = score
                best_match = match
        return best_match if best_score >= 90 else None
    return None

# เริ่มสร้าง UI ด้วย Streamlit
st.title("Cheese Prediction System")

# รับข้อมูลจากผู้ใช้
input_data = []
milk = st.text_input('Milk Type:')
country = st.text_input('Country:')
cheese_type = st.text_input('Cheese Type:')
fat_content = st.text_input('Fat Content:')
texture = st.text_input('Texture:')
rind = st.text_input('Rind:')
color = st.text_input('Color:')
flavor = st.text_input('Flavor:')
aroma = st.text_input('Aroma:')
vegetarian = st.selectbox('Vegetarian:', ['Yes', 'No'])
vegan = st.selectbox('Vegan:', ['Yes', 'No'])

if st.button("Predict Cheese Class"):
    # ดึงข้อมูล input
    input_data = [milk, country, cheese_type, fat_content, texture, rind, color, flavor, aroma, vegetarian, vegan]
    
    encoded_input = []
    index_key_mapping = ['milk', 'country', 'type', 'fat_content', 'texture', 'rind', 'color', 'flavor', 'aroma', 'vegetarian', 'vegan']
    
    # ใช้ static mapping และ fuzzy matching
    for i, value in enumerate(input_data):
        if i < len(index_key_mapping):
            col_name = index_key_mapping[i]
            if value is None:
                encoded_input.append(None)
            elif any(key.lower() == value.lower() and len(key) == len(value) for key in en.static_mapping[col_name].keys()):
                exact_key = next(key for key in en.static_mapping[col_name].keys() if key.lower() == value.lower() and len(key) == len(value))
                encoded_input.append(en.static_mapping[col_name][exact_key])
            else:
                closest_match = get_closest_match(value, en.static_mapping[col_name].keys())
                if closest_match:
                    encoded_input.append(en.static_mapping[col_name][closest_match])
                else:
                    encoded_input.append(None)

    # ตรวจสอบและเติมค่า missing values
    encoded_input = np.array(encoded_input, dtype=object).reshape(1, -1)
    filled_input = fill_missing_values(encoded_input[0], df)
    filled_input = np.array(filled_input, dtype=float).reshape(1, -1)

    # ทำการพยากรณ์
    predictions = loaded_rf_classifier.predict(filled_input)

    # แปลงผลลัพธ์เป็นชื่อ class
    class_names = ['Cheddar', 'Blue', 'Brie', 'Pecorino', 'Gouda', 'Parmesan', 'Camembert', 'Feta', 'Cottage', 'Pasta filata', 'Swiss Cheese', 'Mozzarella', 'Tomme']
    predictions_index = [3, 0, 1, 10, 6, 8, 2, 5, 4, 9, 11, 7, 12]
    predicted_class_names = [class_names[predictions_index.index(pred)] for pred in predictions]

    # แสดงผลลัพธ์
    st.write(f"Predicted Cheese Class: {predicted_class_names[0]}")
