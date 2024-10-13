import sys
import pandas as pd
from sklearn import preprocessing
import numpy as np

df = pd.read_csv('before_encode.csv')

np.random.seed(42)

# ตั้งค่าการเข้ารหัสของ stdout ให้เป็น utf-8 เพื่อหลีกเลี่ยงปัญหาการเข้ารหัส
sys.stdout.reconfigure(encoding='utf-8')

# สร้าง dictionary สำหรับการเก็บ mapping ที่แน่นอน
static_mapping = {}

# แปลงทุกคอลัมน์ใน DataFrame ให้เป็น string
for col in df.columns:
    df[col] = df[col].astype(str)

    # ถ้ายังไม่มี mapping สำหรับคอลัมน์นี้ ให้สร้างใหม่
    if col not in static_mapping:
        le = preprocessing.LabelEncoder()
        le.fit(df[col])
        static_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# ใช้ static_mapping ในการเข้ารหัส DataFrame
for col, map_dict in static_mapping.items():
    df[col] = df[col].map(map_dict)

del static_mapping['family']

#print(len(static_mapping.keys()))

#print(static_mapping.keys())
# แสดง mapping ที่ถูกตรึง
#for col, map_dict in static_mapping.items():
    #print(f"Static Mapping for {col}: {map_dict}")

#print(static_mapping.keys())