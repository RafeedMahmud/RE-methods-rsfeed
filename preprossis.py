import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# 📂 تحميل البيانات من مجلد CICIoT2023
def load_dataset(base_path):
    data = []
    labels = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    df['label'] = folder   # اسم المجلد كـ ground-truth label
                    data.append(df)
    return pd.concat(data, ignore_index=True)

# 🧹 تنظيف البيانات
def preprocess_data(df):
    # إزالة القيم الغير صالحة
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # فصل الميزات عن التصنيفات
    X = df.drop(columns=['label'])
    y = df['label']

    # تحويل التصنيفات إلى أرقام
    le = LabelEncoder()
    y = le.fit_transform(y)

    # معالجة مشكلة عدم التوازن
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # تطبيع البيانات
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # تحويل إلى شكل 3D (samples, features, 1) للـ CNN
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    return X_scaled, y_resampled, le

# 🚀 مثال تشغيل
base_path = "CICIoT2023"
df = load_dataset(base_path)
X, y, label_encoder = preprocess_data(df)

print("✅ Dataset ready for CNN training")
print("Shape:", X.shape, "Labels:", len(np.unique(y)))
