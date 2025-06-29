import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("rf_model.joblib")

# Judul
st.title("💧 Deteksi Kelayakan Air Minum")
st.write("Prediksi apakah air layak dikonsumsi berdasarkan parameter kimia")

# Nama fitur
features = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# Tabs: Upload CSV / Manual Input
tab1, tab2 = st.tabs(["📄 Upload CSV", "✍️ Input Manual"])

# =========================
# 📄 TAB 1: UPLOAD CSV
# =========================
with tab1:
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom fitur lengkap:", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        if set(features).issubset(data.columns):
            st.success("✅ Fitur sesuai, memproses prediksi...")
            pred = model.predict(data[features])
            data["Potability_Prediction"] = pred
            data["Kelayakan"] = data["Potability_Prediction"].map({1: "Layak Minum", 0: "Tidak Layak"})
            st.write(data)
        else:
            st.error("❌ CSV tidak mengandung semua fitur yang dibutuhkan.")

# =========================
# ✍️ TAB 2: INPUT MANUAL
# =========================
with tab2:
    st.subheader("Masukkan Nilai Fitur Air")

    inputs = []
    for feature in features:
        value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
        inputs.append(value)

    if st.button("Prediksi"):
        input_df = pd.DataFrame([inputs], columns=features)
        pred = model.predict(input_df)[0]
        result = "✅ Layak Minum" if pred == 1 else "❌ Tidak Layak Minum"
        st.success(f"Hasil Prediksi: **{result}**")
