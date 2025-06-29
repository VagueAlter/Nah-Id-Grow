import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA


# Load model
model = joblib.load("rf_model.joblib")
kmeans = joblib.load("kmeans.joblib")
pca = joblib.load("pca.joblib")
metrics = joblib.load("metrics.joblib")
kmeans_metrics = joblib.load("kmeans_metrics.joblib")
scaler = joblib.load("scaler.joblib")

# Judul
st.title("💧 Deteksi Kelayakan Air Minum")
st.write("Prediksi apakah air layak dikonsumsi berdasarkan parameter kimia")

# Nama fitur
features = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# === Interpretasi Klaster ===
def interpretasi_klaster(cluster_label):
    if cluster_label == 0:
        return {
            "judul": "Air Padat Mineral & Keruh",
            "penjelasan": (
                "Air dalam klaster ini memiliki kandungan partikel padat (solids) yang tinggi, "
                "dengan kadar sulfate dan chloramines rendah. Kondisi ini menunjukkan air belum melalui proses "
                "pengolahan kimia dan berasal dari sumber tanah mentah. Meskipun tidak langsung berbahaya, "
                "air ini kurang layak dikonsumsi tanpa penyaringan."
            ),
            "rekomendasi": [
                "Gunakan sistem filtrasi fisik (pasir, arang aktif)",
                "Lakukan uji tambahan terhadap kandungan logam berat",
                "Direkomendasikan untuk keperluan non-konsumsi atau pengolahan lanjut"
            ]
        }

    elif cluster_label == 1:
        return {
            "judul": "Air Terolah Kimia (pH Rendah, Klorinasi Tinggi)",
            "penjelasan": (
                "Air ini memiliki kadar chloramines dan sulfate yang tinggi, dengan pH yang cenderung rendah. "
                "Hal ini mengindikasikan kontaminasi kimia atau sisa proses klorinasi berlebihan. "
                "Tingkat turbiditas juga cukup tinggi, menandakan adanya senyawa tersuspensi."
            ),
            "rekomendasi": [
                "Netralisasi pH menggunakan bahan alkali (misal: NaOH, kapur)",
                "Lakukan filtrasi dan aerasi untuk mengurangi kadar klorin",
                "Sebaiknya tidak digunakan langsung untuk konsumsi"
            ]
        }

    elif cluster_label == 2:
        return {
            "judul": "Air Layak Minum (pH & Kandungan Kimia Seimbang)",
            "penjelasan": (
                "Klaster ini menunjukkan karakteristik air dengan pH dan hardness yang optimal, "
                "serta kandungan kimia (chloramines, organic carbon, sulfate) yang seimbang. "
                "Air dalam klaster ini paling mendekati standar air minum layak konsumsi."
            ),
            "rekomendasi": [
                "Dapat digunakan sebagai air minum",
                "Pantau secara berkala kualitas mikrobiologis",
                "Sosialisasikan sebagai sumber air aman di wilayah setempat"
            ]
        }

    else:
        return {
            "judul": "Klaster Tidak Dikenali",
            "penjelasan": "Label klaster tidak termasuk dalam kategori 0, 1, atau 2.",
            "rekomendasi": ["Periksa kembali proses klasterisasi atau label klaster"]
        }

# Tabs: Upload CSV / Manual Input
tab1, tab2, tab3 = st.tabs(["📄 Upload CSV", "✍️ Input Manual", "📊 Performa Model"])

# TAB 1: UPLOAD CSV
with tab1:
    uploaded_file = st.file_uploader("Upload file CSV dengan kolom fitur lengkap:", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if set(features).issubset(data.columns):
            st.success("✅ Fitur sesuai, memproses prediksi...")

            # === Prediksi
            scaled = scaler.transform(data[features])
            pred = model.predict(scaled)
            data["Potability_Prediction"] = pred
            data["Kelayakan"] = data["Potability_Prediction"].map({1: "✅ Layak Minum", 0: "❌ Tidak Layak"})

            input_pca = pca.transform(scaled)
            clusters = kmeans.predict(input_pca)
            data["Cluster"] = clusters

            # === Tampilkan insight per cluster (sekali saja)
            unique_clusters = sorted(data["Cluster"].unique())
            st.markdown("### 🧠 Interpretasi Klaster:")
            for clust in unique_clusters:
                interpretasi = interpretasi_klaster(clust)
                with st.expander(f"Cluster {clust}: {interpretasi['judul']}", expanded=False):
                    st.write(interpretasi["penjelasan"])
                    st.markdown("**🔧 Rekomendasi:**")
                    for r in interpretasi["rekomendasi"]:
                        st.markdown(f"- {r}")

            # === Tampilkan hasil per baris
            st.markdown("---")
            st.markdown("### 🔍 Hasil Prediksi per Baris:")

            for i, row in data.iterrows():
                cluster_label = row["Cluster"]
                interpretasi = interpretasi_klaster(cluster_label)
                st.markdown(f"**{i+1}. {row['Kelayakan']}**")
                st.markdown(f"Cluster: {cluster_label} – *{interpretasi['judul']}*")



# TAB 2: INPUT MANUAL
with tab2:
    st.subheader("Masukkan Nilai Fitur Air")

    inputs = []
    for feature in features:
        value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
        inputs.append(value)

    if st.button("Prediksi"):
        input_df = pd.DataFrame([inputs], columns=features)

        # Prediksi potability
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        result = "✅ Layak Minum" if pred == 1 else "❌ Tidak Layak Minum"
        st.success(f"Hasil Prediksi: **{result}**")

        # Prediksi cluster KMeans
        # Transformasi input ke PCA space
        input_pca = pca.transform(input_scaled)

        # Prediksi cluster
        cluster_pred = kmeans.predict(input_pca)[0]
        st.info(f"Cluster KMeans: **{cluster_pred}**")

        # Interpretasi klaster
        interpretasi = interpretasi_klaster(cluster_pred)
        st.markdown(f"### 🧠 Interpretasi Klaster: {interpretasi['judul']}")
        st.write(interpretasi["penjelasan"])
        st.markdown("#### 🔧 Rekomendasi:")
        for r in interpretasi["rekomendasi"]:
            st.markdown(f"- {r}")

with tab3:
    st.subheader("📊 Evaluasi Performa Model")

    # === METRIK UTAMA
    report = metrics["classification_report"]
    st.markdown("### 📌 Classification Report")
    st.write(pd.DataFrame(report).transpose())

    # === CONFUSION MATRIX
    st.markdown("### ❗ Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["Tidak Layak", "Layak"], columns=["Pred: Tidak Layak", "Pred: Layak"])
    st.dataframe(cm_df)

    # === FEATURE IMPORTANCE
    st.markdown("### 🌟 Feature Importance (Random Forest)")
    importance = pd.Series(metrics["feature_importance"], index=features).sort_values(ascending=True)

    st.bar_chart(importance)

    # === KMeans Metrics
    st.markdown("### 📎 Evaluasi KMeans (Unsupervised)")

    col1, col2, col3 = st.columns(3)
    col1.metric("📉 Inertia", f"{kmeans_metrics['inertia']:.2f}")
    col2.metric("🧭 Silhouette Score", f"{kmeans_metrics['silhouette_score']:.3f}")
    col3.metric("📐 Davies-Bouldin", f"{kmeans_metrics['davies_bouldin_score']:.3f}")

    st.caption("Catatan: Semakin kecil nilai Davies-Bouldin & Inertia, dan semakin tinggi Silhouette Score, semakin baik performa klaster.")


