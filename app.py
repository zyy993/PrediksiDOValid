# app.py
import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Prediksi DO Mahasiswa", layout="centered")
st.title("🎓 Prediksi DO Mahasiswa")

# ============================
# 📥 Unduh Template CSV
# ============================
st.markdown("### 📄 Unduh Template CSV")
st.info("📌 Kolom yang dibutuhkan: `ipk`, `kehadiran`, `penghasilan_orang_tua`, `motivasi_belajar`, `usia`, `semester`, `beban_sks`")

template_df = pd.DataFrame({
    'ipk': [3.25],
    'kehadiran': [90],
    'penghasilan_orang_tua': [5000000],
    'motivasi_belajar': [4],
    'usia': [21],
    'semester': [4],
    'beban_sks': [20]
})
st.download_button(
    label="📥 Unduh Template CSV",
    data=template_df.to_csv(index=False).encode('utf-8'),
    file_name='template_mahasiswa.csv',
    mime='text/csv',
)

# ============================
# 📤 Upload CSV
# ============================
st.markdown("---")
uploaded_file = st.file_uploader("📤 Unggah file CSV data mahasiswa", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # =======================
        # ✅ Validasi Kolom
        # =======================
        fitur = ['ipk', 'kehadiran', 'penghasilan_orang_tua', 'motivasi_belajar', 'usia', 'semester', 'beban_sks']
        if not all(col in data.columns for col in fitur):
            missing = [col for col in fitur if col not in data.columns]
            st.error(f"❌ Kolom berikut tidak ditemukan: {', '.join(missing)}")
            st.stop()

        # =======================
        # 🧼 Preprocessing
        # =======================
        original_data = data.copy()
        if 'status_do' in data.columns:
            data = data.drop(columns=['status_do'])
            original_data = original_data.drop(columns=['status_do'])

        imputer_num = SimpleImputer(strategy='mean')
        data[fitur] = imputer_num.fit_transform(data[fitur])

        fitur_data = data[fitur]

        # =======================
        # 🔍 Load Model & Selector
        # =======================
        base_dir = os.path.join(os.path.dirname(__file__), 'model')
        model_path = os.path.join(base_dir, 'model_numpy.pkl')
        selector_path = os.path.join(base_dir, 'selector.pkl')

        if not os.path.exists(model_path) or not os.path.exists(selector_path):
            st.error("❌ File model atau selector tidak ditemukan.")
            st.stop()

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(selector_path, 'rb') as f:
            selector = pickle.load(f)

        # =======================
        # 🔎 Transformasi fitur
        # =======================
        fitur_terpilih = selector.transform(fitur_data)

        # =======================
        # 🤖 Prediksi
        # =======================
        prediction = model.predict(fitur_terpilih)
        original_data['Hasil Prediksi'] = ['Tidak DO' if p == 0 else 'DO' for p in prediction]

        # =======================
        # 📊 Tampilkan hasil
        # =======================
        st.subheader("📈 Hasil Prediksi")
        st.dataframe(original_data)

        csv = original_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Prediksi sebagai CSV",
            data=csv,
            file_name='hasil_prediksi_mahasiswa.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"❌ Terjadi error saat prediksi: {e}")
