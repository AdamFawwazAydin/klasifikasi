import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️", layout="centered")

# ======================
# DOWNLOAD MODEL
# ======================
MODEL_PATH = "model_sampah.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model dari Google Drive..."):
        # Menggunakan ID file dari log kamu
        file_id = "18hc8WHannK0asctqfjgyFiDp2ZxbRsDY"
        url = f'https://drive.google.com/file/d/18hc8WHannK0asctqfjgyFiDp2ZxbRsDY/view?usp=drive_link'
        gdown.download(url, MODEL_PATH, quiet=False)

# ======================
# LOAD MODEL (Sesuai Versi)
# ======================
@st.cache_resource
def load_ml_model():
    import tensorflow as tf
    try:
        # Menggunakan compile=False untuk menghindari masalah kompatibilitas optimizer
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.info("Saran: Pastikan versi TensorFlow di requirements.txt adalah 2.16.1 atau lebih baru.")
        return None

model = load_ml_model()

# ======================
# FUNGSI PREDIKSI
# ======================
def process_and_predict(image):
    # Preprocessing: Pastikan ukuran sesuai input model (150x150)
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Prediksi
    pred = model.predict(img)
    confidence = pred[0][0]
    
    # Jika output sigmoid: close to 1 = Anorganik, close to 0 = Organik
    if confidence > 0.5:
        label = "Anorganik"
        prob = confidence * 100
    else:
        label = "Organik"
        prob = (1 - confidence) * 100
        
    return label, prob

# ======================
# UI INTERFACE
# ======================
st.title("♻️ Klasifikasi Sampah CNN")
st.write("Proyek Semester 8 - Klasifikasi **Organik** vs **Anorganik**")

if model is not None:
    tab1, tab2 = st.tabs(["📤 Upload Gambar", "📷 Ambil Foto"])

    # --- TAB 1: UPLOAD ---
    with tab1:
        uploaded_file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img_upload = Image.open(uploaded_file)
            st.image(img_upload, caption="Gambar Terupload", use_container_width=True)
            
            label, prob = process_and_predict(img_upload)
            
            st.divider()
            st.success(f"### Hasil: {label}")
            st.write(f"Keyakinan Model: **{prob:.2f}%**")
            st.progress(int(prob))

    # --- TAB 2: KAMERA ---
    with tab2:
        camera_image = st.camera_input("Ambil foto objek sampah")
        if camera_image:
            img_cam = Image.open(camera_image)
            
            label, prob = process_and_predict(img_cam)
            
            st.divider()
            st.success(f"### Hasil: {label}")
            st.write(f"Keyakinan Model: **{prob:.2f}%**")
            st.progress(int(prob))
else:
    st.warning("Aplikasi tidak dapat dijalankan karena model gagal dimuat.")

# Footer
st.markdown("---")
st.caption(f"NPM: 50422073")