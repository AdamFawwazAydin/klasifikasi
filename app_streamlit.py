import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️")

# ======================
# DOWNLOAD MODEL
# ======================
MODEL_PATH = "model_sampah.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model dari Google Drive..."):
        # Pastikan ID File benar dan akses publik aktif
        file_id = "18hc8WHannK0asctqfjgyFiDp2ZxbRsDY" 
        url = f'https://drive.google.com/file/d/18hc8WHannK0asctqfjgyFiDp2ZxbRsDY/view?usp=drive_link'
        gdown.download(url, MODEL_PATH, quiet=False)

# ======================
# LOAD MODEL (SAFE)
# ======================
@st.cache_resource
def load_ml_model():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)

model = load_ml_model()

# ======================
# FUNGSI PREPROCESS & PREDIKSI
# ======================
def process_and_predict(image):
    # Preprocessing
    img = np.array(image.convert('RGB')) # Pastikan 3 channel (RGB)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Prediksi
    pred = model.predict(img)
    confidence = pred[0][0]
    
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
st.write("Aplikasi pendeteksi jenis sampah **Organik** vs **Anorganik** berbasis Deep Learning.")

tab1, tab2 = st.tabs(["📤 Upload File", "📷 Ambil Foto"])

# --- TAB 1: UPLOAD ---
with tab1:
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img_upload = Image.open(uploaded_file)
        st.image(img_upload, caption="Gambar Berhasil Diunggah", use_container_width=True)
        
        with st.spinner("Menganalisis..."):
            label, prob = process_and_predict(img_upload)
            
            st.divider()
            st.subheader(f"Hasil: {label}")
            st.write(f"Tingkat Keyakinan: **{prob:.2f}%**")
            st.progress(int(prob))

# --- TAB 2: KAMERA ---
with tab2:
    camera_image = st.camera_input("Arahkan kamera ke objek sampah")
    if camera_image:
        img_cam = Image.open(camera_image)
        
        with st.spinner("Menganalisis..."):
            label, prob = process_and_predict(img_cam)
            
            st.divider()
            st.subheader(f"Hasil: {label}")
            st.write(f"Tingkat Keyakinan: **{prob:.2f}%**")
            st.progress(int(prob))

# Footer
st.markdown("---")
st.caption("Dibuat untuk keperluan Lab Web Development / Klasifikasi Citra")