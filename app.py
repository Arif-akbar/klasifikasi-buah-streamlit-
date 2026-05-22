import streamlit as st
import numpy as np
import cv2
import joblib
import plotly.graph_objects as go
from PIL import Image
import time

# --- 1. KONFIGURASI & THEMING ---
st.set_page_config(page_title="Fruit Intel Pro", layout="wide", page_icon="🍎")

# Custom CSS untuk tampilan Light Mode yang Clean
st.markdown("""
    <style>
    /* Background & Font */
    .stApp {
        background-color: #F7F9FC;
        color: #2D3748;
    }
    
    /* Custom Header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3.2rem;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    
    /* Result Card Styling - Soft Glassmorphism */
    .res-card {
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .fresh-bg { 
        background-color: #E6FFFA; 
        color: #234E52;
        border-left: 8px solid #38B2AC;
    }
    .rotten-bg { 
        background-color: #FFF5F5; 
        color: #742A2A;
        border-left: 8px solid #E53E3E;
    }
    
    /* Metric styling */
    .metric-val { font-size: 2.8rem; font-weight: 800; margin: 0; }
    .metric-label { font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; }

    /* Customizing Streamlit Widgets */
    .stButton>button {
        background-color: #4A5568;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2D3748;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        rf = joblib.load('model_fruit_rf.pkl')
        cnn = joblib.load('model_fruit_cnn.pkl')
        return rf, cnn
    except:
        return None, None

rf_model, cnn_model = load_models()

# --- 3. HELPER FUNCTIONS ---
CLASS_NAMES = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']

def preprocess_cnn(image_pil):
    img = np.array(image_pil.convert('RGB'))
    img_res = cv2.resize(img, (224, 224))
    return np.expand_dims((img_res.astype(np.float32) / 127.5) - 1.0, axis=0)

# --- 4. HEADER ---
st.markdown('<h1 class="main-header">Fruit Intel <span style="color:#38B2AC">2.0</span></h1>', unsafe_allow_html=True)
st.markdown("<p style='font-size:1.1rem; color:#4A5568;'>Sistem Klasifikasi Kualitas Buah Berbasis Deep Learning</p>", unsafe_allow_html=True)
st.divider()

if rf_model is None or cnn_model is None:
    st.warning("⚠️ File model (.pkl) tidak terdeteksi. Pastikan Anda telah menjalankan proses training.")
    st.stop()

# --- 5. MAIN CONTENT ---
uploaded_file = st.file_uploader("📂 Unggah foto buah di sini...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img_pil = Image.open(uploaded_file)
    img_np = np.array(img_pil.convert('RGB'))
    
    col_img, col_chart = st.columns([1, 1.2])
    
    with col_img:
        st.markdown("### 📸 Pratinjau Gambar")
        st.image(img_pil, use_container_width=True)
        
    with col_chart:
        st.markdown("### 📊 Analisis Spektrum Warna")
        fig = go.Figure()
        colors = ['#E53E3E', '#38A169', '#3182CE'] # Red, Green, Blue
        for i, (channel, color) in enumerate(zip(['Merah', 'Hijau', 'Biru'], colors)):
            hist = np.histogram(img_np[:,:,i], bins=256, range=(0,255))[0]
            fig.add_trace(go.Scatter(y=hist, name=channel, line=dict(color=color, width=2.5)))
        
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🚀 MULAI DIAGNOSA AI", use_container_width=True):
        with st.spinner("Menganalisis data piksel..."):
            time.sleep(0.8)
            in_cnn = preprocess_cnn(img_pil)
            probs = cnn_model.predict(in_cnn)[0]
            idx = np.argmax(probs)
            conf = probs[idx]
            
            is_fresh = idx < 3
            bg_class = "fresh-bg" if is_fresh else "rotten-bg"
            
            # --- HASIL PREDIKSI ---
            st.markdown(f"""
                <div class="res-card {bg_class}">
                    <p class="metric-label">Hasil Identifikasi</p>
                    <h1 class="metric-val">{CLASS_NAMES[idx]}</h1>
                    <p style="font-size:1.1rem; font-weight:500;">Tingkat Keyakinan: {conf*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.progress(float(conf))
                if is_fresh:
                    st.info(f"💡 **Informasi:** Buah {CLASS_NAMES[idx].split()[-1]} terlihat segar dan siap didistribusikan.")
                else:
                    st.warning(f"💡 **Informasi:** Deteksi menunjukkan tanda pembusukan pada {CLASS_NAMES[idx].split()[-1]}.")
            
            with c2:
                st.markdown("**Detail Teknis:**")
                st.caption(f"Intensitas Cahaya: {np.mean(img_np):.2f}")
                st.caption(f"Kategori Model: CNN MobileNetV2")

else:
    st.info("👋 Selamat datang! Silakan unggah gambar buah untuk memulai analisis otomatis.")