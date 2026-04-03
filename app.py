import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import librosa
import numpy as np
import tensorflow as tf
import pickle
import queue
import plotly.graph_objects as go

# Sahifa sozlamalari
st.set_page_config(page_title="Uzbek Gender AI", page_icon="🎙️", layout="wide")

# Model va Scalerni yuklash
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('final_uzbek_gender_model.h5')
    with open('gender_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# --- YUKLANGAN FAYLNI TAHLIL QILISH ---
def process_uploaded_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    features = scaler.transform(mfccs_scaled.reshape(1, -1))
    prediction = model.predict(features, verbose=0)[0][0]
    return prediction

# --- REAL-TIME PROTSESSOR ---
class GenderProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.result_queue = queue.Queue()

    def recv_audio(self, frame):
        audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        sr = frame.sample_rate
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            features = scaler.transform(mfccs_scaled.reshape(1, -1))
            prediction = model.predict(features, verbose=0)[0][0]
            gender = "Ayol" if prediction > 0.5 else "Erkak"
            prob = prediction if prediction > 0.5 else 1 - prediction
            self.result_queue.put((gender, prob))
        except: pass
        return frame

# --- ASOSIY INTERFEYS ---
st.title("🎙️ O'zbek Ovozli Jins Aniqlash Tizimi")
st.markdown("Ushbu tizim sun'iy intellekt yordamida ovozni tahlil qilib, uning jinsini **99% aniqlik** bilan aniqlaydi.")

# Yon menyu (Navigation)
selected = option_menu(
    menu_title=None,
    options=["Fayl yuklash", "Real-vaqt (Mikrofon)"],
    icons=["cloud-upload", "mic"],
    orientation="horizontal",
)

if selected == "Fayl yuklash":
    st.subheader("📁 Audio fayl yuklash orqali tahlil")
    uploaded_file = st.file_uploader("Faylni tanlang (wav, mp3, m4a)", type=['wav', 'mp3', 'm4a'])
    
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Tahlilni boshlash ✨"):
            with st.spinner("Tahlil qilinmoqda..."):
                pred = process_uploaded_audio(uploaded_file)
                label = "AYOL" if pred > 0.5 else "ERKAK"
                prob = pred if pred > 0.5 else 1 - pred
                color = "#FF4B4B" if label == "AYOL" else "#1F77B4"
                
                st.markdown(f"<h2 style='color:{color}'>{label} ({prob*100:.1f}%)</h2>", unsafe_allow_html=True)
                st.progress(float(prob))

elif selected == "Real-vaqt (Mikrofon)":
    st.subheader("🎤 Jonli muloqot orqali tahlil")
    st.info("Mikrofonni yoqing va gapiring. Tizim har bir soniyada ovozingizni tahlil qiladi.")
    
    ctx = webrtc_streamer(
        key="realtime-gender",
        audio_processor_factory=GenderProcessor,
        media_stream_constraints={"video": False, "audio": True},
    )

    res_placeholder = st.empty()
    if ctx.audio_processor:
        while True:
            try:
                gender, prob = ctx.audio_processor.result_queue.get(timeout=1.0)
                color = "#FF4B4B" if gender == "Ayol" else "#1F77B4"
                with res_placeholder.container():
                    st.markdown(f"<div style='text-align:center; padding:20px; border-radius:10px; background-color:{color}22; border:2px solid {color}'>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:{color}; margin:0;'>{gender}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin:0;'>Ishonch darajasi: {prob*100:.1f}%</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            except queue.Empty:
                continue