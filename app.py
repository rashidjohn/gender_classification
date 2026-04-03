import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import librosa
import numpy as np
import tensorflow as tf
import pickle
import queue
import time

# --- SAHIFA SOZLAMALARI ---
st.set_page_config(page_title="AI Gender Recognition", page_icon="🎙️", layout="centered")

# Dizayn uchun maxsus CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .result-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODELNI YUKLASH (KESHLANGAN) ---
@st.cache_resource
def load_resources():
    try:
        # Model va Scaler nomlarini o'z fayllaringizga moslang
        model = tf.keras.models.load_model('final_uzbek_gender_model.h5')
        with open('gender_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Resurslarni yuklashda xatolik: {e}")
        return None, None

model, scaler = load_resources()

# --- AUDIO PROTSESSOR KLASSI ---
class GenderAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.result_queue = queue.Queue()

    def recv_audio(self, frame):
        # Audio signallarni normallashtirish
        audio = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        sr = frame.sample_rate
        
        try:
            # 1. Feature extraction (MFCC)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
            
            # 2. Prediction
            if model and scaler:
                features = scaler.transform(mfccs_processed)
                prediction = model.predict(features, verbose=0)[0][0]
                
                gender = "Ayol" if prediction > 0.5 else "Erkak"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                
                # Natijani navbatga (queue) qo'shish
                self.result_queue.put((gender, confidence))
        except Exception:
            pass # Shovqin yoki xatolik bo'lsa o'tkazib yuborish
        
        return frame

# --- ASOSIY INTERFEYS ---
st.title("🎙️ O'zbek Ovozli Jins Aniqlash")
st.caption("Sun'iy intellekt yordamida real vaqtda ovoz tahlili")

tab_choice = option_menu(
    menu_title=None,
    options=["Fayl yuklash", "Jonli Mikrofon"],
    icons=["cloud-upload", "mic"],
    orientation="horizontal",
    styles={"container": {"padding": "0!important", "background-color": "#fafafa"}}
)

# STUN serverlar (ulanish barqarorligi uchun)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

if tab_choice == "Fayl yuklash":
    uploaded_file = st.file_uploader("Audio faylni tanlang (MP3, WAV)", type=['wav', 'mp3', 'm4a'], label_visibility="collapsed")
    
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Tahlilni boshlash"):
            with st.spinner("Tahlil qilinmoqda..."):
                try:
                    y, sr = librosa.load(uploaded_file, sr=16000)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)
                    
                    features = scaler.transform(mfccs_scaled)
                    pred = model.predict(features, verbose=0)[0][0]
                    
                    res_label = "AYOL" if pred > 0.5 else "ERKAK"
                    res_prob = pred if pred > 0.5 else 1 - pred
                    color = "#FF4B4B" if res_label == "AYOL" else "#1F77B4"
                    
                    st.markdown(f"""
                        <div class="result-box" style="border: 2px solid {color}; background-color: {color}10;">
                            <h2 style="color: {color};">{res_label}</h2>
                            <p>Ishonch darajasi: {res_prob*100:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Faylni o'qishda xatolik: {e}")

else:
    st.info("START tugmasini bosing va gapiring. Tizim avtomatik tahlil qiladi.")
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="gender-streamer-pro",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=GenderAudioProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

    output_placeholder = st.empty()

    # Xatoliklarning oldini olish uchun asosiy tekshiruv
    if webrtc_ctx.state.playing:
        while True:
            # 1. Protsessor yaratilganini tekshirish
            if webrtc_ctx.audio_processor is not None:
                try:
                    # 2. Queue'dan ma'lumot olish (timeout xatolikni oldini oladi)
                    gender, prob = webrtc_ctx.audio_processor.result_queue.get(timeout=2.0)
                    
                    bg_color = "#FF4B4B" if gender == "Ayol" else "#1F77B4"
                    
                    with output_placeholder.container():
                        st.markdown(f"""
                            <div class="result-box" style="border: 2px solid {bg_color}; background-color: {bg_color}15;">
                                <h1 style="color: {bg_color};">{gender.upper()}</h1>
                                <h3>Aniqlik: {prob*100:.1f}%</h3>
                                <small>Jonli tahlil rejimi faol</small>
                            </div>
                        """, unsafe_allow_html=True)
                except queue.Empty:
                    # Hali ma'lumot kelmagan bo'lsa tsiklni davom ettirish
                    continue
                except Exception:
                    break
            else:
                output_placeholder.write("⏳ Ovoz oqimi ulanmoqda...")
                time.sleep(0.5)
            
            # Agar STOP tugmasi bosilsa, tsiklni to'xtatish
            if not webrtc_ctx.state.playing:
                break