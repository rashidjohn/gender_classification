import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import librosa
import numpy as np
import tensorflow as tf
import pickle
import queue
import time
import logging

# Loglardagi keraksiz asyncio xatoliklarini kamaytirish
logging.basicConfig(level=logging.ERROR)

# Sahifa sozlamalari
st.set_page_config(page_title="AI Gender Recognition", page_icon="🎙️", layout="centered")

# --- RESURSLARNI YUKLASH ---
@st.cache_resource
def load_assets():
    try:
        # Modelni yuklash
        model = tf.keras.models.load_model('final_uzbek_gender_model.h5', compile=False)
        # Scalerni yuklash
        with open('gender_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Resurslarni yuklashda xatolik yuz berdi: {e}")
        return None, None

model, scaler = load_assets()

# --- AUDIO PROTSESSOR ---
class GenderProcessor(AudioProcessorBase):
    def __init__(self):
        self.result_queue = queue.Queue()

    def recv_audio(self, frame):
        try:
            # Audio ma'lumotni olish
            audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
            sr = frame.sample_rate
            
            # MFCC xususiyatlarini ajratish
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)
            
            if model and scaler:
                # Bashorat qilish
                features = scaler.transform(mfccs_mean)
                prediction = model.predict(features, verbose=0)[0][0]
                
                gender = "Ayol" if prediction > 0.5 else "Erkak"
                prob = prediction if prediction > 0.5 else 1 - prediction
                
                # Natijani xavfsiz navbatga qo'yish
                self.result_queue.put({"gender": gender, "prob": prob})
        except Exception:
            pass # Xatolik bo'lsa protsessni to'xtatmaslik
            
        return frame

# --- INTERFEYS ---
st.title("🎙️ Uzbek Gender AI")
st.markdown("Ovoz orqali jinsni aniqlovchi aqlli tizim")

selected = option_menu(
    menu_title=None,
    options=["Fayl yuklash", "Jonli tahlil"],
    icons=["cloud-upload", "mic"],
    orientation="horizontal",
    styles={"container": {"padding": "0!important", "background-color": "#fafafa"}}
)

# STUN serverlar ulanish barqarorligi uchun juda muhim
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

if selected == "Fayl yuklash":
    # Label warning'ni oldini olish uchun label_visibility="collapsed"
    file = st.file_uploader("Audio yuklang", type=['wav', 'mp3', 'm4a'], label_visibility="collapsed")
    
    if file:
        st.audio(file)
        if st.button("Tahlil qilish", use_container_width=True):
            with st.spinner("Aniqlanmoqda..."):
                try:
                    audio, sr = librosa.load(file, sr=16000)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                    mfccs_scaled = np.mean(mfccs.T, axis=0).reshape(1, -1)
                    
                    features = scaler.transform(mfccs_scaled)
                    pred = model.predict(features, verbose=0)[0][0]
                    
                    res = "AYOL" if pred > 0.5 else "ERKAK"
                    confidence = pred if pred > 0.5 else 1 - pred
                    color = "#FF4B4B" if res == "AYOL" else "#1F77B4"
                    
                    st.markdown(f"""
                        <div style="border: 2px solid {color}; padding: 20px; border-radius: 15px; text-align: center;">
                            <h1 style="color: {color};">{res}</h1>
                            <h3>Ishonch: {confidence*100:.1f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Faylni tahlil qilishda xatolik: {e}")

else:
    st.info("Mikrofonni yoqing va gapiring. Tizim avtomatik tahlil qiladi.")
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="gender-recognition-v4",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=GenderProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

    res_placeholder = st.empty()

    # NoneType xatoligidan qochish uchun ctx va audio_processor'ni tekshirish
    if ctx and ctx.state.playing:
        while True:
            # Asosiy tekshiruv: protsessor mavjudmi?
            if ctx.audio_processor is not None:
                try:
                    # Ma'lumotni navbatdan olish
                    data = ctx.audio_processor.result_queue.get(timeout=2.0)
                    gender = data["gender"]
                    prob = data["prob"]
                    
                    clr = "#FF4B4B" if gender == "Ayol" else "#1F77B4"
                    
                    with res_placeholder.container():
                        st.markdown(f"""
                            <div style="border: 3px solid {clr}; padding: 30px; border-radius: 20px; text-align: center; background-color: {clr}10;">
                                <h1 style="color: {clr};">{gender.upper()}</h1>
                                <h2>{prob*100:.1f}%</h2>
                                <p>Jonli tahlil...</p>
                            </div>
                        """, unsafe_allow_html=True)
                except queue.Empty:
                    # Navbat bo'sh bo'lsa kutish
                    continue
                except Exception:
                    break
            else:
                # Protsessor hali yuklanayotgan bo'lsa
                res_placeholder.warning("⏳ Mikrofon ulanmoqda, iltimos kuting...")
                time.sleep(0.5)
            
            # Agar foydalanuvchi Stop bossa
            if not ctx.state.playing:
                break