import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import librosa
import numpy as np
import tensorflow as tf
import pickle
import queue
import time

# Sahifa sozlamalari
st.set_page_config(page_title="AI Gender Recognition", page_icon="🎙️", layout="centered")

# CSS orqali dizaynni biroz chiroyliroq qilamiz
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; }
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Model va Scalerni keshga yuklash
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('final_uzbek_gender_model.h5')
        with open('gender_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Model yuklashda xatolik: {e}")
        return None, None

model, scaler = load_assets()

# --- REAL-TIME AUDIO PROTSESSOR ---
class GenderProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.result_queue = queue.Queue()

    def recv_audio(self, frame):
        # Audio ma'lumotni massivga o'tkazish
        audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        sr = frame.sample_rate
        
        try:
            # MFCC (Treningdagi kabi 40 ta parametr)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            
            if model and scaler:
                features = scaler.transform(mfccs_scaled.reshape(1, -1))
                prediction = model.predict(features, verbose=0)[0][0]
                
                gender = "Ayol" if prediction > 0.5 else "Erkak"
                prob = prediction if prediction > 0.5 else 1 - prediction
                self.result_queue.put((gender, prob))
        except Exception:
            pass
        
        return frame

# --- ASOSIY INTERFEYS ---
st.title("🎙️ Uzbek Gender AI")
st.write("Ovoz orqali jinsni aniqlash tizimi (99% aniqlik)")

selected = option_menu(
    menu_title=None,
    options=["Fayl yuklash", "Jonli muloqot"],
    icons=["cloud-upload", "mic"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# STUN serverlar (ulanish barqaror bo'lishi uchun)
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if selected == "Fayl yuklash":
    st.info("Audio faylni (.wav, .mp3) yuklang")
    file = st.file_uploader("", type=['wav', 'mp3', 'm4a'])
    
    if file:
        st.audio(file)
        if st.button("Tahlil qilish"):
            with st.spinner("Sun'iy intellekt o'ylamoqda..."):
                audio, sr = librosa.load(file, sr=16000)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfccs_scaled = np.mean(mfccs.T, axis=0)
                features = scaler.transform(mfccs_scaled.reshape(1, -1))
                pred = model.predict(features, verbose=0)[0][0]
                
                res = "AYOL" if pred > 0.5 else "ERKAK"
                p = pred if pred > 0.5 else 1 - pred
                clr = "#FF4B4B" if res == "AYOL" else "#1F77B4"
                
                st.markdown(f"""
                    <div class="result-card" style="border: 3px solid {clr}; background-color: {clr}11;">
                        <h1 style="color: {clr};">{res}</h1>
                        <h3>Ishonch darajasi: {p*100:.1f}%</h3>
                    </div>
                """, unsafe_allow_html=True)

else:
    st.warning("Diqqat: Mikrofon ishlashi uchun START tugmasini bosing va ruxsat bering.")
    
    ctx = webrtc_streamer(
        key="gender-mic-v3",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIG,
        audio_processor_factory=GenderProcessor,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )

    status_placeholder = st.empty()

    if ctx.state.playing:
        while True:
            if ctx.audio_processor:
                try:
                    # Queue'dan natijani olish (timeout bilan)
                    gender, prob = ctx.audio_processor.result_queue.get(timeout=1.5)
                    color = "#FF4B4B" if gender == "Ayol" else "#1F77B4"
                    
                    with status_placeholder.container():
                        st.markdown(f"""
                            <div class="result-card" style="border: 3px solid {color}; background-color: {color}22;">
                                <h1 style="color: {color};">{gender}</h1>
                                <h3>Jonli tahlil: {prob*100:.1f}%</h3>
                                <p>Gapirishda davom eting...</p>
                            </div>
                        """, unsafe_allow_html=True)
                except queue.Empty:
                    continue
            else:
                # Protsessor hali yuklanmagan bo'lsa kutish
                status_placeholder.write("Ulanmoqda...")
                time.sleep(0.5)
            
            # Agar foydalanuvchi STOP bossa tsikldan chiqish
            if not ctx.state.playing:
                break