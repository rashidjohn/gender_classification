import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import librosa
import numpy as np
import tensorflow as tf
import pickle
import queue

# Sahifa sozlamalari
st.set_page_config(page_title="Uzbek Gender AI", page_icon="🎙️", layout="wide")

@st.cache_resource
def load_assets():
    # Modelni yuklash
    model = tf.keras.models.load_model('final_uzbek_gender_model.h5')
    # Scalerni yuklash
    with open('gender_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

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
            # InconsistentVersionWarning ni chetlab o'tish uchun reshape
            features = scaler.transform(mfccs_scaled.reshape(1, -1))
            prediction = model.predict(features, verbose=0)[0][0]
            
            gender = "Ayol" if prediction > 0.5 else "Erkak"
            prob = prediction if prediction > 0.5 else 1 - prediction
            self.result_queue.put((gender, prob))
        except Exception:
            pass
        return frame

st.title("🎙️ O'zbek Ovozli Jins Aniqlash Tizimi")

selected = option_menu(
    menu_title=None,
    options=["Fayl yuklash", "Real-vaqt (Mikrofon)"],
    icons=["cloud-upload", "mic"],
    orientation="horizontal",
)

if selected == "Fayl yuklash":
    uploaded_file = st.file_uploader("Audio faylni tanlang", type=['wav', 'mp3', 'm4a'])
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Tahlil qilish ✨"):
            with st.spinner("Tahlil ketmoqda..."):
                audio, sr = librosa.load(uploaded_file, sr=16000)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfccs_scaled = np.mean(mfccs.T, axis=0)
                features = scaler.transform(mfccs_scaled.reshape(1, -1))
                pred = model.predict(features, verbose=0)[0][0]
                
                label = "AYOL" if pred > 0.5 else "ERKAK"
                prob = pred if pred > 0.5 else 1 - pred
                color = "#FF4B4B" if label == "AYOL" else "#1F77B4"
                st.markdown(f"<h2 style='color:{color}'>{label} ({prob*100:.1f}%)</h2>", unsafe_allow_html=True)

elif selected == "Real-vaqt (Mikrofon)":
    st.info("Mikrofonni yoqing va gapiring...")
    
    ctx = webrtc_streamer(
        key="gender-streamer",
        audio_processor_factory=GenderProcessor,
        media_stream_constraints={"video": False, "audio": True},
    )

    res_placeholder = st.empty()
    
    # Xatolikni oldini oluvchi asosiy qism
    if ctx.state.playing and ctx.audio_processor is not None:
        while True:
            try:
                # result_queue borligini tekshirish
                if hasattr(ctx.audio_processor, 'result_queue'):
                    gender, prob = ctx.audio_processor.result_queue.get(timeout=1.0)
                    color = "#FF4B4B" if gender == "Ayol" else "#1F77B4"
                    with res_placeholder.container():
                        st.markdown(f"""
                            <div style='text-align:center; padding:20px; border-radius:10px; border:2px solid {color}'>
                                <h1 style='color:{color};'>{gender}</h1>
                                <p>Ishonch: {prob*100:.1f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    break
            except queue.Empty:
                continue
            except Exception:
                break