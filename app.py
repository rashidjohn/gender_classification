import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import io

# Sahifa sozlamalari
st.set_page_config(page_title="Uzbek Gender AI", page_icon="🎤", layout="centered")

# --- MODELLARNI YUKLASH ---
@st.cache_resource
def load_resources():
    try:
        # Modelni yuklash (compile=False xatoliklarni kamaytiradi)
        model = tf.keras.models.load_model('final_uzbek_gender_model.h5', compile=False)
        # Scalerni yuklash
        with open('gender_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Fayllarni yuklashda xatolik: {e}")
        return None, None

model, scaler = load_resources()

# --- TAHLIL FUNKSIYASI ---
def process_audio(audio_data):
    try:
        # Audio faylni xotiradan o'qish (BytesIO)
        with io.BytesIO(audio_data) as bf:
            y, sr = librosa.load(bf, sr=16000)
            
            # 10 soniyalik cheklov (16000 * 10 = 160,000 ta sample)
            if len(y) > 160000:
                y = y[:160000]
            
            # MFCC xususiyatlarini olish (40 ta parametr)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
            
            if model and scaler:
                # Bashorat
                features = scaler.transform(mfccs_processed)
                prediction = model.predict(features, verbose=0)[0][0]
                
                gender = "AYOL" if prediction > 0.5 else "ERKAK"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                return gender, confidence
        return None, None
    except Exception as e:
        st.error(f"Tahlil jarayonida xatolik: {e}")
        return None, None

# --- INTERFEYS ---
st.title("🎙️ Uzbek Gender AI")
st.markdown("Ovoz orqali jinsni aniqlash (Brauzerda yozish yoki fayl yuklash)")

# Ikkita variant: Ovoz yozish yoki Fayl yuklash
choice = st.radio("Usulni tanlang:", ["Brauzerda ovoz yozish", "Fayl yuklash (.mp3, .wav)"], horizontal=True)

final_audio = None

if choice == "Brauzerda ovoz yozish":
    # Streamlit-ning rasmiy audio recorder vidjeti
    recorded_audio = st.audio_input("Mikrofonni yoqish uchun bosing")
    if recorded_audio:
        final_audio = recorded_audio.read()

else:
    # Fayl yuklash vidjeti
    uploaded_file = st.file_uploader("Audio faylni tanlang", type=['wav', 'mp3', 'm4a'], label_visibility="collapsed")
    if uploaded_file:
        final_audio = uploaded_file.read()

# --- NATIJANI CHIQARISH ---
if final_audio:
    st.audio(final_audio)
    
    if st.button("Tahlil qilish", use_container_width=True, type="primary"):
        with st.spinner("Sun'iy intellekt tahlil qilmoqda..."):
            gender, prob = process_audio(final_audio)
            
            if gender:
                color = "#FF4B4B" if gender == "AYOL" else "#1F77B4"
                
                st.markdown(f"""
                    <div style="border: 3px solid {color}; padding: 30px; border-radius: 20px; text-align: center; background-color: {color}10; margin-top: 20px;">
                        <h1 style="color: {color}; margin-bottom: 0;">{gender}</h1>
                        <h3 style="margin-top: 10px;">Ishonch: {prob*100:.1f}%</h3>
                        <p style="color: gray; font-size: 0.9em;">(Eslatma: Maksimal 10 soniya tahlil qilindi)</p>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("Rashid Doniyorov")