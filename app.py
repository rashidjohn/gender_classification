import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import pickle
import os
import plotly.graph_objects as go

st.set_page_config(page_title="O'zbek Ovozli Jins Aniqlagich", page_icon="🎙️")

@st.cache_resource
def load_assets():
    """Model va Scalerni xotiraga bir marta yuklab oladi"""
    model = tf.keras.models.load_model('final_uzbek_gender_model.h5')
    with open('gender_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def process_audio(audio_file, scaler):
    """Audioni MFCC xususiyatlariga o'tkazish"""
    audio, sr = librosa.load(audio_file, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return scaler.transform(mfccs_scaled.reshape(1, -1))

st.title("🎙️ Ovoz orqali jinsni aniqlash")
st.markdown("Ushbu sun'iy intellekt modeli o'zbek tilidagi ovozlarni **99% aniqlik** bilan tahlil qiladi.")

try:
    model, scaler = load_assets()
    
    uploaded_file = st.file_uploader("Audio faylni tanlang (wav, mp3, m4a)", type=['wav', 'mp3', 'm4a'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Tahlil qilish ✨"):
            with st.spinner("Ovoz tahlil qilinmoqda..."):
                features = process_audio(uploaded_file, scaler)
                
                prediction = model.predict(features, verbose=0)[0][0]
                
                if prediction > 0.5:
                    label = "AYOL"
                    prob = prediction
                    color = "#FF4B4B" # Pushti/Qizil
                else:
                    label = "ERKAK"
                    prob = 1 - prediction
                    color = "#1F77B4" # Ko'k
                
                st.subheader(f"Natija: :{color}[{label}]")
                st.progress(float(prob))
                st.write(f"Model ishonchi: **{prob*100:.2f}%**")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    title = {'text': f"{label} ehtimolligi (%)"},
                    gauge = {'axis': {'range': [0, 100]},
                             'bar': {'color': color}}
                ))
                st.plotly_chart(fig)

except Exception as e:
    st.error(f"Xatolik yuz berdi: {e}")
    st.info("Iltimos, model va scaler fayllari borligini tekshiring.")