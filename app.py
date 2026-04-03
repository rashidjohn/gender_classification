import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
import os
import tensorflow as tf
from pathlib import Path
from audio_recorder_streamlit import audio_recorder

# ─────────────────────────────────────────────
#  Sahifa sozlamalari
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="O'zbek Jins Aniqlash",
    page_icon="🎙️",
    layout="centered",
)

# ─────────────────────────────────────────────
#  CSS dizayn
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}

.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.result-card {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeIn 0.5s ease;
}

.male-card {
    background: linear-gradient(135deg, #1e3a5f, #2563eb22);
    border: 2px solid #3b82f6;
    box-shadow: 0 0 30px #3b82f655;
}

.female-card {
    background: linear-gradient(135deg, #4c1d95, #db277722);
    border: 2px solid #ec4899;
    box-shadow: 0 0 30px #ec489955;
}

.result-emoji  { font-size: 4rem; margin-bottom: 0.5rem; }
.result-label  { font-size: 2rem; font-weight: 700; color: white; }
.result-confidence { font-size: 1rem; color: #cbd5e1; margin-top: 0.3rem; }

.info-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(167,139,250,0.4);
    border-radius: 16px;
    padding: 1rem;
}

div[data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.8);
}

.stProgress > div > div {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Model va scaler yuklash (kesh bilan)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model("final_uzbek_gender_model.h5")
    with open("gender_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ─────────────────────────────────────────────
#  MFCC xususiyatlarini chiqarish
# ─────────────────────────────────────────────
def extract_features(audio_path: str, n_mfcc: int = 40) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trim) < sr * 0.3:
        y_trim = y
    mfccs = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


# ─────────────────────────────────────────────
#  Bashorat qilish (bytes dan)
# ─────────────────────────────────────────────
def predict_from_bytes(audio_bytes: bytes, model, scaler, suffix=".wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        features = extract_features(tmp_path)
        features_scaled = scaler.transform([features])
        prob = float(model.predict(features_scaled, verbose=0)[0][0])
        gender     = "Ayol" if prob >= 0.5 else "Erkak"
        confidence = prob if prob >= 0.5 else 1 - prob
        return gender, confidence, prob
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
#  Natija kartasini ko'rsatish
# ─────────────────────────────────────────────
def show_result(gender, confidence, raw_prob):
    if gender == "Erkak":
        card_cls, emoji, label_color = "male-card", "👨", "#60a5fa"
    else:
        card_cls, emoji, label_color = "female-card", "👩", "#f472b6"

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label" style="color:{label_color}">{gender}</div>
        <div class="result-confidence">Ishonchlilik: {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Ehtimollik taqsimoti")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👨 Erkak", f"{(1-raw_prob)*100:.1f}%")
        st.progress(1 - raw_prob)
    with col2:
        st.metric("👩 Ayol", f"{raw_prob*100:.1f}%")
        st.progress(raw_prob)


# ═══════════════════════════════════════════════
#  SAHIFA BOSHLANISHI
# ═══════════════════════════════════════════════
st.markdown('<div class="hero-title">🎙️ Jins Aniqlash</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>O'zbek tilida ovoz orqali jinsni aniqlash · "
    "Deep Learning modeli</div>",
    unsafe_allow_html=True,
)

st.markdown("""
<div class="info-box">
📌 <strong>Qanday ishlaydi?</strong> Audio fayldan 40 ta MFCC (Mel-Frequency Cepstral 
Coefficients) xususiyati olinib, normallashtiriladi va neyron tarmoq orqali jins aniqlanadi.
</div>
""", unsafe_allow_html=True)

# ─── Model yuklash ───
try:
    model, scaler = load_model_and_scaler()
    st.success("✅ Model muvaffaqiyatli yuklandi", icon="🤖")
except FileNotFoundError:
    st.error(
        "❌ Model fayllari topilmadi! "
        "`final_uzbek_gender_model.h5` va `gender_scaler.pkl` "
        "fayllarini `app.py` bilan bir papkaga joylang."
    )
    st.stop()


# ═══════════════════════════════════════════════
#  TABLAR
# ═══════════════════════════════════════════════
tab_mic, tab_file = st.tabs(["🎤  Mikrofondan yozib olish", "📁  Fayl yuklash"])


# ──────────────────────────────────────────────
#  TAB 1 — Mikrofon
# ──────────────────────────────────────────────
with tab_mic:
    st.markdown("### 🎤 Brauzerdan ovoz yozib olish")
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.9rem;'>"
        "Quyidagi tugmani bosib gapiring, to'xtatish uchun yana bosing.</p>",
        unsafe_allow_html=True,
    )

    col_btn, col_hint = st.columns([1, 2])
    with col_btn:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#ef4444",
            neutral_color="#a78bfa",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=3.0,   # 3 soniya sukunatda avtomatik to'xtaydi
            sample_rate=44_100,
        )

    with col_hint:
        st.markdown("""
        <div style='color:#64748b; font-size:0.85rem; padding-top:1.2rem; line-height:2;'>
        🟣 &nbsp;<b>Bosing</b> → yozib olish boshlanadi<br>
        🔴 &nbsp;<b>Yozilmoqda…</b> qayta bosing → to'xtaydi<br>
        ⏱️ &nbsp;3 soniya jim qolsa avtomatik to'xtaydi
        </div>
        """, unsafe_allow_html=True)

    if audio_bytes:
        st.markdown("---")
        st.markdown("**🔊 Yozib olingan ovoz:**")
        st.audio(audio_bytes, format="audio/wav")

        if st.button("🔍 Jinsni aniqlash", key="btn_mic", use_container_width=True):
            with st.spinner("Tahlil qilinmoqda…"):
                try:
                    gender, confidence, raw_prob = predict_from_bytes(
                        audio_bytes, model, scaler, suffix=".wav"
                    )
                    show_result(gender, confidence, raw_prob)
                except Exception as e:
                    st.error(f"Xatolik: {e}")
    else:
        st.markdown("""
        <div style='text-align:center; color:#334155; margin-top:2.5rem; font-size:0.9rem;'>
        ☝️ Mikrofon tugmasini bosib gapiring
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  TAB 2 — Fayl yuklash
# ──────────────────────────────────────────────
with tab_file:
    st.markdown("### 📁 Audio fayl yuklash")

    uploaded = st.file_uploader(
        "Fayl tanlang",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        help="Ovoz fayli kamida 1 soniya bo'lishi kerak",
        label_visibility="collapsed",
    )

    if uploaded:
        st.audio(uploaded, format=uploaded.type)

        if st.button("🔍 Jinsni aniqlash", key="btn_file", use_container_width=True):
            with st.spinner("Tahlil qilinmoqda…"):
                suffix = Path(uploaded.name).suffix or ".wav"
                uploaded.seek(0)
                try:
                    gender, confidence, raw_prob = predict_from_bytes(
                        uploaded.read(), model, scaler, suffix=suffix
                    )
                    show_result(gender, confidence, raw_prob)
                except Exception as e:
                    st.error(f"Xatolik: {e}")

        with st.expander("🔬 Texnik tafsilotlar (MFCC xususiyatlari)"):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded.name).suffix or ".wav"
            ) as tmp:
                uploaded.seek(0)
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                features = extract_features(tmp_path)
            finally:
                os.unlink(tmp_path)
            feat_str = ", ".join([f"{v:.3f}" for v in features[:10]])
            st.code(
                f"MFCC [0:10]: [{feat_str}, …]\n"
                f"O'lcham: {features.shape[0]} ta xususiyat"
            )
    else:
        st.markdown("""
        <div style='text-align:center; color:#334155; margin-top:3rem; font-size:2.5rem;'>📂</div>
        <div style='text-align:center; color:#334155; margin-top:0.5rem;'>
        WAV, MP3, OGG, FLAC yoki M4A faylni yuklang
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#334155; font-size:0.8rem;'>"
    "O'zbek Gender Aniqlash Modeli · TensorFlow + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)