import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from streamlit_lottie import st_lottie
import requests
import json

# Page Config
st.set_page_config(page_title="MediSureAI", layout="wide", page_icon="🧬")

# --- Custom CSS ---
st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #fff8fd 0%, #e0c3fc 100%) !important; }
    .title { font-size:48px !important; font-weight:800; color:#2c3e50; text-align: center; letter-spacing:2px; text-shadow: 2px 2px 0 #e8439311;}
    .subheader { font-size:22px !important; color:#7f8c8d; margin-bottom:24px;}
    .card {
        background: #fff0f9; border-radius: 18px; box-shadow: 0 4px 24px #ee82c6aa;
        padding: 2rem; margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #e84393 60%, #6c47b0 100%);
        color:white; border-radius:10px; font-weight:bold; border: none;
        transition: 0.2s all; box-shadow:0 2px 8px #e1aaff44;
    }
    .stButton>button:hover { background: #6c47b0; color: #fff; }
    .result-card {
        background: linear-gradient(90deg, #ffe6f0 60%, #dbe6fd 100%);
        border-radius: 18px; box-shadow: 0 2px 12px #6c47b066;
        padding: 1.2rem 2rem; margin-top: 1rem;
    }
    .result { color:#d63031; font-size:28px; font-weight:900; }
    .confidence { color:#0984e3; font-weight:700; font-size:22px;}
    .section-anim { margin-bottom:-32px; }
    .sidebar-anim { margin-bottom:48px;}
    </style>
""", unsafe_allow_html=True)

# --- Lottie Animation Loader ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except Exception: return None

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_ai = load_lottiefile("lottie/lottie_ai.json")
lottie_predict = load_lottiefile("lottie/lottie_predict.json")
lottie_cluster = load_lottiefile("lottie/lottie_cluster.json")
lottie_report = load_lottiefile("lottie/lottie_report.json")
lottie_success = load_lottiefile("lottie/lottie_success.json")
lottie_error = load_lottiefile("lottie/lottie_error.json")
lottie_aug = load_lottiefile("lottie/lottie_aug.json")

# --- Sidebar with Animated Logo ---
with st.sidebar:
    st_lottie(lottie_ai, height=110, key="sidebar-anim", speed=1, loop=True)
    menu = st.radio("Navigation", [
        "🏠 Introduction",
        "🧪 Medicine Quality Check",
        "🧠 Clustering & Anomaly Detection",
        "🔄 Augmentation Playground",
        "📄 Report"
    ])
    st.markdown("---")
    st_lottie(lottie_report, height=60, key="sidebar-report", speed=1, loop=True)
    st.caption("Made with ❤️ by MediSureAI Team")

class_labels = ["Damaged", "Contaminated", "Good"]
LOG_FILE = "prediction_logs.csv"

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Image", "Prediction", "Confidence"]).to_csv(LOG_FILE, index=False)

# --- INTRODUCTION SECTION ---
if menu == "🏠 Introduction":
    st.markdown("""<div class='title'>Welcome to MediSureAI</div>""", unsafe_allow_html=True)
    st_lottie(lottie_ai, height=300, key="ai-hero", speed=1, loop=True)
    st.markdown('<div class="subheader">Empowering Hospitals with AI-Powered Medicine Inspection</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="card">
        <ul>
        <li>🔍 Automated medicine defect detection with Deep Learning</li>
        <li>🧠 Clustering-based anomaly discovery using MobileNet features</li>
        <li>🗂️ Prediction history logging and CSV export</li>
        </ul>
        <i>💡 Built with Python, Streamlit, and the power of AI.</i>
        </div>
    """, unsafe_allow_html=True)

# --- QUALITY CHECK SECTION ---
elif menu == "🧪 Medicine Quality Check":
    st_lottie(lottie_predict, speed=1, height=180, key="predict-anim", loop=True)
    st.markdown("<div class='title section-anim'>Medicine Image Quality Check</div>", unsafe_allow_html=True)
    st.markdown('<div class="subheader">Upload a medicine image to analyze its quality using AI.</div>', unsafe_allow_html=True)

    @st.cache_resource
    def load_cnn_model():
        return load_model("DENSE.h5")

    model = load_cnn_model()
    uploaded_file = st.file_uploader("📤 Upload Medicine Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        img = Image.open(uploaded_file).convert("RGB").resize((256, 256))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("🔍 Analyze Image", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                st_lottie(lottie_predict, height=80, key="predict-loading", loop=True)
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = float(np.max(prediction))
                result = class_labels[predicted_class]
            # Result card with animation
            with st.container():
                st_lottie(lottie_success if confidence > 0.8 else lottie_error, height=150, key="result-anim", speed=1)
                st.markdown(f"<div class='result-card'><span class='result'>🩺 Prediction: {result}</span><br><span class='confidence'>Confidence: {confidence * 100:.2f}%</span></div>", unsafe_allow_html=True)
            st.toast(f"Prediction: {result} ({confidence*100:.2f}%)", icon="🤖" if confidence > 0.8 else "⚠️")

            new_entry = pd.DataFrame([[uploaded_file.name, result, f"{confidence*100:.2f}%"]],
                                     columns=["Image", "Prediction", "Confidence"])
            old_data = pd.read_csv(LOG_FILE)
            pd.concat([old_data, new_entry], ignore_index=True).to_csv(LOG_FILE, index=False)

# --- CLUSTERING & ANOMALY DETECTION ---
elif menu == "🧠 Clustering & Anomaly Detection":
    st_lottie(lottie_cluster, speed=1, height=170, key="cluster-anim", loop=True)
    st.markdown("<div class='title section-anim'>Image Clustering + Anomaly Detection</div>", unsafe_allow_html=True)
    st.markdown('<div class="subheader">Discover outlier medicines with clustering and PCA visualization.</div>', unsafe_allow_html=True)

    image_folder = st.text_input("📂 Enter Image Folder Path", value="images/")

    if os.path.exists(image_folder):
        image_paths, image_arrays = [], []
        for root, dirs, files in os.walk(image_folder):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    path = os.path.join(root, f)
                    try:
                        img = Image.open(path).convert("RGB").resize((128, 128))
                        image_paths.append(path)
                        image_arrays.append(np.array(img))
                    except: continue

        if len(image_arrays) > 0:
            st_lottie(lottie_cluster, height=60, key="cluster-mini", loop=True)
            # Feature extraction using MobileNetV2
            mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(128,128,3))
            features = []
            for arr in image_arrays:
                arr = arr / 255.0
                arr = np.expand_dims(arr, axis=0)
                feat = mobilenet.predict(arr)
                features.append(feat.flatten())
            features = np.array(features)
            # PCA for dimensionality reduction
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(features)
            # DBSCAN clustering
            eps = st.slider("DBSCAN eps", 0.1, 10.0, 3.0)
            min_samples = st.slider("Min Samples", 1, 20, 5)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced)
            labels = db.labels_
            anomalies = np.where(labels == -1)[0]

            st.markdown("<div class='subheader'>🚨 Detected Anomalies</div>", unsafe_allow_html=True)
            cols = st.columns(4)
            for i, idx in enumerate(anomalies[:12]):
                try:
                    img = Image.open(image_paths[idx])
                    with cols[i % 4]:
                        st.image(img, caption=os.path.basename(image_paths[idx]), use_column_width=True)
                        st_lottie(lottie_error, height=50, key=f"anomaly-{i}", loop=True)
                except: continue
        else:
            st.info("No images found in the specified folder.")

# --- AUGMENTATION PLAYGROUND ---
elif menu == "🔄 Augmentation Playground":
    st_lottie(lottie_aug, height=180, key="augment-anim", loop=True)
    st.markdown("<div class='title section-anim'>Augmentation Playground</div>", unsafe_allow_html=True)
    st.markdown('<div class="subheader">Try live image augmentations and see instant results!</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload Image for Augmentation", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Original Image", use_column_width=True)

        st.subheader("🎛️ Augmentation Controls")

        col1, col2 = st.columns(2)
        with col1:
            rotate_angle = st.slider("🔄 Rotate", -45, 45, 0)
            flip_horizontal = st.checkbox("↔️ Flip Horizontal")
            flip_vertical = st.checkbox("↕️ Flip Vertical")
        with col2:
            zoom = st.slider("🔍 Zoom (%)", 100, 200, 100)
            brightness = st.slider("💡 Brightness", 0.5, 2.0, 1.0)
            contrast = st.slider("🎨 Contrast", 0.5, 2.0, 1.0)

        if st.button("Apply Augmentation", use_container_width=True):
            with st.spinner("Applying..."):
                st_lottie(lottie_aug, height=60, key="aug-loading", loop=True)
                # Augment image
                aug_img = img.rotate(rotate_angle)
                if flip_horizontal:
                    aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
                if flip_vertical:
                    aug_img = aug_img.transpose(Image.FLIP_TOP_BOTTOM)
                if zoom > 100:
                    w, h = aug_img.size
                    zoomed = aug_img.resize((int(w * zoom / 100), int(h * zoom / 100)))
                    left = (zoomed.width - w) // 2
                    top = (zoomed.height - h) // 2
                    aug_img = zoomed.crop((left, top, left + w, top + h))
                aug_img = ImageEnhance.Brightness(aug_img).enhance(brightness)
                aug_img = ImageEnhance.Contrast(aug_img).enhance(contrast)

            st.markdown("<div class='subheader'>🖼️ Augmented Output</div>", unsafe_allow_html=True)
            st.image(aug_img, caption="Augmented Image", use_column_width=True)
            st_lottie(lottie_success, height=60, key="aug-success", loop=True)

# --- REPORT SECTION ---
elif menu == "📄 Report":
    st_lottie(lottie_report, speed=1, height=160, key="report-anim", loop=True)
    st.markdown("<div class='title section-anim'>Prediction Report</div>", unsafe_allow_html=True)
    st.markdown('<div class="subheader">See all your predictions and download the CSV report.</div>', unsafe_allow_html=True)

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df, use_container_width=True)
        col1, col2 = st.columns([1, 8])
        with col1:
            st_lottie(lottie_success, height=60, key="report-download", loop=True)
        with col2:
            st.download_button("📥 Download CSV Report", df.to_csv(index=False), file_name="medicine_quality_report.csv")
    else:
        st.warning("No prediction report found.")