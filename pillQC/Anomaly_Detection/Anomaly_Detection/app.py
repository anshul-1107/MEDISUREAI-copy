import os
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configure Streamlit page
st.set_page_config(layout="wide")
st.title("📊 Image Clustering & Anomaly Detection (DBSCAN & KMeans)")

# 🔹 Root folder setup
root_folder = st.text_input("Enter root images folder", 
                          value=r"C:\Users\wasfa\OneDrive\Desktop\coding\Xebia_internship\project\Anomaly_Detection\Anomaly_Detection\seg_img\seg_img")

if not os.path.exists(root_folder):
    st.error("❌ Root folder not found. Please check the path.")
    st.stop()

# 🔹 Detect categories like 'normal', 'chip', 'dirt'
categories = sorted([f for f in os.listdir(root_folder) 
                   if os.path.isdir(os.path.join(root_folder, f))])

if not categories:
    st.error("❌ No subfolders found in root folder.")
    st.stop()

selected_category = st.selectbox("Select Image Category", categories)
selected_folder = os.path.join(root_folder, selected_category)
st.success(f"✅ Selected folder: {selected_folder}")

# 🔹 Load images with progress bar
@st.cache_data
def load_images(folder_path, target_size=(128, 128)):
    paths, data = [], []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        try:
            img_path = os.path.join(folder_path, file)
            img = load_img(img_path, target_size=target_size)
            paths.append(img_path)
            data.append(img_to_array(img))
            
            # Update progress
            progress = (i + 1) / len(files)
            progress_bar.progress(progress)
            status_text.text(f"Loading images... {int(progress*100)}%")
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    return np.array(data), paths

images, image_paths = load_images(selected_folder)
st.success(f"✅ Loaded {len(images)} images from {selected_category}")

if len(images) == 0:
    st.warning("⚠️ No valid images found.")
    st.stop()

# 🔹 Feature extraction with caching
@st.cache_data
def extract_features(_images):
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(128, 128, 3))
    features = model.predict(preprocess_input(_images), verbose=0)
    features_scaled = StandardScaler().fit_transform(features)
    features_pca = PCA(n_components=2).fit_transform(features_scaled)
    return features_pca

features_pca = extract_features(images)

# 🔹 Clustering parameters
st.sidebar.header("Clustering Parameters")
col1, col2 = st.sidebar.columns(2)

with col1:
    eps_val = st.slider("DBSCAN eps", 0.1, 10.0, 3.0, step=0.1)
    min_samples = st.slider("DBSCAN min_samples", 1, 20, 5)

with col2:
    k = st.slider("KMeans clusters", 2, 10, 3)
    n_std = st.slider("Anomaly threshold (σ)", 1.0, 3.0, 2.0, step=0.5)

# 🔹 Perform clustering
@st.cache_data
def perform_clustering(_features, eps, min_samples, n_clusters):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = db.fit_predict(_features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(_features)
    
    return db_labels, kmeans_labels, kmeans

db_labels, kmeans_labels, kmeans = perform_clustering(features_pca, eps_val, min_samples, k)

# 🔹 Anomaly detection functions
def detect_anomalies_dbscan(labels, features):
    return np.where(labels == -1)[0].tolist()

def detect_anomalies_kmeans(labels, features, model, threshold):
    anomalies = []
    centroids = model.cluster_centers_
    
    for i, (label, point) in enumerate(zip(labels, features)):
        centroid = centroids[label]
        distance = np.linalg.norm(point - centroid)
        
        # Calculate threshold for this cluster
        cluster_points = features[labels == label]
        cluster_distances = [np.linalg.norm(p - centroid) for p in cluster_points]
        cluster_threshold = np.mean(cluster_distances) + threshold * np.std(cluster_distances)
        
        if distance > cluster_threshold:
            anomalies.append(i)
    
    return anomalies

# 🔹 Detect anomalies
db_anomalies = detect_anomalies_dbscan(db_labels, features_pca)
kmeans_anomalies = detect_anomalies_kmeans(kmeans_labels, features_pca, kmeans, n_std)
combined_anomalies = list(set(db_anomalies) & set(kmeans_anomalies))

# 🔹 Visualization functions
def plot_clusters(pca, labels, anomalies, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normal points
    normal_mask = ~np.isin(np.arange(len(labels)), anomalies)
    for lbl in set(labels[normal_mask]):
        if lbl == -1: continue
        xy = pca[labels == lbl]
        ax.scatter(xy[:, 0], xy[:, 1], label=f"Cluster {lbl}", alpha=0.7)
    
    # Anomalies
    if len(anomalies) > 0:
        anomaly_points = pca[anomalies]
        ax.scatter(anomaly_points[:, 0], anomaly_points[:, 1],
                  color='red', marker='X', s=100,
                  label='Anomalies', linewidths=1.5)
    
    ax.set_title(title)
    ax.legend()
    return fig

def display_image_grid(indices, paths, title, max_images=12):
    st.subheader(title)
    
    # Convert numpy array to list if needed and check if empty
    if isinstance(indices, np.ndarray):
        if indices.size == 0:
            st.info("No images found")
            return
        indices = indices.tolist()
    elif not indices:  # For regular empty lists
        st.info("No images found")
        return
    
    n_cols = 4
    n_rows = int(np.ceil(min(len(indices), max_images) / n_cols))
    
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx < len(indices) and idx < max_images:
                try:
                    img = Image.open(paths[indices[idx]]).convert("RGB")
                    cols[col].image(img, use_container_width=True,
                                  caption=f"Image {indices[idx]+1}")
                except Exception as e:
                    cols[col].error(f"Error loading image: {e}")

# 🔹 Main display
tab1, tab2, tab3 = st.tabs(["Cluster Visualization", "Anomaly Detection", "Image Explorer"])

with tab1:
    st.header("Cluster Visualization")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_clusters(features_pca, db_labels, db_anomalies, "DBSCAN Clustering"))
    with col2:
        st.pyplot(plot_clusters(features_pca, kmeans_labels, kmeans_anomalies, "KMeans Clustering"))

with tab2:
    st.header("🚨 Anomaly Detection Results")
    
    st.subheader("Detection Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("DBSCAN Anomalies", len(db_anomalies))
    col2.metric("KMeans Anomalies", len(kmeans_anomalies))
    col3.metric("High-Confidence Anomalies", len(combined_anomalies))
    
    st.subheader("Anomaly Images")
    anomaly_type = st.radio("Show anomalies detected by:", 
                          ["DBSCAN only", "KMeans only", "Both methods"],
                          horizontal=True)
    
    if anomaly_type == "DBSCAN only":
        display_image_grid(db_anomalies, image_paths, "DBSCAN-Detected Anomalies")
    elif anomaly_type == "KMeans only":
        display_image_grid(kmeans_anomalies, image_paths, "KMeans-Detected Anomalies")
    else:
        display_image_grid(combined_anomalies, image_paths, "High-Confidence Anomalies (Detected by Both)")

with tab3:
    st.header("🔍 Image Explorer")
    cluster_method = st.radio("Clustering method:", ["DBSCAN", "KMeans"], horizontal=True)
    
    if cluster_method == "DBSCAN":
        clusters = sorted(set(db_labels))
        selected_cluster = st.selectbox("Select cluster", clusters)
        cluster_indices = np.where(db_labels == selected_cluster)[0].tolist()  # Convert to list
    else:
        clusters = sorted(set(kmeans_labels))
        selected_cluster = st.selectbox("Select cluster", clusters)
        cluster_indices = np.where(kmeans_labels == selected_cluster)[0].tolist()  # Convert to list
    
    display_image_grid(cluster_indices, image_paths, f"Images in Cluster {selected_cluster}")

# 🔹 Download results
st.sidebar.header("Export Results")
if st.sidebar.button("📥 Save Anomaly Report"):
    # Create a report with anomaly information
    report = f"Anomaly Detection Report\n\n"
    report += f"Category: {selected_category}\n"
    report += f"Total Images: {len(images)}\n\n"
    report += f"DBSCAN Parameters: eps={eps_val}, min_samples={min_samples}\n"
    report += f"DBSCAN Anomalies: {len(db_anomalies)}\n\n"
    report += f"KMeans Parameters: clusters={k}, threshold={n_std}σ\n"
    report += f"KMeans Anomalies: {len(kmeans_anomalies)}\n\n"
    report += f"High-Confidence Anomalies: {len(combined_anomalies)}\n\n"
    
    # Add anomaly file names
    if combined_anomalies:
        report += "Anomaly Files:\n"
        for idx in combined_anomalies:
            report += f"- {os.path.basename(image_paths[idx])}\n"
    
    # Save to file
    with open("anomaly_report.txt", "w") as f:
        f.write(report)
    
    st.sidebar.success("Report saved as anomaly_report.txt")