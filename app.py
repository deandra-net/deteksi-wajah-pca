import streamlit as st
import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Perbandingan Wajah PCA", page_icon="👤", layout="centered")

# --- KUSTOMISASI CSS (FRONTEND STYLE) ---
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: white; }
    .st-emotion-cache-1v0mbdj > img { border-radius: 10px; }
    .footer { text-align: center; font-size: 11px; color: #888; margin-top: 50px; padding: 10px; background-color: #1e1e1e; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

IMG_SIZE = (100, 100)
THRESHOLD = 0.80  # Batas kelulusan kemiripan sesuai standar web pengujianmu

def process_uploaded_image(uploaded_file):
    """Mendeteksi wajah dari kamera/unggah, crop, grayscale, resize, lalu flatten."""
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        face_crop = gray
    else:
        x, y, w, h = faces[0]
        face_crop = gray[y:y+h, x:x+w]
        
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    face_normalized = face_resized / 255.0
    return face_normalized.flatten().reshape(1, -1)

# --- FUNGSI MEMBACA DATASET DARI FOLDER ---
def load_face_image_from_path(path):
    """Membaca file gambar FGNET (JPG/PNG) langsung sebagai grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    resized = cv2.resize(img, IMG_SIZE)
    normalized = resized / 255.0
    return normalized.flatten()

# --- TRAINING PCA (OTAK AI) ---
@st.cache_resource
def train_pca_model(dataset_path="dataset"):
    """Membaca seluruh folder orang_XXX dari FGNET dan melatih PCA."""
    X = []
    if not os.path.exists(dataset_path):
        st.warning("Folder 'dataset' tidak ditemukan. Pastikan skrip rapikan_fgnet.py sudah dijalankan.")
        dummy = np.random.rand(5, 10000)
        pca = PCA(n_components=min(50, len(dummy)))
        pca.fit(dummy)
        return pca

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder): continue
        
        for filename in os.listdir(person_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".pgm")):
                image_path = os.path.join(person_folder, filename)
                vector = load_face_image_from_path(image_path)
                if vector is not None:
                    X.append(vector)
    
    X_matrix = np.array(X)
    if len(X_matrix) > 0:
        n_components = min(50, len(X_matrix)) 
        pca = PCA(n_components=n_components)
        pca.fit(X_matrix)
        return pca
    else:
        st.error("Folder dataset kosong! Pastikan file gambar FGNET sudah masuk.")
        return None

pca_model = train_pca_model()

# --- ANTARMUKA WEB (UI FRONTEND) ---
st.markdown("<h2 style='text-align: center;'>👤 Perbandingan Wajah PCA</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>Unggah foto masa kecil & foto sekarang untuk dibandingkan</p>", unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns(2)
with col1:
    st.markdown("<p style='text-align: center;'>📸 <b>Foto Masa Kecil</b></p>", unsafe_allow_html=True)
    file_1 = st.file_uploader("Upload Foto Masa Kecil", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if file_1: st.image(file_1)
with col2:
    st.markdown("<p style='text-align: center;'>📸 <b>Foto Sekarang</b></p>", unsafe_allow_html=True)
    file_2 = st.file_uploader("Upload Foto Sekarang", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if file_2: st.image(file_2)

st.write("")
if st.button("🔍 Bandingkan Wajah", use_container_width=True):
    if file_1 and file_2 and pca_model is not None:
        with st.spinner("Memproyeksikan ke ruang PCA..."):
            face_1_vector = process_uploaded_image(file_1)
            face_2_vector = process_uploaded_image(file_2)
            
            face_1_pca = pca_model.transform(face_1_vector)
            face_2_pca = pca_model.transform(face_2_vector)
            
            similarity = cosine_similarity(face_1_pca, face_2_pca)[0][0]
            
            st.markdown("---")
            st.markdown("### Hasil Analisis")
            
            if similarity >= THRESHOLD:
                st.success(f"**Wajah Mirip!** (Similarity: {similarity:.2f})")
                st.info(f"Nilai similarity di atas threshold ({THRESHOLD:.2f}).")
            else:
                st.error(f"**Wajah Tidak Mirip.** (Similarity: {similarity:.2f})")
                st.info(f"Nilai similarity di bawah threshold ({THRESHOLD:.2f}).")
    else:
        st.warning("Unggah kedua foto terlebih dahulu!")

# --- FOOTER KEAMANAN ---
st.markdown("<div class='footer'>🔒 Semua pemrosesan dilakukan di perangkat Anda. Gambar tidak dikirim ke server manapun.</div>", unsafe_allow_html=True)