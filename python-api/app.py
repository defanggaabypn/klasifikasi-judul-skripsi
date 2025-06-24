from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Menghindari error GUI thread
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import pickle
import json
import re
import random
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import BaggingClassifier
from transformers import AutoTokenizer, AutoModel
import pymysql
import hashlib
from datetime import datetime
import gc

app = Flask(__name__)
CORS(app)  # Memungkinkan request dari domain lain (PHP frontend)

# Fungsi untuk mengkonversi objek NumPy ke tipe data Python standar
def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Direktori untuk menyimpan file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
CACHE_FOLDER = os.path.join(BASE_DIR, 'cache')

# Buat direktori jika belum ada
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, CACHE_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Konstanta untuk model
MODEL_FILENAME = os.path.join(MODEL_FOLDER, 'models.pkl')
PCA_FILENAME = os.path.join(MODEL_FOLDER, 'pca_model.pkl')
SELECTOR_FILENAME = os.path.join(MODEL_FOLDER, 'feature_selector.pkl')
EMBEDDING_CACHE_FILENAME = os.path.join(CACHE_FOLDER, 'embedding_cache.json')
MAX_LENGTH = 64  # Mengurangi dari 128 ke 64 untuk efisiensi memori

# Konfigurasi database
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'db': 'skripsi_classification',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# Fungsi koneksi database
def get_db_connection():
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

# Inisialisasi model IndoBERT
print("Loading IndoBERT model... (bisa memakan waktu beberapa menit)")
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
print("IndoBERT model loaded successfully!")

# Variabel global untuk menyimpan model yang sudah dilatih
trained_knn = None
trained_dt = None
pca_model = None
feature_selector = None
embedding_cache = {}
train_features = []
train_labels = []

# Load embedding cache jika ada
if os.path.exists(EMBEDDING_CACHE_FILENAME):
    try:
        with open(EMBEDDING_CACHE_FILENAME, 'r') as f:
            embedding_cache = json.load(f)
        print(f"Loaded {len(embedding_cache)} cached embeddings")
    except Exception as e:
        print(f"Error loading embedding cache: {str(e)}")
        embedding_cache = {}

# Load model jika ada
if os.path.exists(MODEL_FILENAME):
    try:
        with open(MODEL_FILENAME, 'rb') as f:
            models_data = pickle.load(f)
            trained_knn = models_data.get('knn')
            trained_dt = models_data.get('dt')
            train_features = models_data.get('train_features', [])
            train_labels = models_data.get('train_labels', [])
        print("Models loaded successfully from disk")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        trained_knn = None
        trained_dt = None

# Load PCA model jika ada
if os.path.exists(PCA_FILENAME):
    try:
        with open(PCA_FILENAME, 'rb') as f:
            pca_model = pickle.load(f)
        print(f"PCA model loaded with {pca_model.n_components_} components")
    except Exception as e:
        print(f"Error loading PCA model: {str(e)}")
        pca_model = None

# Load feature selector jika ada
if os.path.exists(SELECTOR_FILENAME):
    try:
        with open(SELECTOR_FILENAME, 'rb') as f:
            feature_selector = pickle.load(f)
        print(f"Feature selector loaded")
    except Exception as e:
        print(f"Error loading feature selector: {str(e)}")
        feature_selector = None

# Preprocessing teks yang lebih baik
def preprocess_text(text):
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).lower()
    
    # Hapus stopwords bahasa Indonesia
    stopwords = ["yang", "untuk", "pada", "ke", "para", "namun", "dan", "dengan", 
                 "dari", "di", "dalam", "secara", "oleh", "atau", "ini", "itu"]
    
    # Filter stopwords dengan cara yang lebih efisien
    words = []
    for word in text.split():
        if word not in stopwords and len(word) > 2:  # Abaikan kata pendek
            words.append(word)
    
    text = " ".join(words)
    
    # Hapus karakter khusus, angka dan spasi berlebih
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Fungsi untuk mendapatkan embedding yang lebih baik
def get_embedding(text):
    preprocessed_text = preprocess_text(text)
    
    # Cek apakah embedding sudah ada di cache
    cache_key = f"cls_{preprocessed_text}"
    if cache_key in embedding_cache:
        return np.array(embedding_cache[cache_key])
    
    # Jika tidak ada di cache, hitung embedding baru
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling - ambil rata-rata semua token, lebih baik dari hanya CLS token
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    
    # Combine CLS token dengan mean pooling
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    # Mean pooling untuk token lainnya
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embedding = (sum_embeddings / sum_mask).squeeze().numpy()
    
    # Gabungkan kedua embedding
    combined_embedding = np.concatenate([cls_embedding, mean_embedding])
    
    # Simpan ke cache
    embedding_cache[cache_key] = combined_embedding.tolist()
    
    # Simpan cache ke disk secara periodik
    if len(embedding_cache) % 10 == 0:
        save_embedding_cache()
    
    return combined_embedding

# Simpan cache embedding ke disk
def save_embedding_cache():
    try:
        with open(EMBEDDING_CACHE_FILENAME, 'w') as f:
            json.dump(embedding_cache, f)
        print(f"Saved {len(embedding_cache)} embeddings to cache")
    except Exception as e:
        print(f"Error saving embedding cache: {str(e)}")

# Simpan hasil prediksi ke database
def save_prediction_to_db(title, actual_category, knn_pred, dt_pred, confidence=0, upload_file_id=None):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                # Dapatkan ID kategori
                cursor.execute("SELECT id FROM categories WHERE name = %s", (actual_category,))
                actual_result = cursor.fetchone()
                actual_id = actual_result['id'] if actual_result else None
                
                cursor.execute("SELECT id FROM categories WHERE name = %s", (knn_pred,))
                knn_result = cursor.fetchone()
                knn_id = knn_result['id'] if knn_result else None
                
                cursor.execute("SELECT id FROM categories WHERE name = %s", (dt_pred,))
                dt_result = cursor.fetchone()
                dt_id = dt_result['id'] if dt_result else None
                
                # Simpan prediksi
                cursor.execute("""
                    INSERT INTO predictions (title, actual_category_id, knn_prediction_id, dt_prediction_id, confidence, upload_file_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (title, actual_id, knn_id, dt_id, confidence, upload_file_id))
                
                conn.commit()
                prediction_id = cursor.lastrowid
                
                # Juga simpan di classification_results jika title_id tersedia
                if actual_id:
                    # Cek apakah judul sudah ada di thesis_titles
                    cursor.execute("SELECT id FROM thesis_titles WHERE title = %s", (title,))
                    title_result = cursor.fetchone()
                    
                    if title_result:
                        title_id = title_result['id']
                        
                        # Simpan ke classification_results
                        cursor.execute("""
                            INSERT INTO classification_results 
                            (title_id, knn_prediction_id, dt_prediction_id, confidence, ensemble_prediction_id)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (title_id, knn_id, dt_id, confidence, knn_id))  # Menggunakan knn_id sebagai ensemble_prediction
                        
                        conn.commit()
                
                return prediction_id
            conn.close()
    except Exception as e:
        print(f"Error saving prediction to database: {str(e)}")
        return None

# Simpan data training dan testing ke database
def save_training_testing_data(titles, labels, predictions_knn, predictions_dt, data_types, upload_file_id=None):
    """Simpan data training dan testing ke database"""
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                for i, (title, label, knn_pred, dt_pred, data_type) in enumerate(zip(titles, labels, predictions_knn, predictions_dt, data_types)):
                    # Dapatkan ID kategori
                    cursor.execute("SELECT id FROM categories WHERE name = %s", (label,))
                    actual_result = cursor.fetchone()
                    actual_id = actual_result['id'] if actual_result else None
                    
                    cursor.execute("SELECT id FROM categories WHERE name = %s", (knn_pred,))
                    knn_result = cursor.fetchone()
                    knn_id = knn_result['id'] if knn_result else None
                    
                    cursor.execute("SELECT id FROM categories WHERE name = %s", (dt_pred,))
                    dt_result = cursor.fetchone()
                    dt_id = dt_result['id'] if dt_result else None
                    
                    # Hitung apakah prediksi benar
                    is_correct_knn = 1 if label == knn_pred else 0
                    is_correct_dt = 1 if label == dt_pred else 0
                    
                    # Simpan ke training_data
                    cursor.execute("""
                        INSERT INTO training_data 
                        (upload_file_id, title, actual_category_id, data_type, knn_prediction_id, dt_prediction_id, is_correct_knn, is_correct_dt)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (upload_file_id, title, actual_id, data_type, knn_id, dt_id, is_correct_knn, is_correct_dt))
                
                conn.commit()
                print(f"Saved {len(titles)} training/testing records to database")
                return True
            conn.close()
    except Exception as e:
        print(f"Error saving training/testing data: {str(e)}")
        return False

def save_data_split_info(total_data, training_size, testing_size, upload_file_id=None):
    """Simpan informasi pembagian data"""
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                training_percentage = (training_size / total_data) * 100
                testing_percentage = (testing_size / total_data) * 100
                
                cursor.execute("""
                    INSERT INTO data_split_info 
                    (upload_file_id, total_data, training_size, testing_size, training_percentage, testing_percentage, split_method, random_state)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (upload_file_id, total_data, training_size, testing_size, training_percentage, testing_percentage, 'train_test_split', 42))
                
                conn.commit()
                print(f"Saved data split info: {total_data} total, {training_size} train, {testing_size} test")
                return True
            conn.close()
    except Exception as e:
        print(f"Error saving data split info: {str(e)}")
        return False

def save_model_metrics(model_name, data_type, accuracy, precision_macro, recall_macro, f1_macro, 
                      precision_weighted, recall_weighted, f1_weighted, upload_file_id=None):
    """Simpan metrik model detail"""
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO model_metrics 
                    (upload_file_id, model_name, data_type, accuracy, precision_macro, recall_macro, f1_macro,
                     precision_weighted, recall_weighted, f1_weighted)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (upload_file_id, model_name, data_type, accuracy, precision_macro, recall_macro, f1_macro,
                      precision_weighted, recall_weighted, f1_weighted))
                
                conn.commit()
                return True
            conn.close()
    except Exception as e:
        print(f"Error saving model metrics: {str(e)}")
        return False

# Simpan data confusion matrix ke database
def save_confusion_matrix_data(y_true, y_pred, model_name, data_type, upload_file_id=None):
    """Simpan data confusion matrix ke database"""
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                # Hitung confusion matrix
                from sklearn.metrics import confusion_matrix
                import numpy as np
                
                # Dapatkan label unik
                labels = sorted(list(set(y_true + y_pred)))
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                
                # Simpan setiap cell dari confusion matrix
                for i, actual_label in enumerate(labels):
                    for j, predicted_label in enumerate(labels):
                        count = int(cm[i][j])
                        if count > 0:  # Hanya simpan yang ada nilainya
                            # Dapatkan ID kategori
                            cursor.execute("SELECT id FROM categories WHERE name = %s", (actual_label,))
                            actual_result = cursor.fetchone()
                            actual_id = actual_result['id'] if actual_result else None
                            
                            cursor.execute("SELECT id FROM categories WHERE name = %s", (predicted_label,))
                            predicted_result = cursor.fetchone()
                            predicted_id = predicted_result['id'] if predicted_result else None
                            
                            if actual_id and predicted_id:
                                # Cek apakah data sudah ada
                                cursor.execute("""
                                    SELECT id FROM confusion_matrix_data 
                                    WHERE upload_file_id = %s AND model_name = %s 
                                    AND actual_category_id = %s AND predicted_category_id = %s
                                """, (upload_file_id, f"{model_name}_{data_type}", actual_id, predicted_id))
                                
                                existing = cursor.fetchone()
                                if existing:
                                    # Update count
                                    cursor.execute("""
                                        UPDATE confusion_matrix_data 
                                        SET count = %s 
                                        WHERE id = %s
                                    """, (count, existing['id']))
                                else:
                                    # Insert baru
                                    cursor.execute("""
                                        INSERT INTO confusion_matrix_data 
                                        (upload_file_id, model_name, actual_category_id, predicted_category_id, count)
                                        VALUES (%s, %s, %s, %s, %s)
                                    """, (upload_file_id, f"{model_name}_{data_type}", actual_id, predicted_id, count))
                
                conn.commit()
                return True
            conn.close()
    except Exception as e:
        print(f"Error saving confusion matrix data: {str(e)}")
        return False

def generate_combined_confusion_matrix(knn_train_cm, knn_test_cm, dt_train_cm, dt_test_cm, labels):
    """Generate confusion matrix untuk training dan testing secara bersamaan"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrix Comparison: Training vs Testing', fontsize=16, fontweight='bold')
    
    # KNN Training
    im1 = axes[0, 0].imshow(knn_train_cm, interpolation='nearest', cmap='Blues')
    axes[0, 0].set_title('KNN - Training Data', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xticks(range(len(labels)))
    axes[0, 0].set_yticks(range(len(labels)))
    axes[0, 0].set_xticklabels(labels, rotation=45)
    axes[0, 0].set_yticklabels(labels)
    
    # Tambahkan text annotations untuk KNN Training
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[0, 0].text(j, i, str(knn_train_cm[i, j]), ha='center', va='center', 
                          color='white' if knn_train_cm[i, j] > knn_train_cm.max()/2 else 'black')
    
    # KNN Testing
    im2 = axes[0, 1].imshow(knn_test_cm, interpolation='nearest', cmap='Blues')
    axes[0, 1].set_title('KNN - Testing Data', fontweight='bold')
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xticks(range(len(labels)))
    axes[0, 1].set_yticks(range(len(labels)))
    axes[0, 1].set_xticklabels(labels, rotation=45)
    axes[0, 1].set_yticklabels(labels)
    
    # Tambahkan text annotations untuk KNN Testing
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[0, 1].text(j, i, str(knn_test_cm[i, j]), ha='center', va='center',
                          color='white' if knn_test_cm[i, j] > knn_test_cm.max()/2 else 'black')
    
    # DT Training
    im3 = axes[1, 0].imshow(dt_train_cm, interpolation='nearest', cmap='Greens')
    axes[1, 0].set_title('Decision Tree - Training Data', fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xticks(range(len(labels)))
    axes[1, 0].set_yticks(range(len(labels)))
    axes[1, 0].set_xticklabels(labels, rotation=45)
    axes[1, 0].set_yticklabels(labels)
    
    # Tambahkan text annotations untuk DT Training
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1, 0].text(j, i, str(dt_train_cm[i, j]), ha='center', va='center',
                          color='white' if dt_train_cm[i, j] > dt_train_cm.max()/2 else 'black')
    
    # DT Testing
    im4 = axes[1, 1].imshow(dt_test_cm, interpolation='nearest', cmap='Greens')
    axes[1, 1].set_title('Decision Tree - Testing Data', fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xticks(range(len(labels)))
    axes[1, 1].set_yticks(range(len(labels)))
    axes[1, 1].set_xticklabels(labels, rotation=45)
    axes[1, 1].set_yticklabels(labels)
    
    # Tambahkan text annotations untuk DT Testing
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1, 1].text(j, i, str(dt_test_cm[i, j]), ha='center', va='center',
                          color='white' if dt_test_cm[i, j] > dt_test_cm.max()/2 else 'black')
    
    # Adjust layout
    plt.tight_layout()
    
    # Simpan ke buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Convert ke base64
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_str

# Simpan performa model ke database
def save_model_performance(model_name, accuracy, parameters=None, upload_file_id=None):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO model_performances (model_name, accuracy, parameters, upload_file_id)
                    VALUES (%s, %s, %s, %s)
                """, (model_name, accuracy, json.dumps(parameters) if parameters else None, upload_file_id))
                
                conn.commit()
                return cursor.lastrowid
            conn.close()
    except Exception as e:
        print(f"Error saving model performance: {str(e)}")
        return None

# Analisis kata kunci dan simpan ke database
def analyze_keywords(category, titles, upload_file_id=None):
    try:
        # Preprocess dan ekstrak keyword
        keywords = {}
        for title in titles:
            words = preprocess_text(title).split()
            for word in words:
                if len(word) > 3:  # Filter kata pendek
                    keywords[word] = keywords.get(word, 0) + 1
        
        # Ambil top 20 keywords
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Simpan ke database
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                # Dapatkan ID kategori
                cursor.execute("SELECT id FROM categories WHERE name = %s", (category,))
                category_result = cursor.fetchone()
                category_id = category_result['id'] if category_result else None
                
                if category_id:
                    # Hapus analisis sebelumnya untuk kategori ini dan upload_id ini jika ada
                    if upload_file_id:
                        cursor.execute("DELETE FROM keyword_analysis WHERE category_id = %s AND upload_file_id = %s", 
                                     (category_id, upload_file_id))
                    else:
                        cursor.execute("DELETE FROM keyword_analysis WHERE category_id = %s AND upload_file_id IS NULL", 
                                     (category_id,))
                    
                    # Simpan keyword baru
                    for keyword, frequency in top_keywords:
                        cursor.execute("""
                            INSERT INTO keyword_analysis (category_id, keyword, frequency, upload_file_id)
                            VALUES (%s, %s, %s, %s)
                        """, (category_id, keyword, frequency, upload_file_id))
                    
                    conn.commit()
                    return True
            conn.close()
    except Exception as e:
        print(f"Error analyzing keywords: {str(e)}")
        return False

# Fungsi untuk menambahkan fitur domain
def add_domain_features(title):
    # Kata kunci spesifik domain yang sangat relevan
    keywords = {
        'RPL': ['sistem', 'aplikasi', 'web', 'android', 'database', 'framework', 'platform', 'informasi', 
                'online', 'digital', 'manajemen', 'data', 'otomatisasi', 'mobile', 'komputerisasi'],
        'Jaringan': ['jaringan', 'network', 'server', 'router', 'protokol', 'keamanan', 'vpn', 'lan', 
                     'wan', 'topologi', 'bandwidth', 'transmisi', 'infrastruktur', 'monitoring', 'konfigurasi'],
        'Multimedia': ['multimedia', 'game', 'animasi', 'grafis', 'audio', 'video', 'visual', 
                       'interaktif', 'gambar', '3d', 'rendering', 'virtual', 'augmented', 'reality', 'desain']
    }
    
    # Inisialisasi fitur
    features = []
    
    # 1. Panjang judul (jumlah kata)
    features.append(len(title.split()))
    
    # 2. Hitung kata kunci per kategori
    title_lower = title.lower()
    for category, words in keywords.items():
        count = sum(1 for word in words if word in title_lower)
        features.append(count)
    
    # 3. Apakah ada kata metodologi?
    methods = ['analisis', 'perancangan', 'implementasi', 'pengembangan', 'evaluasi', 'rancang', 'bangun']
    method_count = sum(1 for word in methods if word in title_lower)
    features.append(method_count)
    
    # 4. Ciri khas tertentu (contoh: apakah judul bersifat teknis?)
    technical_words = ['sistem', 'aplikasi', 'metode', 'algoritma', 'teknologi', 'framework', 'arsitektur']
    tech_count = sum(1 for word in technical_words if word in title_lower)
    features.append(tech_count)
    
    # 5. Rasio kata unik
    words = title_lower.split()
    unique_ratio = len(set(words)) / (len(words) + 1e-6)
    features.append(unique_ratio)
    
    return np.array(features)

# Gabungkan embedding dengan fitur domain
def create_enhanced_features(embedding, title):
    domain_features = add_domain_features(title)
    return np.concatenate([embedding, domain_features])

# Fungsi untuk PCA reduction
def apply_pca(features, n_components=100, fit=False):
    global pca_model
    
    features_array = np.array(features)
    
    if fit or pca_model is None:
        pca_model = PCA(n_components=n_components)
        reduced_features = pca_model.fit_transform(features_array)
        
        # Simpan model PCA
        with open(PCA_FILENAME, 'wb') as f:
            pickle.dump(pca_model, f)
    else:
              reduced_features = pca_model.transform(features_array)
    
    return reduced_features

# Fungsi untuk feature selection
def select_best_features(features, labels, k=100, fit=False):
    global feature_selector
    
    features_array = np.array(features)
    
    if fit or feature_selector is None:
        feature_selector = SelectKBest(f_classif, k=k)
        selected_features = feature_selector.fit_transform(features_array, labels)
        
        # Simpan feature selector
        with open(SELECTOR_FILENAME, 'wb') as f:
            pickle.dump(feature_selector, f)
    else:
        selected_features = feature_selector.transform(features_array)
    
    return selected_features

# Fungsi untuk oversampling kelas minoritas
def oversample_minority_classes(embeddings, titles, labels):
    # Hitung jumlah sampel per kelas
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Tentukan kelas mayoritas dan jumlah targetnya
    max_class_count = max(class_counts.values())
    
    # Data asli
    augmented_embeddings = list(embeddings)
    augmented_titles = list(titles)
    augmented_labels = list(labels)
    
    # Simpan indeks untuk setiap kelas
    class_indices = {}
    for i, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    # Augmentasi data untuk kelas minoritas
    for label, count in class_counts.items():
        if count < max_class_count:
            # Berapa banyak sampel yang perlu ditambahkan
            samples_to_add = max_class_count - count
            
            # Tambahkan sampel dengan sedikit variasi
            for _ in range(samples_to_add):
                # Pilih sampel acak dari kelas ini
                idx = random.choice(class_indices[label])
                
                # Tambahkan sedikit noise ke embedding
                noise_scale = 0.01  # Skala noise
                noise = np.random.normal(0, noise_scale, size=embeddings[idx].shape)
                new_embedding = embeddings[idx] + noise
                
                # Tambahkan ke dataset
                augmented_embeddings.append(new_embedding)
                augmented_titles.append(titles[idx])
                augmented_labels.append(label)
    
    print(f"Data asli: {len(embeddings)} sampel")
    print(f"Data setelah oversampling: {len(augmented_embeddings)} sampel")
    
    return augmented_embeddings, augmented_titles, augmented_labels

# Fungsi tuning untuk KNN
def optimize_knn(X_train, y_train):
    best_k = 3
    best_accuracy = 0
    best_weight = 'uniform'
    
    # Coba berbagai nilai k
    for k in [1, 3, 5, 7, 9, 11, 13]:
        for weight in ['uniform', 'distance']:
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            
            # Cross-validation sederhana
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X_train, y_train):
                X_fold_train = [X_train[i] for i in train_idx]
                y_fold_train = [y_train[i] for i in train_idx]
                X_fold_val = [X_train[i] for i in val_idx]
                y_fold_val = [y_train[i] for i in val_idx]
                
                knn.fit(X_fold_train, y_fold_train)
                score = accuracy_score(y_fold_val, knn.predict(X_fold_val))
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score > best_accuracy:
                best_accuracy = avg_score
                best_k = k
                best_weight = weight
    
    print(f"Best KNN parameters: k={best_k}, weight={best_weight}, accuracy={best_accuracy:.4f}")
    return KNeighborsClassifier(n_neighbors=best_k, weights=best_weight)

# Fungsi tuning untuk Decision Tree
def optimize_dt(X_train, y_train):
    best_accuracy = 0
    best_params = {'max_depth': None, 'criterion': 'gini', 'min_samples_split': 2}
    
    # Coba berbagai parameter
    for max_depth in [None, 10, 20, 30]:
        for criterion in ['gini', 'entropy']:
            for min_samples_split in [2, 5, 10]:
                # Gunakan bagging dengan decision tree
                base_dt = DecisionTreeClassifier(
                    max_depth=max_depth,
                    criterion=criterion,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                
                # Cross-validation sederhana
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = []
                
                for train_idx, val_idx in kf.split(X_train, y_train):
                    X_fold_train = [X_train[i] for i in train_idx]
                    y_fold_train = [y_train[i] for i in train_idx]
                    X_fold_val = [X_train[i] for i in val_idx]
                    y_fold_val = [y_train[i] for i in val_idx]
                    
                    base_dt.fit(X_fold_train, y_fold_train)
                    score = accuracy_score(y_fold_val, base_dt.predict(X_fold_val))
                    scores.append(score)
                
                avg_score = np.mean(scores)
                if avg_score > best_accuracy:
                    best_accuracy = avg_score
                    best_params = {
                        'max_depth': max_depth,
                        'criterion': criterion,
                        'min_samples_split': min_samples_split
                    }
    
    print(f"Best DT parameters: {best_params}, accuracy={best_accuracy:.4f}")
    return DecisionTreeClassifier(
        max_depth=best_params['max_depth'],
        criterion=best_params['criterion'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    )

# Fungsi untuk implementasi bagging dengan DT
def create_bagged_dt(dt_base, n_estimators=10):
    bagged_dt = BaggingClassifier(
        estimator=dt_base,
        n_estimators=n_estimators,
        random_state=42
    )
    return bagged_dt

# Simpan model yang lebih lengkap ke disk
def save_complete_models(models_dict):
    try:
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(models_dict, f)
        print("Complete models saved successfully")
    except Exception as e:
        print(f"Error saving models: {str(e)}")

# Class untuk model KNN dengan detil performa
class KNNDetailedModel:
    def __init__(self, n_neighbors=3, weights='uniform'):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metrics = {}
        self.confusion_matrix = None
        self.classification_report = None
        self.class_metrics = {}
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        distances, indices = self.model.kneighbors(X)
        # Convert distances to probabilities (simplified)
        probs = 1 / (1 + distances)
        # Normalize
        return probs / np.sum(probs, axis=1)[:, np.newaxis]
    
    def kneighbors(self, X):
        return self.model.kneighbors(X)
    
    def evaluate(self, X_test, y_test):
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.metrics['accuracy'] = accuracy
        
        # Confusion matrix
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        # Classification report (precision, recall, f1)
        self.classification_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        classes = sorted(list(set(y_test)))
        
        for i, cls in enumerate(classes):
            self.class_metrics[cls] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        return self.metrics

# Class untuk Decision Tree dengan detil performa
class DTDetailedModel:
    def __init__(self, max_depth=None, criterion='gini', min_samples_split=2, random_state=42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, 
            criterion=criterion,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.metrics = {}
        self.confusion_matrix = None
        self.classification_report = None
        self.class_metrics = {}
        self.feature_importances = None
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def evaluate(self, X_test, y_test):
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.metrics['accuracy'] = accuracy
        
        # Confusion matrix
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        # Classification report (precision, recall, f1)
        self.classification_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        classes = sorted(list(set(y_test)))
        
        for i, cls in enumerate(classes):
            self.class_metrics[cls] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        # Additional DT metrics
        if hasattr(self.model, 'get_depth'):
            self.metrics['tree_depth'] = self.model.get_depth()
        if hasattr(self.model, 'get_n_leaves'):
            self.metrics['n_leaves'] = self.model.get_n_leaves()
        
        return self.metrics

# Class untuk Bagged Decision Tree
class BaggedDTDetailedModel(DTDetailedModel):
    def __init__(self, max_depth=None, criterion='gini', min_samples_split=2, random_state=42, n_estimators=10):
        super().__init__(max_depth, criterion, min_samples_split, random_state)
        
        base_estimator = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        
        self.model = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.metrics = {}
        self.confusion_matrix = None
        self.classification_report = None
        self.class_metrics = {}

# Ensemble voting berdasarkan KNN dan DT
def predict_with_ensemble(knn_model, dt_model, X):
    knn_pred = knn_model.predict(X)
    dt_pred = dt_model.predict(X)
    
    # Voting (jika sama gunakan prediksi itu, jika berbeda gunakan KNN)
    final_pred = []
    for i in range(len(X)):
        if knn_pred[i] == dt_pred[i]:
            final_pred.append(knn_pred[i])
        else:
            # Jika berbeda, gunakan KNN karena umumnya lebih baik
            final_pred.append(knn_pred[i])
    
    return final_pred

# Fungsi untuk menghasilkan gambar confusion matrix
def generate_confusion_matrix_image(cm, labels, title, cmap):
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap)
    plt.title(title)
    
    # Simpan plot ke buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert plot ke base64 untuk dapat ditampilkan di web
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_str

# Fungsi untuk menghasilkan gambar perbandingan metrik performance
def generate_performance_comparison(knn_metrics, dt_metrics, category_labels):
    # Create a figure for metrics comparison
    plt.figure(figsize=(10, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Prepare metrics data
    knn_values = []
    dt_values = []
    
    # Accuracy
    knn_values.append(knn_metrics['accuracy'])
    dt_values.append(dt_metrics['accuracy'])
    
    # Average precision, recall, f1 (weighted)
    for metric in ['precision', 'recall', 'f1-score']:
        knn_values.append(knn_metrics['classification_report']['weighted avg'][metric])
        dt_values.append(dt_metrics['classification_report']['weighted avg'][metric])
    
    plt.bar(x - width/2, knn_values, width, label='KNN', color='blue', alpha=0.7)
    plt.bar(x + width/2, dt_values, width, label='Decision Tree', color='green', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, metric_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(knn_values):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    for i, v in enumerate(dt_values):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # Simpan plot ke buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert plot ke base64 untuk dapat ditampilkan di web
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_str

# Fungsi untuk membuat grafik perbandingan akurasi training vs testing
def generate_train_test_comparison(knn_train_acc, dt_train_acc, knn_test_acc, dt_test_acc):
    plt.figure(figsize=(10, 6))
    algorithms = ['KNN', 'Decision Tree']
    training_accuracies = [knn_train_acc, dt_train_acc]
    testing_accuracies = [knn_test_acc, dt_test_acc]

    x = np.arange(len(algorithms))
    width = 0.35

    plt.bar(x - width/2, training_accuracies, width, label='Training Accuracy', color=['lightblue', 'lightgreen'], alpha=0.7)
    plt.bar(x + width/2, testing_accuracies, width, label='Testing Accuracy', color=['blue', 'green'], alpha=0.7)

    plt.xlabel('Algoritma')
    plt.ylabel('Akurasi')
    plt.title('Perbandingan Akurasi Training vs Testing')
    plt.xticks(x, algorithms)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Tambahkan label nilai
    for i, v in enumerate(training_accuracies):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    for i, v in enumerate(testing_accuracies):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    # Simpan plot ke buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert plot ke base64
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_str

# Endpoint untuk proses file Excel dan klasifikasi
@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.xlsx'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Simpan informasi file ke database
        upload_id = None
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cursor:
                    file_size = os.path.getsize(file_path)
                    
                    cursor.execute("""
                        INSERT INTO uploaded_files (filename, original_filename, file_size, processed)
                        VALUES (%s, %s, %s, %s)
                    """, (file_path, file.filename, file_size, False))
                    
                    conn.commit()
                    upload_id = cursor.lastrowid
                conn.close()
        except Exception as e:
            print(f"Error saving upload info: {str(e)}")
        
        try:
            # Baca file Excel
            df = pd.read_excel(file_path)
            
            # Cek kolom yang berisi judul skripsi
            title_column = None
            label_column = None
            
            # Cari kolom judul
            for col in df.columns:
                if 'judul' in str(col).lower() or 'title' in str(col).lower():
                    title_column = col
                    break
            
            # Jika tidak ada kolom spesifik, gunakan kolom pertama
            if title_column is None:
                title_column = df.columns[0]
            
            # Cari kolom label jika ada
            for col in df.columns:
                if 'label' in str(col).lower() or 'kategori' in str(col).lower() or 'category' in str(col).lower():
                    label_column = col
                    break
            
            print(f"Menggunakan kolom '{title_column}' sebagai judul skripsi")
            
            # Bersihkan data
            df = df.dropna(subset=[title_column])
            
            # Jika tidak ada kolom label, tambahkan label dummy berdasarkan kata kunci
            if label_column is None:
                print("Tidak ada kolom label ditemukan, menerapkan pelabelan otomatis berdasarkan kata kunci")
                
                # Fungsi pelabelan otomatis sederhana berdasarkan kata kunci
                def assign_label(title):
                    title_lower = str(title).lower()
                    
                    # Kata kunci untuk RPL
                    rpl_keywords = ['sistem', 'aplikasi', 'web', 'android', 'software', 'perangkat lunak', 
                                   'database', 'basis data', 'framework', 'e-commerce', 'situs web']
                    
                    # Kata kunci untuk Jaringan
                    network_keywords = ['jaringan', 'network', 'wifi', 'lan', 'wan', 'server', 'router', 
                                       'protocol', 'protokol', 'tcp/ip', 'keamanan jaringan', 'security']
                    
                    # Kata kunci untuk Multimedia
                    multimedia_keywords = ['multimedia', 'game', 'permainan', 'animasi', 'grafis', 'visual', 
                                          'audio', 'video', 'interaktif', 'augmented reality', 'virtual reality']
                    
                    # Hitung kemunculan kata kunci
                    rpl_count = sum(1 for keyword in rpl_keywords if keyword in title_lower)
                    network_count = sum(1 for keyword in network_keywords if keyword in title_lower)
                    multimedia_count = sum(1 for keyword in multimedia_keywords if keyword in title_lower)
                    
                    # Tentukan label berdasarkan jumlah kata kunci
                    counts = [rpl_count, network_count, multimedia_count]
                    max_index = counts.index(max(counts))
                    
                    if max_index == 0 and rpl_count > 0:
                        return 'RPL'
                    elif max_index == 1 and network_count > 0:
                        return 'Jaringan'
                    elif max_index == 2 and multimedia_count > 0:
                        return 'Multimedia'
                    else:
                        # Jika tidak ada kata kunci yang cocok, pilih secara acak
                        import random
                        return random.choice(['RPL', 'Jaringan', 'Multimedia'])
                
                # Terapkan pelabelan otomatis
                df['label'] = df[title_column].apply(assign_label)
                label_column = 'label'
            
            print(f"Menggunakan kolom '{label_column}' sebagai label kategori")
            
            # Generate embedding untuk semua judul dengan batch processing
            print("Generating embeddings in batches... (bisa memakan waktu)")
            embeddings = []
            titles = []
            labels = []
            
            # Batch size untuk menghindari memory issue
            batch_size = 10
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{len(df)//batch_size + 1}")
                
                for _, row in batch_df.iterrows():
                    title = row[title_column]
                    if pd.notna(title) and str(title).strip():  # Skip NaN atau string kosong
                        try:
                            # Ambil embedding dan tambahkan fitur domain
                            embedding = get_embedding(str(title))
                            embeddings.append(embedding)
                            titles.append(str(title))
                            labels.append(str(row[label_column]))
                        except Exception as e:
                            print(f"Error in processing title: {title}. Error: {str(e)}")
                
                # Clean up memory
                if i % 50 == 0 and i > 0:
                    gc.collect()
            
            # Jika tidak ada data yang valid
            if not embeddings:
                return jsonify({'error': 'No valid title data found in the uploaded file'}), 400
            
            print(f"Total data: {len(embeddings)} judul skripsi")
            
            # Tambahkan fitur domain ke embeddings
            print("Adding domain features...")
            enhanced_embeddings = []
            for i, emb in enumerate(embeddings):
                enhanced_emb = create_enhanced_features(emb, titles[i])
                enhanced_embeddings.append(enhanced_emb)
            
            # Oversampling untuk menangani class imbalance
            print("Performing oversampling for imbalanced classes...")
            balanced_embeddings, balanced_titles, balanced_labels = oversample_minority_classes(
                enhanced_embeddings, titles, labels
            )
            
            # Reduksi dimensi dengan PCA
            print("Applying dimensionality reduction...")
            reduced_embeddings = apply_pca(balanced_embeddings, n_components=min(100, len(balanced_embeddings[0])), fit=True)
            
            # Feature selection untuk memilih fitur terbaik
            print("Selecting best features...")
            selected_features = select_best_features(reduced_embeddings, balanced_labels, k=min(50, reduced_embeddings.shape[1]), fit=True)
            
            # Split data
            print("Splitting data into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                selected_features, balanced_labels, test_size=0.2, random_state=42, stratify=balanced_labels
            )
            
            # Hyperparameter tuning untuk KNN
            print("Optimizing KNN model...")
            optimized_knn = optimize_knn(X_train, y_train)
            
            # Hyperparameter tuning untuk Decision Tree
            print("Optimizing Decision Tree model...")
            optimized_dt = optimize_dt(X_train, y_train)
            
            # Bagging untuk Decision Tree
            print("Creating bagged Decision Tree model...")
            bagged_dt = create_bagged_dt(optimized_dt, n_estimators=10)
            
            print("Training KNN model...")
            # Train KNN with optimized parameters
            knn = KNNDetailedModel(n_neighbors=optimized_knn.n_neighbors, weights=optimized_knn.weights)
            knn.model = optimized_knn
            knn.fit(X_train, y_train)
            knn_metrics = knn.evaluate(X_test, y_test)
            knn_pred = knn.predict(X_test)
            knn_acc = knn_metrics['accuracy']
            
            print("Training Bagged Decision Tree model...")
            # Train Bagged Decision Tree
            dt = BaggedDTDetailedModel(
                max_depth=optimized_dt.max_depth if hasattr(optimized_dt, 'max_depth') else None,
                criterion=optimized_dt.criterion if hasattr(optimized_dt, 'criterion') else 'gini',
                min_samples_split=optimized_dt.min_samples_split if hasattr(optimized_dt, 'min_samples_split') else 2,
                random_state=42,
                n_estimators=10
            )
            dt.model = bagged_dt
            dt.fit(X_train, y_train)
            dt_metrics = dt.evaluate(X_test, y_test)
            dt_pred = dt.predict(X_test)
            dt_acc = dt_metrics['accuracy']
            
            # Prediksi pada data training untuk perbandingan
            print("Evaluating on training data...")
            knn_train_pred = knn.predict(X_train)
            dt_train_pred = dt.predict(X_train)
            knn_train_acc = accuracy_score(y_train, knn_train_pred)
            dt_train_acc = accuracy_score(y_train, dt_train_pred)
            
            print(f"Training Accuracy - KNN: {knn_train_acc:.4f}, DT: {dt_train_acc:.4f}")
            print(f"Testing Accuracy - KNN: {knn_acc:.4f}, DT: {dt_acc:.4f}")
            
            # Simpan informasi split data
            save_data_split_info(len(balanced_embeddings), len(y_train), len(y_test), upload_id)

            # Siapkan data untuk disimpan - PERBAIKAN MAPPING INDEKS
            # Karena train_test_split mengacak data, kita perlu menggunakan indeks yang benar
            from sklearn.model_selection import train_test_split as tts_indices
            
            # Buat indeks untuk mapping yang benar
            indices = list(range(len(balanced_embeddings)))
            train_indices, test_indices = tts_indices(
                indices, test_size=0.2, random_state=42, 
                stratify=balanced_labels
            )
            
            # Ambil judul berdasarkan indeks yang benar
            train_titles_for_db = [balanced_titles[i] for i in train_indices]
            test_titles_for_db = [balanced_titles[i] for i in test_indices]

            # Simpan data training dan testing
            all_titles = train_titles_for_db + test_titles_for_db
            all_labels = y_train + y_test
            all_knn_preds = knn_train_pred.tolist() + knn_pred.tolist()
            all_dt_preds = dt_train_pred.tolist() + dt_pred.tolist()
            all_data_types = ['training'] * len(y_train) + ['testing'] * len(y_test)

            save_training_testing_data(all_titles, all_labels, all_knn_preds, all_dt_preds, all_data_types, upload_id)

            # Simpan metrik detail
            # Training metrics
            save_model_metrics('KNN', 'training', knn_train_acc, 
                              knn.classification_report['macro avg']['precision'],
                              knn.classification_report['macro avg']['recall'],
                              knn.classification_report['macro avg']['f1-score'],
                              knn.classification_report['weighted avg']['precision'],
                              knn.classification_report['weighted avg']['recall'],
                              knn.classification_report['weighted avg']['f1-score'],
                              upload_id)

            save_model_metrics('Decision Tree', 'training', dt_train_acc,
                              dt.classification_report['macro avg']['precision'],
                              dt.classification_report['macro avg']['recall'],
                              dt.classification_report['macro avg']['f1-score'],
                              dt.classification_report['weighted avg']['precision'],
                              dt.classification_report['weighted avg']['recall'],
                              dt.classification_report['weighted avg']['f1-score'],
                              upload_id)

            # Testing metrics  
            save_model_metrics('KNN', 'testing', knn_acc,
                              knn.classification_report['macro avg']['precision'],
                              knn.classification_report['macro avg']['recall'],
                              knn.classification_report['macro avg']['f1-score'],
                              knn.classification_report['weighted avg']['precision'],
                              knn.classification_report['weighted avg']['recall'],
                              knn.classification_report['weighted avg']['f1-score'],
                              upload_id)

            save_model_metrics('Decision Tree', 'testing', dt_acc,
                              dt.classification_report['macro avg']['precision'],
                              dt.classification_report['macro avg']['recall'],
                              dt.classification_report['macro avg']['f1-score'],
                              dt.classification_report['weighted avg']['precision'],
                              dt.classification_report['weighted avg']['recall'],
                              dt.classification_report['weighted avg']['f1-score'],
                              upload_id)

            # Simpan confusion matrix data ke database
            save_confusion_matrix_data(y_train, knn_train_pred.tolist(), 'KNN', 'training', upload_id)
            save_confusion_matrix_data(y_test, knn_pred.tolist(), 'KNN', 'testing', upload_id)
            save_confusion_matrix_data(y_train, dt_train_pred.tolist(), 'Decision Tree', 'training', upload_id)
            save_confusion_matrix_data(y_test, dt_pred.tolist(), 'Decision Tree', 'testing', upload_id)
            
            # Simpan model yang dilatih
            global trained_knn, trained_dt, train_features, train_labels
            trained_knn = knn.model
            trained_dt = dt.model
            train_features = X_train
            train_labels = y_train
            
            # Simpan model ke disk dengan semua komponen
            models_data = {
                'knn': knn.model,
                'dt': dt.model,
                'train_features': X_train,
                'train_labels': y_train,
                'pca_model': pca_model,
                'feature_selector': feature_selector,
                'preprocessing_params': {
                    'max_length': MAX_LENGTH,
                    'embedding_method': 'combined_pooling'
                }
            }
            save_complete_models(models_data)
            
            # Simpan cache embedding ke disk
            save_embedding_cache()
            
            print("Generating visualizations...")
            # Dapatkan kategori unik
            unique_labels = sorted(list(set(labels)))
            
            # Confusion Matrix
            knn_cm = knn.confusion_matrix
            dt_cm = dt.confusion_matrix
            
            # Confusion matrix untuk training data
            knn_train_cm = confusion_matrix(y_train, knn_train_pred, labels=unique_labels)
            dt_train_cm = confusion_matrix(y_train, dt_train_pred, labels=unique_labels)
            
            # Generate gambar confusion matrix
            knn_cm_img = generate_confusion_matrix_image(knn_cm, unique_labels, "Confusion Matrix - KNN", 'Blues')
            dt_cm_img = generate_confusion_matrix_image(dt_cm, unique_labels, "Confusion Matrix - Decision Tree", 'Greens')
            
            # Generate combined confusion matrix
            combined_cm_img = generate_combined_confusion_matrix(knn_train_cm, knn_cm, dt_train_cm, dt_cm, unique_labels)
            
            # Generate perbandingan metrik performa
            performance_metrics = {
                'accuracy': knn_acc,
                'classification_report': knn.classification_report
            }
            dt_performance_metrics = {
                'accuracy': dt_acc,
                'classification_report': dt.classification_report
            }
            performance_comparison_img = generate_performance_comparison(
                performance_metrics, 
                dt_performance_metrics, 
                unique_labels
            )
            
            # Buat grafik perbandingan akurasi
            plt.figure(figsize=(8, 5))
            algorithms = ['KNN', 'Decision Tree']
            accuracies = [knn_acc, dt_acc]
            plt.bar(algorithms, accuracies, color=['blue', 'green'])
            plt.ylim(0, 1)
            plt.title('Perbandingan Akurasi Model (Testing Data)')
            plt.ylabel('Akurasi')
            
            # Tambahkan label nilai
            for i, v in enumerate(accuracies):
                plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
            
            # Simpan plot ke buffer
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert plot ke base64
            accuracy_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # Generate grafik perbandingan training vs testing
            train_test_comparison_img = generate_train_test_comparison(
                knn_train_acc, dt_train_acc, knn_acc, dt_acc
            )
            
            # Persiapkan hasil prediksi untuk ditampilkan - DATA TESTING
            # Gunakan indeks yang sudah diperbaiki
            test_titles = test_titles_for_db
            train_titles = train_titles_for_db
            
            # Data Testing Results
            results_table = []
            for i in range(len(y_test)):
                results_table.append({
                    'title': test_titles[i][:100] + '...' if len(test_titles[i]) > 100 else test_titles[i],
                    'full_title': test_titles[i],
                    'actual': y_test[i],
                    'knn_pred': knn_pred[i],
                    'dt_pred': dt_pred[i],
                    'data_type': 'testing'
                })
            
            # Data Training Results
            training_results_table = []
            for i in range(len(y_train)):
                training_results_table.append({
                    'title': train_titles[i][:100] + '...' if len(train_titles[i]) > 100 else train_titles[i],
                    'full_title': train_titles[i],
                    'actual': y_train[i],
                    'knn_pred': knn_train_pred[i],
                    'dt_pred': dt_train_pred[i],
                    'data_type': 'training'
                })
            
            # Detailed metrics for each model
            knn_detailed = {
                'overall': knn_metrics,
                'per_class': knn.class_metrics,
                'classification_report': knn.classification_report,
                'training_accuracy': knn_train_acc
            }
            
            dt_detailed = {
                'overall': dt_metrics,
                'per_class': dt.class_metrics,
                'classification_report': dt.classification_report,
                'tree_depth': dt.metrics.get('tree_depth', 0) if hasattr(dt, 'metrics') else 0,
                'n_leaves': dt.metrics.get('n_leaves', 0) if hasattr(dt, 'metrics') else 0,
                'training_accuracy': dt_train_acc
            }
            
            # Setelah proses klasifikasi selesai, simpan data ke database
            try:
                conn = get_db_connection()
                if conn:
                    with conn.cursor() as cursor:
                       # Simpan performa model dengan upload_id
                        save_model_performance('KNN', knn_acc, {'n_neighbors': knn.n_neighbors}, upload_id)
                        save_model_performance('Decision Tree', dt_acc, {  
                            'max_depth': dt.max_depth, 
                            'criterion': dt.criterion,
                            'n_estimators': dt.n_estimators if hasattr(dt, 'n_estimators') else 10,
                            'is_bagged': True  
                        }, upload_id)
                        
                        # Simpan data judul dan hasil klasifikasi dengan upload_id
                        for i, (title, label) in enumerate(zip(titles, labels)):
                            # Dapatkan ID kategori
                            cursor.execute("SELECT id FROM categories WHERE name = %s", (label,))
                            category_result = cursor.fetchone()
                            category_id = category_result['id'] if category_result else None
                            
                            if category_id:
                                # Cek apakah judul sudah ada
                                cursor.execute("SELECT id FROM thesis_titles WHERE title = %s", (title,))
                                existing = cursor.fetchone()
                                
                                if not existing:
                                    # Simpan judul baru dengan upload_id
                                    cursor.execute("""
                                        INSERT INTO thesis_titles (title, category_id, upload_file_id)
                                        VALUES (%s, %s, %s)
                                    """, (title, category_id, upload_id))
                        
                        # Simpan hasil prediksi test set dengan upload_id
                        for i, result in enumerate(results_table):
                            save_prediction_to_db(
                                result['full_title'], 
                                result['actual'], 
                                result['knn_pred'], 
                                result['dt_pred'],
                                confidence=0.9,  # Default confidence
                                upload_file_id=upload_id
                            )
                        
                        conn.commit()
                        
                        # Analisis kata kunci per kategori dengan upload_id
                        for category in unique_labels:
                            category_titles = [titles[i] for i in range(len(titles)) if labels[i] == category]
                            analyze_keywords(category, category_titles, upload_id)
                            
                        # Update status file upload menjadi processed
                        if upload_id:
                            cursor.execute("UPDATE uploaded_files SET processed = 1 WHERE id = %s", (upload_id,))
                            conn.commit()
                    conn.close()
            except Exception as e:
                print(f"Error saving data to database: {str(e)}")
            
            print("Processing completed successfully!")
            
            # Simpan visualisasi ke database jika tabel model_visualizations ada
            try:
                conn = get_db_connection()
                if conn:
                    with conn.cursor() as cursor:
                        # Cek apakah tabel model_visualizations ada
                        cursor.execute("SHOW TABLES LIKE 'model_visualizations'")
                        table_exists = cursor.fetchone()
                        
                        # Debug logs
                        print(f"Upload ID for visualizations: {upload_id}")
                        print(f"KNN CM Image size: {len(knn_cm_img) if knn_cm_img else 'None'}")
                        print(f"DT CM Image size: {len(dt_cm_img) if dt_cm_img else 'None'}")
                        print(f"Combined CM Image size: {len(combined_cm_img) if combined_cm_img else 'None'}")
                        
                        if table_exists:
                            # Cek apakah sudah ada entri untuk upload_id ini
                            if upload_id:
                                cursor.execute("SELECT id FROM model_visualizations WHERE upload_file_id = %s", (upload_id,))
                                existing = cursor.fetchone()
                                
                                if existing:
                                    # Update visualisasi yang sudah ada
                                    cursor.execute("""
                                        UPDATE model_visualizations 
                                        SET knn_cm_img = %s, dt_cm_img = %s, performance_comparison_img = %s, 
                                            accuracy_img = %s, train_test_comparison_img = %s, combined_cm_img = %s
                                        WHERE upload_file_id = %s
                                    """, (
                                        knn_cm_img,
                                        dt_cm_img,
                                        performance_comparison_img,
                                        accuracy_img,
                                        train_test_comparison_img,
                                        combined_cm_img,
                                        upload_id
                                    ))
                                else:
                                    # Simpan visualisasi baru
                                    cursor.execute("""
                                        INSERT INTO model_visualizations 
                                        (upload_file_id, knn_cm_img, dt_cm_img, performance_comparison_img, 
                                         accuracy_img, train_test_comparison_img, combined_cm_img)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    """, (
                                        upload_id,
                                        knn_cm_img,
                                        dt_cm_img,
                                        performance_comparison_img,
                                        accuracy_img,
                                        train_test_comparison_img,
                                        combined_cm_img
                                    ))
                                
                                conn.commit()
                                print(f"Visualizations saved to database successfully for upload_id: {upload_id}")
                            else:
                                print("Warning: No upload_id available, skipping visualization save")
                    conn.close()
            except Exception as e:
                print(f"Error saving visualizations to database: {str(e)}")
                # Print full stack trace for better debugging
                import traceback
                traceback.print_exc()

            # Kirim hasil ke frontend dengan konversi nilai numpy
            return jsonify(convert_numpy_types({
                'knn_accuracy': knn_acc,
                'dt_accuracy': dt_acc,
                'knn_train_accuracy': knn_train_acc,
                'dt_train_accuracy': dt_train_acc,
                'knn_cm_img': knn_cm_img,
                'dt_cm_img': dt_cm_img,
                'accuracy_img': accuracy_img,
                'performance_comparison_img': performance_comparison_img,
                'train_test_comparison_img': train_test_comparison_img,
                'combined_cm_img': combined_cm_img,
                'results_table': results_table,  # Data testing
                'training_results_table': training_results_table,  # Data training
                'categories': unique_labels,
                'knn_detailed': knn_detailed,
                'dt_detailed': dt_detailed,
                'upload_id': upload_id,
                'data_split_info': {
                    'total_data': len(balanced_embeddings),
                    'training_size': len(y_train),
                    'testing_size': len(y_test),
                    'training_percentage': (len(y_train) / len(balanced_embeddings)) * 100,
                    'testing_percentage': (len(y_test) / len(balanced_embeddings)) * 100
                }
            }))
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

# Endpoint untuk memprediksi judul baru
@app.route('/predict', methods=['POST'])
def predict_title():
    data = request.json
    
    if 'title' not in data:
        return jsonify({'error': 'No title provided'}), 400
    
    try:
        title = data['title']
        upload_id = data.get('upload_id', None)
        
        # Cek apakah model sudah dilatih
        if trained_knn is None or trained_dt is None:
            return jsonify({'error': 'Models not trained yet. Please upload and process training data first.'}), 400
        
        # Buat embedding untuk judul baru
        embedding = get_embedding(title)
        enhanced_embedding = create_enhanced_features(embedding, title)
        
        # Terapkan transformasi yang sama seperti saat training
        if pca_model is None or feature_selector is None:
            return jsonify({'error': 'Feature transformation models not available. Please retrain the models.'}), 400
        
        # Reduksi dimensi dan seleksi fitur
        reduced_embedding = pca_model.transform([enhanced_embedding])
        selected_features = feature_selector.transform(reduced_embedding)
        
        # Prediksi dengan kedua model
        knn_pred = trained_knn.predict(selected_features)[0]
        dt_pred = trained_dt.predict(selected_features)[0]
        
        # Dapatkan confidence score jika tersedia
        knn_confidence = 0.0
        dt_confidence = 0.0
        
        try:
            if hasattr(trained_knn, 'predict_proba'):
                knn_proba = trained_knn.predict_proba(selected_features)[0]
                knn_confidence = float(max(knn_proba))
            
            if hasattr(trained_dt, 'predict_proba'):
                dt_proba = trained_dt.predict_proba(selected_features)[0]
                dt_confidence = float(max(dt_proba))
        except:
            pass
        
        # Ensemble prediction (voting)
        ensemble_pred = knn_pred if knn_pred == dt_pred else knn_pred  # Default ke KNN jika berbeda
        
        # Simpan hasil prediksi ke database
        prediction_id = save_prediction_to_db(
            title, 
            'Unknown',  # Actual category tidak diketahui untuk prediksi baru
            knn_pred, 
            dt_pred, 
            max(knn_confidence, dt_confidence),
            upload_id
        )
        
        return jsonify(convert_numpy_types({
            'title': title,
            'knn_prediction': knn_pred,
            'dt_prediction': dt_pred,
            'ensemble_prediction': ensemble_pred,
            'knn_confidence': knn_confidence,
            'dt_confidence': dt_confidence,
            'prediction_id': prediction_id
        }))
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan riwayat prediksi
@app.route('/history', methods=['GET'])
def get_prediction_history():
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT p.id, p.title, p.confidence, p.created_at,
                           c1.name as actual_category,
                           c2.name as knn_prediction,
                           c3.name as dt_prediction,
                           uf.original_filename
                    FROM predictions p
                    LEFT JOIN categories c1 ON p.actual_category_id = c1.id
                    LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
                    LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
                    LEFT JOIN uploaded_files uf ON p.upload_file_id = uf.id
                    ORDER BY p.created_at DESC
                    LIMIT 100
                """)
                
                predictions = cursor.fetchall()
                return jsonify(convert_numpy_types(predictions))
            conn.close()
    except Exception as e:
        print(f"Error getting prediction history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan performa model
@app.route('/model_performance', methods=['GET'])
def get_model_performance():
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT mp.id, mp.model_name, mp.accuracy, mp.parameters, mp.created_at,
                           uf.original_filename
                    FROM model_performances mp
                    LEFT JOIN uploaded_files uf ON mp.upload_file_id = uf.id
                    ORDER BY mp.created_at DESC
                    LIMIT 50
                """)
                
                performances = cursor.fetchall()
                
                # Parse JSON parameters
                for perf in performances:
                    if perf['parameters']:
                        try:
                            perf['parameters'] = json.loads(perf['parameters'])
                        except:
                            perf['parameters'] = {}
                
                return jsonify(convert_numpy_types(performances))
            conn.close()
    except Exception as e:
        print(f"Error getting model performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan analisis kata kunci
@app.route('/keyword_analysis', methods=['GET'])
def get_keyword_analysis():
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT ka.id, ka.keyword, ka.frequency, ka.created_at,
                           c.name as category_name,
                           uf.original_filename
                    FROM keyword_analysis ka
                    LEFT JOIN categories c ON ka.category_id = c.id
                    LEFT JOIN uploaded_files uf ON ka.upload_file_id = uf.id
                    ORDER BY c.name, ka.frequency DESC
                """)
                
                keywords = cursor.fetchall()
                
                # Group by category
                grouped_keywords = {}
                for keyword in keywords:
                    category = keyword['category_name']
                    if category not in grouped_keywords:
                        grouped_keywords[category] = []
                    grouped_keywords[category].append(keyword)
                
                return jsonify(convert_numpy_types(grouped_keywords))
            conn.close()
    except Exception as e:
        print(f"Error getting keyword analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan statistik umum
@app.route('/statistics', methods=['GET'])
def get_statistics():
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                stats = {}
                
                # Total prediksi
                cursor.execute("SELECT COUNT(*) as total FROM predictions")
                stats['total_predictions'] = cursor.fetchone()['total']
                
                # Total file yang diupload
                cursor.execute("SELECT COUNT(*) as total FROM uploaded_files")
                stats['total_uploads'] = cursor.fetchone()['total']
                
                # Total judul skripsi
                cursor.execute("SELECT COUNT(*) as total FROM thesis_titles")
                stats['total_thesis'] = cursor.fetchone()['total']
                
                # Distribusi kategori
                cursor.execute("""
                    SELECT c.name, COUNT(tt.id) as count
                    FROM categories c
                    LEFT JOIN thesis_titles tt ON c.id = tt.category_id
                    GROUP BY c.id, c.name
                """)
                stats['category_distribution'] = cursor.fetchall()
                
                # Akurasi rata-rata model
                cursor.execute("""
                    SELECT model_name, AVG(accuracy) as avg_accuracy, COUNT(*) as count
                    FROM model_performances
                    GROUP BY model_name
                """)
                stats['model_accuracies'] = cursor.fetchall()
                
                # Prediksi terbaru
                cursor.execute("""
                    SELECT p.title, c1.name as knn_pred, c2.name as dt_pred, p.created_at
                    FROM predictions p
                    LEFT JOIN categories c1 ON p.knn_prediction_id = c1.id
                    LEFT JOIN categories c2 ON p.dt_prediction_id = c2.id
                    ORDER BY p.created_at DESC
                    LIMIT 5
                """)
                stats['recent_predictions'] = cursor.fetchall()
                
                return jsonify(convert_numpy_types(stats))
            conn.close()
    except Exception as e:
        print(f"Error getting statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan visualisasi yang tersimpan
@app.route('/visualizations/<int:upload_id>', methods=['GET'])
def get_visualizations(upload_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT knn_cm_img, dt_cm_img, performance_comparison_img, accuracy_img, 
                           train_test_comparison_img, combined_cm_img
                    FROM model_visualizations
                    WHERE upload_file_id = %s
                """, (upload_id,))
                
                result = cursor.fetchone()
                if result:
                    return jsonify(convert_numpy_types(result))
                else:
                    return jsonify({'error': 'Visualizations not found'}), 404
            conn.close()
    except Exception as e:
        print(f"Error getting visualizations: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan daftar file yang diupload
@app.route('/uploads', methods=['GET'])
def get_uploads():
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, original_filename, file_size, processed, created_at
                    FROM uploaded_files
                    ORDER BY created_at DESC
                """)
                
                uploads = cursor.fetchall()
                return jsonify(convert_numpy_types(uploads))
            conn.close()
    except Exception as e:
        print(f"Error getting uploads: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk menghapus data upload dan hasil terkait
@app.route('/delete_upload/<int:upload_id>', methods=['DELETE'])
def delete_upload(upload_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                # Hapus data terkait secara berurutan (karena foreign key constraints)
                cursor.execute("DELETE FROM keyword_analysis WHERE upload_file_id = %s", (upload_id,))
                cursor.execute("DELETE FROM predictions WHERE upload_file_id = %s", (upload_id,))
                cursor.execute("DELETE FROM model_performances WHERE upload_file_id = %s", (upload_id,))
                cursor.execute("DELETE FROM thesis_titles WHERE upload_file_id = %s", (upload_id,))
                cursor.execute("DELETE FROM training_data WHERE upload_file_id = %s", (upload_id,))
                cursor.execute("DELETE FROM model_metrics WHERE upload_file_id = %s", (upload_id,))
                cursor.execute("DELETE FROM confusion_matrix_data WHERE upload_file_id = %s", (upload_id,))
                cursor.execute("DELETE FROM data_split_info WHERE upload_file_id = %s", (upload_id,))
                
                # Hapus visualisasi jika ada
                cursor.execute("DELETE FROM model_visualizations WHERE upload_file_id = %s", (upload_id,))
                
                # Hapus record upload file
                cursor.execute("DELETE FROM uploaded_files WHERE id = %s", (upload_id,))
                
                conn.commit()
                
                return jsonify({'message': 'Upload data deleted successfully'})
            conn.close()
    except Exception as e:
        print(f"Error deleting upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk reset semua model dan data
@app.route('/reset', methods=['POST'])
def reset_models():
    try:
        global trained_knn, trained_dt, pca_model, feature_selector, embedding_cache, train_features, train_labels
        
        # Reset variabel global
        trained_knn = None
        trained_dt = None
        pca_model = None
        feature_selector = None
        embedding_cache = {}
        train_features = []
        train_labels = []
        
        # Hapus file model
        for filename in [MODEL_FILENAME, PCA_FILENAME, SELECTOR_FILENAME, EMBEDDING_CACHE_FILENAME]:
            if os.path.exists(filename):
                os.remove(filename)
        
        return jsonify({'message': 'Models and cache reset successfully'})
        
    except Exception as e:
        print(f"Error resetting models: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan status model
@app.route('/model_status', methods=['GET'])
def get_model_status():
    try:
        status = {
            'knn_trained': trained_knn is not None,
            'dt_trained': trained_dt is not None,
            'pca_available': pca_model is not None,
            'feature_selector_available': feature_selector is not None,
            'embedding_cache_size': len(embedding_cache),
            'training_data_size': len(train_features) if train_features else 0
        }
        
        return jsonify(convert_numpy_types(status))
        
        
    except Exception as e:
        print(f"Error getting model status: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan confusion matrix data dari database
@app.route('/confusion_matrix/<int:upload_id>', methods=['GET'])
def get_confusion_matrix_data(upload_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT cmd.model_name, cmd.count,
                           c1.name as actual_category,
                           c2.name as predicted_category
                    FROM confusion_matrix_data cmd
                    LEFT JOIN categories c1 ON cmd.actual_category_id = c1.id
                    LEFT JOIN categories c2 ON cmd.predicted_category_id = c2.id
                    WHERE cmd.upload_file_id = %s
                    ORDER BY cmd.model_name, c1.name, c2.name
                """, (upload_id,))
                
                cm_data = cursor.fetchall()
                
                # Group by model
                grouped_data = {}
                for row in cm_data:
                    model = row['model_name']
                    if model not in grouped_data:
                        grouped_data[model] = []
                    grouped_data[model].append(row)
                
                return jsonify(convert_numpy_types(grouped_data))
            conn.close()
    except Exception as e:
        print(f"Error getting confusion matrix data: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan metrik detail model
@app.route('/model_metrics/<int:upload_id>', methods=['GET'])
def get_model_metrics(upload_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT model_name, data_type, accuracy, precision_macro, recall_macro, f1_macro,
                           precision_weighted, recall_weighted, f1_weighted, created_at
                    FROM model_metrics
                    WHERE upload_file_id = %s
                    ORDER BY model_name, data_type
                """, (upload_id,))
                
                metrics = cursor.fetchall()
                
                # Group by model and data type
                grouped_metrics = {}
                for metric in metrics:
                    model = metric['model_name']
                    data_type = metric['data_type']
                    key = f"{model}_{data_type}"
                    grouped_metrics[key] = metric
                
                return jsonify(convert_numpy_types(grouped_metrics))
            conn.close()
    except Exception as e:
        print(f"Error getting model metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan data training/testing
@app.route('/training_data/<int:upload_id>', methods=['GET'])
def get_training_data(upload_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT td.title, td.data_type, td.is_correct_knn, td.is_correct_dt,
                           c1.name as actual_category,
                           c2.name as knn_prediction,
                           c3.name as dt_prediction,
                           td.created_at
                    FROM training_data td
                    LEFT JOIN categories c1 ON td.actual_category_id = c1.id
                    LEFT JOIN categories c2 ON td.knn_prediction_id = c2.id
                    LEFT JOIN categories c3 ON td.dt_prediction_id = c3.id
                    WHERE td.upload_file_id = %s
                    ORDER BY td.data_type, td.created_at
                """, (upload_id,))
                
                training_data = cursor.fetchall()
                
                # Separate training and testing data
                result = {
                    'training': [],
                    'testing': []
                }
                
                for row in training_data:
                    data_type = row['data_type']
                    if data_type in result:
                        result[data_type].append(row)
                
                return jsonify(convert_numpy_types(result))
            conn.close()
    except Exception as e:
        print(f"Error getting training data: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan informasi split data
@app.route('/data_split_info/<int:upload_id>', methods=['GET'])
def get_data_split_info(upload_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT total_data, training_size, testing_size, training_percentage, 
                           testing_percentage, split_method, random_state, created_at
                    FROM data_split_info
                    WHERE upload_file_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (upload_id,))
                
                split_info = cursor.fetchone()
                
                if split_info:
                    return jsonify(convert_numpy_types(split_info))
                else:
                    return jsonify({'error': 'Data split info not found'}), 404
            conn.close()
    except Exception as e:
        print(f"Error getting data split info: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mendapatkan kategori yang tersedia
@app.route('/categories', methods=['GET'])
def get_categories():
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, name, description, created_at
                    FROM categories
                    ORDER BY name
                """)
                
                categories = cursor.fetchall()
                return jsonify(convert_numpy_types(categories))
            conn.close()
    except Exception as e:
        print(f"Error getting categories: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk menambah kategori baru
@app.route('/categories', methods=['POST'])
def add_category():
    try:
        data = request.json
        if 'name' not in data:
            return jsonify({'error': 'Category name is required'}), 400
        
        name = data['name']
        description = data.get('description', '')
        
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                # Cek apakah kategori sudah ada
                cursor.execute("SELECT id FROM categories WHERE name = %s", (name,))
                existing = cursor.fetchone()
                
                if existing:
                    return jsonify({'error': 'Category already exists'}), 400
                
                # Tambah kategori baru
                cursor.execute("""
                    INSERT INTO categories (name, description)
                    VALUES (%s, %s)
                """, (name, description))
                
                conn.commit()
                category_id = cursor.lastrowid
                
                return jsonify({
                    'id': category_id,
                    'name': name,
                    'description': description,
                    'message': 'Category added successfully'
                })
            conn.close()
    except Exception as e:
        print(f"Error adding category: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk batch prediction (prediksi multiple judul sekaligus)
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        
        if 'titles' not in data or not isinstance(data['titles'], list):
            return jsonify({'error': 'Titles array is required'}), 400
        
        titles = data['titles']
        upload_id = data.get('upload_id', None)
        
        # Cek apakah model sudah dilatih
        if trained_knn is None or trained_dt is None:
            return jsonify({'error': 'Models not trained yet. Please upload and process training data first.'}), 400
        
        if pca_model is None or feature_selector is None:
            return jsonify({'error': 'Feature transformation models not available. Please retrain the models.'}), 400
        
        results = []
        
        for title in titles:
            if not title or not title.strip():
                continue
                
            try:
                # Buat embedding untuk judul
                embedding = get_embedding(title)
                enhanced_embedding = create_enhanced_features(embedding, title)
                
                # Terapkan transformasi yang sama seperti saat training
                reduced_embedding = pca_model.transform([enhanced_embedding])
                selected_features = feature_selector.transform(reduced_embedding)
                
                # Prediksi dengan kedua model
                knn_pred = trained_knn.predict(selected_features)[0]
                dt_pred = trained_dt.predict(selected_features)[0]
                
                # Dapatkan confidence score
                knn_confidence = 0.0
                dt_confidence = 0.0
                
                try:
                    if hasattr(trained_knn, 'predict_proba'):
                        knn_proba = trained_knn.predict_proba(selected_features)[0]
                        knn_confidence = float(max(knn_proba))
                    
                    if hasattr(trained_dt, 'predict_proba'):
                        dt_proba = trained_dt.predict_proba(selected_features)[0]
                        dt_confidence = float(max(dt_proba))
                except:
                    pass
                
                # Ensemble prediction
                ensemble_pred = knn_pred if knn_pred == dt_pred else knn_pred
                
                result = {
                    'title': title,
                    'knn_prediction': knn_pred,
                    'dt_prediction': dt_pred,
                    'ensemble_prediction': ensemble_pred,
                    'knn_confidence': knn_confidence,
                    'dt_confidence': dt_confidence
                }
                
                results.append(result)
                
                # Simpan hasil prediksi ke database
                save_prediction_to_db(
                    title, 
                    'Unknown',
                    knn_pred, 
                    dt_pred, 
                    max(knn_confidence, dt_confidence),
                    upload_id
                )
                
            except Exception as e:
                print(f"Error processing title '{title}': {str(e)}")
                results.append({
                    'title': title,
                    'error': str(e)
                })
        
        return jsonify(convert_numpy_types({
            'results': results,
            'total_processed': len(results),
            'upload_id': upload_id
        }))
        
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk export hasil ke Excel
@app.route('/export_results/<int:upload_id>', methods=['GET'])
def export_results(upload_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                # Ambil data training/testing
                cursor.execute("""
                    SELECT td.title, td.data_type, td.is_correct_knn, td.is_correct_dt,
                           c1.name as actual_category,
                           c2.name as knn_prediction,
                           c3.name as dt_prediction
                    FROM training_data td
                    LEFT JOIN categories c1 ON td.actual_category_id = c1.id
                    LEFT JOIN categories c2 ON td.knn_prediction_id = c2.id
                    LEFT JOIN categories c3 ON td.dt_prediction_id = c3.id
                    WHERE td.upload_file_id = %s
                    ORDER BY td.data_type, td.title
                """, (upload_id,))
                
                data = cursor.fetchall()
                
                if not data:
                    return jsonify({'error': 'No data found for this upload'}), 404
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Simpan ke file Excel
                filename = f"classification_results_{upload_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Sheet untuk semua data
                    df.to_excel(writer, sheet_name='All Results', index=False)
                    
                    # Sheet terpisah untuk training dan testing
                    training_data = df[df['data_type'] == 'training']
                    testing_data = df[df['data_type'] == 'testing']
                    
                    if not training_data.empty:
                        training_data.to_excel(writer, sheet_name='Training Data', index=False)
                    
                    if not testing_data.empty:
                        testing_data.to_excel(writer, sheet_name='Testing Data', index=False)
                
                return jsonify({
                    'message': 'Results exported successfully',
                    'filename': filename,
                    'filepath': filepath,
                    'total_records': len(data)
                })
            conn.close()
    except Exception as e:
        print(f"Error exporting results: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'indobert': model is not None,
            'knn': trained_knn is not None,
            'dt': trained_dt is not None
        },
        'database_connection': get_db_connection() is not None
    })

# Endpoint untuk mendapatkan informasi sistem
@app.route('/system_info', methods=['GET'])
def get_system_info():
    try:
        import psutil
        import torch
        
        # Informasi memori
        memory = psutil.virtual_memory()
        
        # Informasi GPU jika tersedia
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'
            }
        else:
            gpu_info = {'gpu_available': False}
        
        # Informasi cache dan model
        cache_info = {
            'embedding_cache_size': len(embedding_cache),
            'cache_file_exists': os.path.exists(EMBEDDING_CACHE_FILENAME),
            'model_files_exist': {
                'main_model': os.path.exists(MODEL_FILENAME),
                'pca_model': os.path.exists(PCA_FILENAME),
                'feature_selector': os.path.exists(SELECTOR_FILENAME)
            }
        }
        
        return jsonify({
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'gpu': gpu_info,
            'cache': cache_info,
            'models_status': {
                'knn_trained': trained_knn is not None,
                'dt_trained': trained_dt is not None,
                'pca_available': pca_model is not None,
                'feature_selector_available': feature_selector is not None
            },
            'directories': {
                'upload_folder': UPLOAD_FOLDER,
                'model_folder': MODEL_FOLDER,
                'cache_folder': CACHE_FOLDER
            }
        })
        
    except ImportError:
        return jsonify({
            'error': 'psutil not installed, limited system info available',
            'basic_info': {
                'models_status': {
                    'knn_trained': trained_knn is not None,
                    'dt_trained': trained_dt is not None,
                    'pca_available': pca_model is not None,
                    'feature_selector_available': feature_selector is not None
                },
                'cache_size': len(embedding_cache)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handler
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large'}), 413

# Cleanup function yang dipanggil saat aplikasi ditutup
import atexit

def cleanup():
    print("Cleaning up...")
    # Simpan cache embedding terakhir kali
    save_embedding_cache()
    
    # Bersihkan memori
    global embedding_cache, trained_knn, trained_dt
    if embedding_cache:
        embedding_cache.clear()
    
    # Force garbage collection
    gc.collect()
    
    print("Cleanup completed")

atexit.register(cleanup)

# Konfigurasi tambahan untuk Flask
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['JSON_SORT_KEYS'] = False

if __name__ == '__main__':
    print("=" * 60)
    print("STARTING FLASK APPLICATION - THESIS CLASSIFICATION SYSTEM")
    print("=" * 60)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Model folder: {MODEL_FOLDER}")
    print(f"Cache folder: {CACHE_FOLDER}")
    print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print(f"Database config: {DB_CONFIG['host']}:{DB_CONFIG['db']}")
    
    # Cek koneksi database
    test_conn = get_db_connection()
    if test_conn:
        print("✓ Database connection successful")
        test_conn.close()
    else:
        print("✗ Database connection failed")
    
    # Cek model status
    print(f"✓ IndoBERT model loaded: {model is not None}")
    print(f"✓ KNN model trained: {trained_knn is not None}")
    print(f"✓ DT model trained: {trained_dt is not None}")
    print(f"✓ Embedding cache size: {len(embedding_cache)}")
    
    print("=" * 60)
    print("Server starting on http://0.0.0.0:5000")
    print("Available endpoints:")
    print("  POST /process - Upload and process Excel file")
    print("  POST /predict - Predict single title")
    print("  POST /batch_predict - Predict multiple titles")
    print("  GET  /history - Get prediction history")
    print("  GET  /statistics - Get system statistics")
    print("  GET  /health - Health check")
    print("  GET  /system_info - System information")
    print("=" * 60)
    
    # Jalankan aplikasi Flask
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
