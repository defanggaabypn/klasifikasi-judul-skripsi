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
            
            # Generate gambar confusion matrix
            knn_cm_img = generate_confusion_matrix_image(knn_cm, unique_labels, "Confusion Matrix - KNN", 'Blues')
            dt_cm_img = generate_confusion_matrix_image(dt_cm, unique_labels, "Confusion Matrix - Decision Tree", 'Greens')
            
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
            plt.title('Perbandingan Akurasi Model')
            plt.ylabel('Akurasi')
            
            # Simpan plot ke buffer
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert plot ke base64
            accuracy_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            # Persiapkan hasil prediksi untuk ditampilkan
            test_titles = [balanced_titles[i] for i in range(len(balanced_titles) - len(y_test), len(balanced_titles))]
            results_table = []
            
            for i in range(len(y_test)):
                results_table.append({
                    'title': test_titles[i][:100] + '...' if len(test_titles[i]) > 100 else test_titles[i],
                    'full_title': test_titles[i],
                    'actual': y_test[i],
                    'knn_pred': knn_pred[i],
                    'dt_pred': dt_pred[i]
                })
            
            # Detailed metrics for each model
            knn_detailed = {
                'overall': knn_metrics,
                'per_class': knn.class_metrics,
                'classification_report': knn.classification_report
            }
            
            dt_detailed = {
                'overall': dt_metrics,
                'per_class': dt.class_metrics,
                'classification_report': dt.classification_report,
                'tree_depth': dt.metrics.get('tree_depth', 0) if hasattr(dt, 'metrics') else 0,
                'n_leaves': dt.metrics.get('n_leaves', 0) if hasattr(dt, 'metrics') else 0
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
                        
                        if table_exists:
                            # Cek apakah sudah ada entri untuk upload_id ini
                            if upload_id:
                                cursor.execute("SELECT id FROM model_visualizations WHERE upload_file_id = %s", (upload_id,))
                                existing = cursor.fetchone()
                                
                                if existing:
                                    # Update visualisasi yang sudah ada
                                    cursor.execute("""
                                        UPDATE model_visualizations 
                                        SET knn_cm_img = %s, dt_cm_img = %s, performance_comparison_img = %s, accuracy_img = %s 
                                        WHERE upload_file_id = %s
                                    """, (
                                        knn_cm_img,
                                        dt_cm_img,
                                        performance_comparison_img,
                                        accuracy_img,
                                        upload_id
                                    ))
                                else:
                                    # Simpan visualisasi baru
                                    cursor.execute("""
                                        INSERT INTO model_visualizations 
                                        (upload_file_id, knn_cm_img, dt_cm_img, performance_comparison_img, accuracy_img)
                                        VALUES (%s, %s, %s, %s, %s)
                                    """, (
                                        upload_id,
                                        knn_cm_img,
                                        dt_cm_img,
                                        performance_comparison_img,
                                        accuracy_img
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
                'knn_cm_img': knn_cm_img,
                'dt_cm_img': dt_cm_img,
                'accuracy_img': accuracy_img,
                'performance_comparison_img': performance_comparison_img,
                'results_table': results_table,
                'categories': unique_labels,
                'knn_detailed': knn_detailed,
                'dt_detailed': dt_detailed,
                'upload_id': upload_id  # Tambahkan upload_id untuk referensi
            }))
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
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
        upload_id = data.get('upload_id')  # Tambahkan parameter upload_id
        
        # Cek apakah model sudah dilatih
        if trained_knn is None or trained_dt is None:
            return jsonify({'error': 'Models have not been trained yet. Please process data first.'}), 400
        
        # Generate embedding untuk judul baru
        embedding = get_embedding(title)
        
        # Tambahkan fitur domain
        enhanced_embedding = create_enhanced_features(embedding, title)
        
        # Terapkan PCA jika tersedia
        if pca_model is not None:
            reduced_embedding = pca_model.transform([enhanced_embedding])[0]
        else:
            reduced_embedding = enhanced_embedding
        
        # Terapkan feature selection jika tersedia
        if feature_selector is not None:
            selected_embedding = feature_selector.transform([reduced_embedding])[0]
        else:
            selected_embedding = reduced_embedding
        
        # Prediksi dengan KNN
        knn_pred = trained_knn.predict([selected_embedding])[0]
        
        # Prediksi dengan Decision Tree
        dt_pred = trained_dt.predict([selected_embedding])[0]
        
        # Hitung keyakinan prediksi (untuk KNN)
        distances, indices = trained_knn.kneighbors([selected_embedding])
        nearest_titles = [f"Jarak: {distances[0][i]:.4f}" for i in range(len(indices[0]))]
        
        # Confidence score berdasarkan jarak
        confidence = float(1.0 - distances[0][0])
        
        # Simpan cache embedding ke disk
        save_embedding_cache()
        
        # Simpan prediksi ke database dengan upload_id jika ada
        try:
            # Prediksi single title biasanya tidak punya actual_category
            # Gunakan knn_pred sebagai actual_category untuk konsistensi
            prediction_id = save_prediction_to_db(
                title,
                knn_pred,  # Gunakan knn_pred sebagai actual_category
                knn_pred,
                dt_pred,
                confidence,
                upload_id  # Tambahkan upload_id
            )
        except Exception as e:
            print(f"Error saving prediction to database: {str(e)}")
            prediction_id = None
        
        # Kirim hasil ke frontend dengan konversi nilai numpy
        result = convert_numpy_types({
            'title': title,
            'knn_prediction': knn_pred,
            'dt_prediction': dt_pred,
            'nearest_neighbors': nearest_titles,
            'knn_confidence': confidence,
            'prediction_id': prediction_id
        })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mencari judul serupa
@app.route('/similar', methods=['POST'])
def find_similar_titles():
    data = request.json
    
    if 'title' not in data:
        return jsonify({'error': 'No title provided'}), 400
    
    try:
        title = data['title']
        limit = data.get('limit', 5)  # Default 5 hasil
        upload_id = data.get('upload_id')  # Tambahkan parameter upload_id
        
        # Generate embedding untuk judul
        embedding = get_embedding(title)
        
        # Tambahkan fitur domain jika model dilatih dengan fitur tersebut
        if pca_model is not None or feature_selector is not None:
            enhanced_embedding = create_enhanced_features(embedding, title)
            
            # Terapkan PCA jika tersedia
            if pca_model is not None:
                embedding = pca_model.transform([enhanced_embedding])[0]
            else:
                embedding = enhanced_embedding
            
            # Terapkan feature selection jika tersedia
            if feature_selector is not None:
                embedding = feature_selector.transform([embedding])[0]
        
        # Ambil semua judul dari database
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT t.id, t.title, c.name as category
                    FROM thesis_titles t
                    JOIN categories c ON t.category_id = c.id
                """)
                all_titles = cursor.fetchall()
            conn.close()
        
            # Hitung similarity dengan cosine similarity
            similarities = []
            for db_title in all_titles:
                # Dapatkan embedding dari judul di database
                db_embedding = get_embedding(db_title['title'])
                
                # Proses embedding database dengan cara yang sama
                if pca_model is not None or feature_selector is not None:
                    enhanced_db_embedding = create_enhanced_features(db_embedding, db_title['title'])
                    
                    if pca_model is not None:
                        db_embedding = pca_model.transform([enhanced_db_embedding])[0]
                    else:
                        db_embedding = enhanced_db_embedding
                    
                    if feature_selector is not None:
                        db_embedding = feature_selector.transform([db_embedding])[0]
                
                # Hitung cosine similarity
                similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
                
                similarities.append({
                    'id': db_title['id'],
                    'title': db_title['title'],
                    'category': db_title['category'],
                    'similarity': float(similarity)
                })
            
            # Urutkan berdasarkan similarity (terdekat dulu)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Prediksi kategori berdasarkan KNN
            if trained_knn is not None:
                predicted_category = trained_knn.predict([embedding])[0]
            else:
                # Jika model belum ditraining, ambil kategori dari judul terdekat
                predicted_category = similarities[0]['category'] if similarities else None
            
            # Batasi jumlah hasil
            similar_titles = similarities[:limit]
            
            # Simpan pencarian ke database jika fitur search_history ada
            try:
                conn = get_db_connection()
                if conn:
                    with conn.cursor() as cursor:
                        # Cek apakah tabel search_history ada
                        cursor.execute("SHOW TABLES LIKE 'search_history'")
                        table_exists = cursor.fetchone()
                        
                        if table_exists:
                            # Simpan pencarian
                            cursor.execute("""
                                INSERT INTO search_history (query, predicted_category, upload_file_id)
                                VALUES (%s, %s, %s)
                            """, (title, predicted_category, upload_id))
                            conn.commit()
                    conn.close()
            except Exception as e:
                print(f"Error saving search history: {str(e)}")
            
            # Kirim hasil ke frontend dengan konversi nilai numpy
            return jsonify(convert_numpy_types({
                'query': title,
                'similar_titles': similar_titles,
                'predicted_category': predicted_category
            }))
        else:
            return jsonify({'error': 'Database connection failed'}), 500
        
    except Exception as e:
        print(f"Error finding similar titles: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk download template Excel
@app.route('/template', methods=['GET'])
def get_template():
    try:
        # Buat file Excel template
        df = pd.DataFrame({
            'Judul Skripsi': ['Contoh Judul Skripsi 1', 'Contoh Judul Skripsi 2'],
            'Kategori': ['RPL', 'Jaringan']
        })
        
        # Simpan ke buffer
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        
        # Encode sebagai base64
        excel_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'data': excel_data,
            'filename': 'template_klasifikasi_skripsi.xlsx'
        })
    except Exception as e:
        print(f"Error creating template: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint untuk menghapus prediksi berdasarkan ID
@app.route('/delete_prediction/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                # Periksa apakah prediksi ada
                cursor.execute("SELECT id FROM predictions WHERE id = %s", (prediction_id,))
                prediction = cursor.fetchone()
                
                if not prediction:
                    return jsonify({'success': False, 'error': 'Prediction not found'}), 404
                
                # Hapus prediksi
                cursor.execute("DELETE FROM predictions WHERE id = %s", (prediction_id,))
                conn.commit()
                
                return jsonify({'success': True, 'message': 'Prediction deleted successfully'})
            conn.close()
        else:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
    except Exception as e:
        print(f"Error deleting prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Endpoint untuk mengambil semua prediksi (untuk halaman history)
@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT p.id, p.title, p.actual_category_id, p.knn_prediction_id, p.dt_prediction_id, 
                           p.confidence, p.prediction_date, p.upload_file_id,
                           c1.name as actual_category, c2.name as knn_prediction, c3.name as dt_prediction 
                    FROM predictions p
                    LEFT JOIN categories c1 ON p.actual_category_id = c1.id
                    LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
                    LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
                    ORDER BY p.prediction_date DESC
                """)
                predictions = cursor.fetchall()
                
                # Konversi datetime ke string
                for pred in predictions:
                    if 'prediction_date' in pred and pred['prediction_date']:
                        pred['prediction_date'] = pred['prediction_date'].strftime('%Y-%m-%d %H:%M:%S')
                
                return jsonify({'success': True, 'predictions': predictions})
            conn.close()
        else:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
    except Exception as e:
        print(f"Error getting predictions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Endpoint untuk mendapatkan detail prediksi berdasarkan ID
@app.route('/get_prediction/<int:prediction_id>', methods=['GET'])
def get_prediction_detail(prediction_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT p.id, p.title, p.actual_category_id, p.knn_prediction_id, p.dt_prediction_id, 
                           p.confidence, p.prediction_date, p.upload_file_id,
                           c1.name as actual_category, c2.name as knn_prediction, c3.name as dt_prediction 
                    FROM predictions p
                    LEFT JOIN categories c1 ON p.actual_category_id = c1.id
                    LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
                    LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
                    WHERE p.id = %s
                """, (prediction_id,))
                prediction = cursor.fetchone()
                
                if not prediction:
                    return jsonify({'success': False, 'error': 'Prediction not found'}), 404
                
                # Konversi datetime ke string
                if 'prediction_date' in prediction and prediction['prediction_date']:
                    prediction['prediction_date'] = prediction['prediction_date'].strftime('%Y-%m-%d %H:%M:%S')
                
                return jsonify({'success': True, 'prediction': prediction})
            conn.close()
        else:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
    except Exception as e:
        print(f"Error getting prediction detail: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Endpoint baru untuk mendapatkan prediksi berdasarkan upload_id
@app.route('/get_predictions_by_upload/<int:upload_id>', methods=['GET'])
def get_predictions_by_upload(upload_id):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT p.id, p.title, p.actual_category_id, p.knn_prediction_id, p.dt_prediction_id, 
                           p.confidence, p.prediction_date, 
                           c1.name as actual_category, c2.name as knn_prediction, c3.name as dt_prediction 
                    FROM predictions p
                    LEFT JOIN categories c1 ON p.actual_category_id = c1.id
                    LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
                    LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
                    WHERE p.upload_file_id = %s
                    ORDER BY p.prediction_date DESC
                """, (upload_id,))
                predictions = cursor.fetchall()
                
                # Konversi datetime ke string
                for pred in predictions:
                    if 'prediction_date' in pred and pred['prediction_date']:
                        pred['prediction_date'] = pred['prediction_date'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Dapatkan informasi file yang diupload
                cursor.execute("""
                    SELECT id, original_filename, file_size, upload_date, processed
                    FROM uploaded_files
                    WHERE id = %s
                """, (upload_id,))
                file_info = cursor.fetchone()
                
                if file_info and 'upload_date' in file_info and file_info['upload_date']:
                    file_info['upload_date'] = file_info['upload_date'].strftime('%Y-%m-%d %H:%M:%S')
                
                return jsonify({
                    'success': True, 
                    'predictions': predictions,
                    'file_info': file_info
                })
            conn.close()
        else:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
    except Exception as e:
        print(f"Error getting predictions by upload: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Endpoint untuk mendapatkan daftar file yang telah diupload
@app.route('/get_uploaded_files', methods=['GET'])
def get_uploaded_files():
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT uf.id, uf.original_filename, uf.file_size, uf.upload_date, uf.processed,
                           COUNT(p.id) as prediction_count
                    FROM uploaded_files uf
                    LEFT JOIN predictions p ON uf.id = p.upload_file_id
                    GROUP BY uf.id
                    ORDER BY uf.upload_date DESC
                """)
                files = cursor.fetchall()
                
                # Konversi datetime ke string
                for file in files:
                    if 'upload_date' in file and file['upload_date']:
                        file['upload_date'] = file['upload_date'].strftime('%Y-%m-%d %H:%M:%S')
                
                return jsonify({'success': True, 'files': files})
            conn.close()
        else:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
    except Exception as e:
        print(f"Error getting uploaded files: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Endpoint untuk mengecek status model dan akurasi terkini
@app.route('/model_status', methods=['GET'])
def get_model_status():
    try:
        # Cek apakah model sudah dilatih
        if trained_knn is None or trained_dt is None:
            return jsonify({
                'success': True,
                'models_trained': False,
                'message': 'Models have not been trained yet.'
            })
        
        # Ambil performa model terakhir dari database
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT model_name, accuracy, parameters, created_at
                    FROM model_performances
                    ORDER BY created_at DESC
                    LIMIT 2
                """)
                performances = cursor.fetchall()
                
                # Konversi datetime ke string
                for perf in performances:
                    if 'created_at' in perf and perf['created_at']:
                        perf['created_at'] = perf['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Buat statistik model
                model_stats = {
                    'models_trained': True,
                    'knn_params': None,
                    'dt_params': None,
                    'knn_accuracy': None,
                    'dt_accuracy': None,
                    'last_trained': None,
                    'embedding_cache_size': len(embedding_cache),
                    'train_data_size': len(train_features) if train_features else 0
                }
                
                # Tambahkan informasi performa
                for perf in performances:
                    if 'KNN' in perf['model_name']:
                        model_stats['knn_params'] = perf['parameters']
                        model_stats['knn_accuracy'] = perf['accuracy']
                        model_stats['last_trained'] = perf['created_at']
                    elif 'Tree' in perf['model_name'] or 'DT' in perf['model_name']:
                        model_stats['dt_params'] = perf['parameters']
                        model_stats['dt_accuracy'] = perf['accuracy']
                        if not model_stats['last_trained']:
                            model_stats['last_trained'] = perf['created_at']
            
            # Tambahkan info kategori
            with conn.cursor() as cursor:
                cursor.execute("SELECT name FROM categories")
                categories = cursor.fetchall()
                model_stats['available_categories'] = [cat['name'] for cat in categories]
            
            conn.close()
            return jsonify({'success': True, **model_stats})
        else:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
    
    except Exception as e:
        print(f"Error checking model status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Endpoint baru untuk model tuning manual (untuk percobaan parameter)
@app.route('/tune_model', methods=['POST'])
def tune_model():
    data = request.json
    
    if not data:
        return jsonify({'error': 'No parameters provided'}), 400
    
    try:
        # Cek apakah data training tersedia
        if not train_features or not train_labels:
            return jsonify({'error': 'No training data available. Please upload and process data first.'}), 400
        
        # Parameter untuk tuning
        knn_params = data.get('knn', {})
        dt_params = data.get('dt', {})
        upload_id = data.get('upload_id')  # Ambil upload_id jika ada
        
        # Default values jika tidak ada
        n_neighbors = knn_params.get('n_neighbors', 3)
        weights = knn_params.get('weights', 'uniform')
        max_depth = dt_params.get('max_depth', None)
        criterion = dt_params.get('criterion', 'gini')
        min_samples_split = dt_params.get('min_samples_split', 2)
        use_bagging = dt_params.get('use_bagging', True)
        n_estimators = dt_params.get('n_estimators', 10)
        
        # Bagi data menjadi train dan test (gunakan yang sudah ada)
        X_train, X_test, y_train, y_test = train_test_split(
            train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # Train KNN dengan parameter yang diberikan
        print(f"Training KNN with n_neighbors={n_neighbors}, weights={weights}")
        knn = KNNDetailedModel(n_neighbors=n_neighbors, weights=weights)
        knn.fit(X_train, y_train)
        knn_metrics = knn.evaluate(X_test, y_test)
        knn_acc = knn_metrics['accuracy']
        
        # Train Decision Tree dengan parameter yang diberikan
        print(f"Training DT with max_depth={max_depth}, criterion={criterion}, min_samples_split={min_samples_split}")
        if use_bagging:
            dt = BaggedDTDetailedModel(
                max_depth=max_depth,
                criterion=criterion,
                min_samples_split=min_samples_split,
                random_state=42,
                n_estimators=n_estimators
            )
        else:
            dt = DTDetailedModel(
                max_depth=max_depth,
                criterion=criterion,
                min_samples_split=min_samples_split,
                random_state=42
            )
        
        dt.fit(X_train, y_train)
        dt_metrics = dt.evaluate(X_test, y_test)
        dt_acc = dt_metrics['accuracy']
        
        # Simpan performa ke database dengan upload_id
        save_model_performance('KNN (Manual Tuning)', knn_acc, {'n_neighbors': n_neighbors, 'weights': weights}, upload_id)
        
        dt_perf_params = {
            'max_depth': max_depth, 
            'criterion': criterion,
            'min_samples_split': min_samples_split
        }
        if use_bagging:
            dt_perf_params['use_bagging'] = True
            dt_perf_params['n_estimators'] = n_estimators
            
        save_model_performance('Decision Tree (Manual Tuning)', dt_acc, dt_perf_params, upload_id)
        
        # Simpan model jika akurasi lebih baik dari yang ada
        global trained_knn, trained_dt
        
        if trained_knn is None or knn_acc > 0.8:  # Hanya simpan jika akurasi >80% atau tidak ada model sebelumnya
            trained_knn = knn.model
            trained_dt = dt.model
            
            # Simpan model ke disk
            models_data = {
                'knn': knn.model,
                'dt': dt.model,
                'train_features': train_features,
                'train_labels': train_labels,
                'pca_model': pca_model,
                'feature_selector': feature_selector,
                'preprocessing_params': {
                    'max_length': MAX_LENGTH,
                    'embedding_method': 'combined_pooling'
                }
            }
            save_complete_models(models_data)
        
        # Kirim hasil
        return jsonify({
            'success': True,
            'knn_accuracy': knn_acc,
            'dt_accuracy': dt_acc,
            'knn_parameters': {'n_neighbors': n_neighbors, 'weights': weights},
            'dt_parameters': dt_perf_params,
            'models_saved': knn_acc > 0.8 or trained_knn is None,
            'upload_id': upload_id
        })
        
    except Exception as e:
        print(f"Error tuning model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Server starting on http://localhost:5000")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Model folder: {MODEL_FOLDER}")
    print(f"Cache folder: {CACHE_FOLDER}")
    # Jalankan server dalam mode non-threaded untuk menghindari masalah dengan matplotlib
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)