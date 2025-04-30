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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel
import pymysql
import hashlib
from datetime import datetime

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

# Simpan hasil prediksi ke database
def save_prediction_to_db(title, actual_category, knn_pred, dt_pred, confidence=0):
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
                    INSERT INTO predictions (title, actual_category_id, knn_prediction_id, dt_prediction_id, confidence)
                    VALUES (%s, %s, %s, %s, %s)
                """, (title, actual_id, knn_id, dt_id, confidence))
                
                conn.commit()
                return cursor.lastrowid
            conn.close()
    except Exception as e:
        print(f"Error saving prediction to database: {str(e)}")
        return None

# Simpan performa model ke database
def save_model_performance(model_name, accuracy, parameters=None):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO model_performances (model_name, accuracy, parameters)
                    VALUES (%s, %s, %s)
                """, (model_name, accuracy, json.dumps(parameters) if parameters else None))
                
                conn.commit()
                return cursor.lastrowid
            conn.close()
    except Exception as e:
        print(f"Error saving model performance: {str(e)}")
        return None

# Analisis kata kunci dan simpan ke database
def analyze_keywords(category, titles):
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
                    # Hapus analisis sebelumnya
                    cursor.execute("DELETE FROM keyword_analysis WHERE category_id = %s", (category_id,))
                    
                    # Simpan keyword baru
                    for keyword, frequency in top_keywords:
                        cursor.execute("""
                            INSERT INTO keyword_analysis (category_id, keyword, frequency)
                            VALUES (%s, %s, %s)
                        """, (category_id, keyword, frequency))
                    
                    conn.commit()
                    return True
            conn.close()
    except Exception as e:
        print(f"Error analyzing keywords: {str(e)}")
        return False

# Konstanta untuk model
MODEL_FILENAME = os.path.join(MODEL_FOLDER, 'models.pkl')
EMBEDDING_CACHE_FILENAME = os.path.join(CACHE_FOLDER, 'embedding_cache.json')
MAX_LENGTH = 128  # Maximum sequence length for IndoBERT

# Inisialisasi model IndoBERT
print("Loading IndoBERT model... (bisa memakan waktu beberapa menit)")
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
print("IndoBERT model loaded successfully!")

# Variabel global untuk menyimpan model yang sudah dilatih
trained_knn = None
trained_dt = None
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

# Preprocessing teks
def preprocess_text(text):
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).lower()
    # Hapus karakter khusus dan angka
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi untuk mendapatkan embedding dari teks dengan caching
def get_embedding(text):
    preprocessed_text = preprocess_text(text)
    
    # Cek apakah embedding sudah ada di cache
    if preprocessed_text in embedding_cache:
        return np.array(embedding_cache[preprocessed_text])
    
    # Jika tidak ada di cache, hitung embedding baru
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    # Simpan ke cache
    embedding_cache[preprocessed_text] = cls_embedding.tolist()
    
    # Simpan cache ke disk secara periodik (setiap 10 embedding baru)
    if len(embedding_cache) % 10 == 0:
        save_embedding_cache()
    
    return cls_embedding

# Simpan cache embedding ke disk
def save_embedding_cache():
    try:
        with open(EMBEDDING_CACHE_FILENAME, 'w') as f:
            json.dump(embedding_cache, f)
        print(f"Saved {len(embedding_cache)} embeddings to cache")
    except Exception as e:
        print(f"Error saving embedding cache: {str(e)}")

# Simpan model ke disk
def save_models(knn, dt, X_train, y_train):
    try:
        models_data = {
            'knn': knn,
            'dt': dt,
            'train_features': X_train,
            'train_labels': y_train
        }
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(models_data, f)
        print("Models saved to disk successfully")
    except Exception as e:
        print(f"Error saving models: {str(e)}")

# Class untuk model KNN dengan detil performa
class KNNDetailedModel:
    def __init__(self, n_neighbors=3):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.n_neighbors = n_neighbors
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
    def __init__(self, max_depth=None, criterion='gini', random_state=42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, 
            criterion=criterion,
            random_state=random_state
        )
        self.max_depth = max_depth
        self.criterion = criterion
        self.metrics = {}
        self.confusion_matrix = None
        self.classification_report = None
        self.class_metrics = {}
        self.feature_importances = None
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train
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
        self.metrics['tree_depth'] = self.model.get_depth()
        self.metrics['n_leaves'] = self.model.get_n_leaves()
        
        return self.metrics

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
            
            # Generate embedding untuk semua judul
            print("Generating embeddings... (bisa memakan waktu)")
            embeddings = []
            titles = []
            labels = []
            
            for i, row in df.iterrows():
                title = row[title_column]
                if pd.notna(title) and str(title).strip():  # Skip NaN atau string kosong
                    try:
                        embedding = get_embedding(str(title))
                        embeddings.append(embedding)
                        titles.append(str(title))
                        labels.append(str(row[label_column]))
                    except Exception as e:
                        print(f"Error in processing title: {title}. Error: {str(e)}")
            
            # Jika tidak ada data yang valid
            if not embeddings:
                return jsonify({'error': 'No valid title data found in the uploaded file'}), 400
            
            print(f"Total data: {len(embeddings)} judul skripsi")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, test_size=0.2, random_state=42
            )
            
            print("Training KNN model...")
            # Train KNN with detailed metrics
            knn = KNNDetailedModel(n_neighbors=3)
            knn.fit(X_train, y_train)
            knn_metrics = knn.evaluate(X_test, y_test)
            knn_pred = knn.predict(X_test)
            knn_acc = knn_metrics['accuracy']
            
            print("Training Decision Tree model...")
            # Train Decision Tree with detailed metrics
            dt = DTDetailedModel(max_depth=None, criterion='gini', random_state=42)
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
            
            # Simpan model ke disk
            save_models(knn.model, dt.model, X_train, y_train)
            
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
            test_titles = [titles[i] for i in range(len(titles)) if i >= len(titles) - len(y_test)]
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
                'tree_depth': dt.metrics.get('tree_depth', 0),
                'n_leaves': dt.metrics.get('n_leaves', 0)
            }
            
            # Setelah proses klasifikasi selesai, simpan data ke database
            try:
                conn = get_db_connection()
                if conn:
                    with conn.cursor() as cursor:
                        # Simpan file yang diupload
                        file_size = os.path.getsize(file_path)
                        
                        cursor.execute("""
                            INSERT INTO uploaded_files (filename, original_filename, file_size, processed)
                            VALUES (%s, %s, %s, %s)
                        """, (file_path, file.filename, file_size, True))
                        
                        upload_id = cursor.lastrowid
                        
                        # Simpan performa model
                        save_model_performance('KNN', knn_acc, {'n_neighbors': 3})
                        save_model_performance('Decision Tree', dt_acc, {'max_depth': dt.model.get_depth(), 'criterion': dt.criterion})
                        
                        # Simpan data judul dan hasil klasifikasi
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
                                    # Simpan judul baru
                                    cursor.execute("""
                                        INSERT INTO thesis_titles (title, category_id)
                                        VALUES (%s, %s)
                                    """, (title, category_id))
                        
                        # Simpan hasil prediksi test set
                        for i, result in enumerate(results_table):
                            save_prediction_to_db(
                                result['full_title'], 
                                result['actual'], 
                                result['knn_pred'], 
                                result['dt_pred']
                            )
                        
                        conn.commit()
                        
                        # Analisis kata kunci per kategori
                        for category in unique_labels:
                            category_titles = [titles[i] for i in range(len(titles)) if labels[i] == category]
                            analyze_keywords(category, category_titles)
                    conn.close()
            except Exception as e:
                print(f"Error saving data to database: {str(e)}")
            
            print("Processing completed successfully!")
            
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
                'dt_detailed': dt_detailed
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
        
        # Cek apakah model sudah dilatih
        if trained_knn is None or trained_dt is None:
            return jsonify({'error': 'Models have not been trained yet. Please process data first.'}), 400
        
        # Generate embedding untuk judul baru
        embedding = get_embedding(title)
        
        # Prediksi dengan KNN
        knn_pred = trained_knn.predict([embedding])[0]
        
        # Prediksi dengan Decision Tree
        dt_pred = trained_dt.predict([embedding])[0]
        
        # Hitung keyakinan prediksi (untuk KNN)
        distances, indices = trained_knn.kneighbors([embedding])
        nearest_titles = [f"Jarak: {distances[0][i]:.4f}" for i in range(len(indices[0]))]
        
        # Simpan cache embedding ke disk
        save_embedding_cache()
        
        # Simpan prediksi ke database
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cursor:
                    # Ambil kategori yang paling banyak diprediksi sebagai kategori sebenarnya
                    actual_category = None
                    if knn_pred == dt_pred:
                        actual_category = knn_pred
                    else:
                        actual_category = knn_pred  # Default ke KNN jika berbeda
                    
                    # Simpan prediksi
                    prediction_id = save_prediction_to_db(
                        title,
                        actual_category,
                        knn_pred,
                        dt_pred,
                        1.0 - distances[0][0]  # Confidence berdasarkan jarak
                    )
                conn.close()
        except Exception as e:
            print(f"Error saving prediction to database: {str(e)}")
        
        # Kirim hasil ke frontend dengan konversi nilai numpy
        result = convert_numpy_types({
            'title': title,
            'knn_prediction': knn_pred,
            'dt_prediction': dt_pred,
            'nearest_neighbors': nearest_titles,
            'knn_confidence': float(1.0 - distances[0][0])  # Confidence sebagai 1 - jarak
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
        
        # Generate embedding untuk judul
        embedding = get_embedding(title)
        
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

if __name__ == '__main__':
    print(f"Server starting on http://localhost:5000")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Model folder: {MODEL_FOLDER}")
    print(f"Cache folder: {CACHE_FOLDER}")
    # Jalankan server dalam mode non-threaded untuk menghindari masalah dengan matplotlib
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)