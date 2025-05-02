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
import pymysql
import hashlib
from datetime import datetime
import gc  # Garbage collector untuk mengontrol memori

app = Flask(__name__)
CORS(app)  # Memungkinkan request dari domain lain (PHP frontend)

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

# Konstanta untuk model
MODEL_FILENAME = os.path.join(MODEL_FOLDER, 'models.pkl')
EMBEDDING_CACHE_FILENAME = os.path.join(CACHE_FOLDER, 'embedding_cache.json')
MAX_LENGTH = 128  # Maximum sequence length untuk transformer model

# Variabel global
tokenizer = None
model = None
embedding_cache = {}
trained_knn = None
trained_dt = None

# Fungsi koneksi database
def get_db_connection():
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

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
                            (title_id, knn_prediction_id, dt_prediction_id, confidence)
                            VALUES (%s, %s, %s, %s)
                        """, (title_id, knn_id, dt_id, confidence))
                        
                        conn.commit()
                
                return prediction_id
            conn.close()
    except Exception as e:
        print(f"Error saving prediction to database: {str(e)}")
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

def load_transformer_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            from transformers import AutoTokenizer, AutoModel
            print("Loading IndoBERT model... (bisa memakan waktu beberapa menit)")
            tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1", model_max_length=MAX_LENGTH)
            model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
            print("IndoBERT model loaded successfully!")
        except Exception as e:
            print(f"Error loading transformer model: {str(e)}")
            return False
    return True

# Load embedding cache jika ada
def load_embedding_cache():
    global embedding_cache
    if not embedding_cache and os.path.exists(EMBEDDING_CACHE_FILENAME):
        try:
            with open(EMBEDDING_CACHE_FILENAME, 'r') as f:
                embedding_cache = json.load(f)
            print(f"Loaded {len(embedding_cache)} cached embeddings")
        except Exception as e:
            print(f"Error loading embedding cache: {str(e)}")
            embedding_cache = {}

# Load model jika ada
def load_trained_models():
    global trained_knn, trained_dt
    if (trained_knn is None or trained_dt is None) and os.path.exists(MODEL_FILENAME):
        try:
            with open(MODEL_FILENAME, 'rb') as f:
                models_data = pickle.load(f)
                trained_knn = models_data.get('knn')
                trained_dt = models_data.get('dt')
            print("Models loaded successfully from disk")
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            trained_knn = None
            trained_dt = None
            return False
    elif trained_knn is not None and trained_dt is not None:
        return True
    else:
        print("No models found on disk")
        return False

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
    global embedding_cache
    
    # Pastikan model transformer sudah dimuat
    if not load_transformer_model():
        raise Exception("Failed to load transformer model")
    
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
    
    # Panggil garbage collector untuk membebaskan memori
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return cls_embedding

# Simpan cache embedding ke disk
def save_embedding_cache():
    try:
        # Batasi ukuran cache untuk menghemat memori
        if len(embedding_cache) > 1000:  # Batasi 1000 embedding
            # Ambil 800 item terbaru, buang yang lama
            keys = list(embedding_cache.keys())
            for key in keys[:-800]:
                del embedding_cache[key]
        
        with open(EMBEDDING_CACHE_FILENAME, 'w') as f:
            json.dump(embedding_cache, f)
        print(f"Saved {len(embedding_cache)} embeddings to cache")
    except Exception as e:
        print(f"Error saving embedding cache: {str(e)}")

# OPTIMASI: Fungsi batch processing untuk file besar
def process_excel_in_batches(file_path, batch_size=100):
    # Membaca file secara efisien
    chunks = pd.read_excel(file_path, chunksize=batch_size)
    all_data = []
    
    for chunk in chunks:
        # Proses per batch
        all_data.append(chunk)
        # Force memory cleanup
        gc.collect()
    
    # Gabungkan hasil
    return pd.concat(all_data, ignore_index=True)

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
            # Verifikasi model telah dimuat
            if not load_trained_models():
                return jsonify({'error': 'Required models are not available. Please ensure models have been trained first.'}), 400
                
            # Load model transformer ketika diperlukan
            if not load_transformer_model():
                return jsonify({'error': 'Failed to load transformer model'}), 500
                
            # Load cache embedding
            load_embedding_cache()
            
            # Baca file Excel dalam batch untuk menghemat memori
            print("Reading Excel file...")
            if os.path.getsize(file_path) > 5 * 1024 * 1024:  # Jika file > 5MB
                df = process_excel_in_batches(file_path)
            else:
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
                print("Tidak ada kolom label ditemukan, menggunakan prediksi model untuk pelabelan")
                
                # Buat kolom label kosong yang akan diisi dengan prediksi
                df['label'] = None
                label_column = 'label'
            
            print(f"Menggunakan kolom '{label_column}' sebagai label kategori")
            
            # Struktur untuk menyimpan hasil prediksi
            results_table = []
            
            # OPTIMASI: Proses batch untuk analisis
            batch_size = 20
            current_batch = []
            
            print("Predicting categories...")
            
            for i, row in df.iterrows():
                title = row[title_column]
                if pd.notna(title) and str(title).strip():  # Skip NaN atau string kosong
                    current_batch.append({
                        'title': str(title),
                        'original_label': str(row[label_column]) if pd.notna(row[label_column]) else None
                    })
                    
                    # Proses batch ketika mencapai batch_size
                    if len(current_batch) >= batch_size or i == len(df) - 1:
                        for item in current_batch:
                            try:
                                # Ambil embedding judul
                                embedding = get_embedding(item['title'])
                                
                                # Prediksi dengan model yang dimuat
                                knn_pred = trained_knn.predict([embedding])[0] if trained_knn else "Unknown"
                                dt_pred = trained_dt.predict([embedding])[0] if trained_dt else "Unknown"
                                
                                # Gunakan prediksi KNN sebagai label jika tidak ada label asli
                                actual_label = item['original_label'] if item['original_label'] else knn_pred
                                
                                # Hitung confidence (untuk KNN)
                                if trained_knn:
                                    distances, indices = trained_knn.kneighbors([embedding])
                                    confidence = float(1.0 - distances[0][0])  # Confidence sebagai 1 - jarak
                                else:
                                    confidence = 0.5  # Default confidence
                                
                                # Simpan ke hasil
                                results_table.append({
                                    'title': item['title'][:100] + '...' if len(item['title']) > 100 else item['title'],
                                    'full_title': item['title'],
                                    'actual': actual_label,
                                    'knn_pred': knn_pred,
                                    'dt_pred': dt_pred,
                                    'confidence': confidence
                                })
                                
                                # Simpan ke database
                                save_prediction_to_db(
                                    item['title'],
                                    actual_label,
                                    knn_pred,
                                    dt_pred,
                                    confidence,
                                    upload_id
                                )
                                
                            except Exception as e:
                                print(f"Error processing title: {item['title']}. Error: {str(e)}")
                        
                        # Reset batch
                        current_batch = []
                        
                        # Force memory cleanup
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Simpan cache embedding ke disk
            save_embedding_cache()
            
            # Ekstrak unique labels untuk statistik
            unique_labels = sorted(list(set([result['actual'] for result in results_table])))
            
            # Hitung statistik sederhana
            category_counts = {}
            for category in unique_labels:
                category_counts[category] = sum(1 for result in results_table if result['actual'] == category)
            
            # Analisis kata kunci per kategori
            for category in unique_labels:
                category_titles = [result['full_title'] for result in results_table if result['actual'] == category]
                analyze_keywords(category, category_titles)
            
            # Update status file upload menjadi processed
            try:
                conn = get_db_connection()
                if conn:
                    with conn.cursor() as cursor:
                        if upload_id:
                            cursor.execute("UPDATE uploaded_files SET processed = 1 WHERE id = %s", (upload_id,))
                            conn.commit()
                    conn.close()
            except Exception as e:
                print(f"Error updating upload status: {str(e)}")
            
            print("Processing completed successfully!")
            
            # Bersihkan objek besar yang tidak diperlukan
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Generate simple pie chart for categories
            plt.figure(figsize=(8, 6), dpi=60)
            labels = list(category_counts.keys())
            sizes = list(category_counts.values())
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Distribution of Categories')
            
            # Simpan plot ke buffer
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=60)
            buf.seek(0)
            
            # Convert plot ke base64
            category_dist_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            buf.close()
            
            # Kirim hasil ke frontend
            return jsonify(convert_numpy_types({
                'results_table': results_table,
                'categories': unique_labels,
                'category_counts': category_counts,
                'category_distribution': category_dist_img,
                'upload_id': upload_id
            }))
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            # Clear memory on error
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
        upload_id = data.get('upload_id')
        
        # Verifikasi model telah dimuat
        if not load_trained_models():
            return jsonify({'error': 'Required models are not available. Please ensure models have been trained first.'}), 400
        
        # Load cache embedding
        load_embedding_cache()
        
        # Load model transformer jika diperlukan
        if not load_transformer_model():
            return jsonify({'error': 'Failed to load transformer model'}), 500
        
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
        
        # Simpan prediksi ke database dengan upload_id jika ada
        try:
            # Prediksi single title biasanya tidak punya actual_category
            # Gunakan knn_pred sebagai actual_category untuk konsistensi
            prediction_id = save_prediction_to_db(
                title,
                knn_pred,  # Gunakan knn_pred sebagai actual_category
                knn_pred,
                dt_pred,
                1.0 - distances[0][0],  # Confidence berdasarkan jarak
                upload_id  # Tambahkan upload_id
            )
        except Exception as e:
            print(f"Error saving prediction to database: {str(e)}")
            prediction_id = None
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Kirim hasil ke frontend dengan konversi nilai numpy
        result = convert_numpy_types({
            'title': title,
            'knn_prediction': knn_pred,
            'dt_prediction': dt_pred,
            'nearest_neighbors': nearest_titles,
            'knn_confidence': float(1.0 - distances[0][0]),  # Confidence sebagai 1 - jarak
            'prediction_id': prediction_id
        })
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Clear memory on error
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
        
        # Verifikasi model telah dimuat
        if not load_trained_models():
            return jsonify({'error': 'Required models are not available. Please ensure models have been trained first.'}), 400
        
        # Load model transformer jika diperlukan
        if not load_transformer_model():
            return jsonify({'error': 'Failed to load transformer model'}), 500
            
        # Load cache embedding
        load_embedding_cache()
        
        # Generate embedding untuk judul
        embedding = get_embedding(title)
        
        # Ambil semua judul dari database
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                # OPTIMASI: Batasi jumlah judul yang diambil
                cursor.execute("""
                    SELECT t.id, t.title, c.name as category
                    FROM thesis_titles t
                    JOIN categories c ON t.category_id = c.id
                    ORDER BY t.id DESC
                    LIMIT 1000  -- Batasi hanya 1000 judul terbaru
                """)
                all_titles = cursor.fetchall()
            conn.close()
        
            # Hitung similarity dengan cosine similarity
            similarities = []
            
            # OPTIMASI: Proses batch untuk menghemat memori
            batch_size = 20
            current_batch = []
            
            for db_title in all_titles:
                current_batch.append(db_title)
                
                if len(current_batch) >= batch_size:
                    for title_item in current_batch:
                        # Dapatkan embedding dari judul di database
                        db_embedding = get_embedding(title_item['title'])
                        
                        # Hitung cosine similarity
                        similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
                        
                        similarities.append({
                            'id': title_item['id'],
                            'title': title_item['title'],
                            'category': title_item['category'],
                            'similarity': float(similarity)
                        })
                    
                    # Reset batch
                    current_batch = []
                    
                    # Force memory cleanup
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Proses sisa batch
            for title_item in current_batch:
                db_embedding = get_embedding(title_item['title'])
                similarity = np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
                
                similarities.append({
                    'id': title_item['id'],
                    'title': title_item['title'],
                    'category': title_item['category'],
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
            
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
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
        # Clear memory on error
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
        
        # Free memory
        buffer.close()
        gc.collect()
        
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
                # OPTIMASI: Batasi jumlah hasil untuk mengurangi beban
                cursor.execute("""
                    SELECT p.id, p.title, p.actual_category_id, p.knn_prediction_id, p.dt_prediction_id, 
                           p.confidence, p.prediction_date, p.upload_file_id,
                           c1.name as actual_category, c2.name as knn_prediction, c3.name as dt_prediction 
                    FROM predictions p
                    LEFT JOIN categories c1 ON p.actual_category_id = c1.id
                    LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
                    LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
                    ORDER BY p.prediction_date DESC
                    LIMIT 1000  -- Batasi hanya 1000 prediksi terbaru
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
                           p.confidence, p.prediction_date, 
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

# Endpoint untuk mendapatkan prediksi berdasarkan upload_id
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
                    LIMIT 500  -- Batasi hanya 500 file terbaru
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

# OPTIMASI: Fungsi untuk membersihkan memori secara periodik
@app.route('/cleanup_memory', methods=['GET'])
def cleanup_memory():
    try:
        # Simpan cache embedding
        save_embedding_cache()
        
        # Bersihkan model tensor
        global model, tokenizer
        if model is not None:
            del model
            model = None
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return jsonify({'success': True, 'message': 'Memory cleanup completed'})
    except Exception as e:
        print(f"Error during memory cleanup: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Fungsi untuk mengecek ketersediaan transformers
@app.route('/check_status', methods=['GET'])
def check_status():
    try:
        # Cek status file dan folder
        folders_ok = all(os.path.exists(folder) for folder in [UPLOAD_FOLDER, MODEL_FOLDER, CACHE_FOLDER])
        
        # Cek koneksi database
        db_ok = False
        conn = get_db_connection()
        if conn:
            db_ok = True
            conn.close()
        
        # Cek ketersediaan model
        models_available = os.path.exists(MODEL_FILENAME)
        
        # Cek ketersediaan transformers
        transformers_available = False
        try:
            from transformers import __version__ as transformers_version
            transformers_available = True
        except:
            transformers_version = "Not available"
        
        # Cek status memory
        import psutil
        memory = psutil.virtual_memory()
        memory_usage = {
            'total': memory.total / (1024 * 1024 * 1024),  # GB
            'available': memory.available / (1024 * 1024 * 1024),  # GB
            'percent': memory.percent
        }
        
        return jsonify({
            'success': True,
            'folders_ok': folders_ok,
            'db_connection': db_ok,
            'models_available': models_available,
            'transformers_available': transformers_available,
            'transformers_version': transformers_version if transformers_available else None,
            'memory': memory_usage,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Server starting on http://localhost:5000")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Model folder: {MODEL_FOLDER}")
    print(f"Cache folder: {CACHE_FOLDER}")
    
    # Cek model dan cache saat startup
    print("Checking for existing models and cache...")
    if os.path.exists(MODEL_FILENAME):
        print(f"Found existing models at {MODEL_FILENAME}")
    else:
        print("WARNING: No classification models found. Please ensure models have been trained before using the system.")
        
    if os.path.exists(EMBEDDING_CACHE_FILENAME):
        try:
            with open(EMBEDDING_CACHE_FILENAME, 'r') as f:
                cache_size = len(json.load(f))
                print(f"Found embedding cache with {cache_size} entries")
        except:
            print("Found embedding cache file but could not read it")
    else:
        print("No embedding cache found. A new cache will be created when processing data.")
    
    # Jalankan server dalam mode non-threaded untuk menghindari masalah dengan matplotlib
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=False)