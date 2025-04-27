from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
import numpy as np
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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModel

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
            # Train KNN
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            knn_pred = knn.predict(X_test)
            knn_acc = accuracy_score(y_test, knn_pred)
            
            print("Training Decision Tree model...")
            # Train Decision Tree
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X_train, y_train)
            dt_pred = dt.predict(X_test)
            dt_acc = accuracy_score(y_test, dt_pred)
            
            # Simpan model yang dilatih
            global trained_knn, trained_dt, train_features
            trained_knn = knn
            trained_dt = dt
            train_features = X_train
            
            # Simpan model ke disk
            save_models(knn, dt, X_train, y_train)
            
            # Simpan cache embedding ke disk
            save_embedding_cache()
            
            print("Generating visualizations...")
            # Dapatkan kategori unik
            unique_labels = sorted(list(set(labels)))
            
            # Confusion Matrix
            knn_cm = confusion_matrix(y_test, knn_pred, labels=unique_labels)
            dt_cm = confusion_matrix(y_test, dt_pred, labels=unique_labels)
            
            # Generate gambar confusion matrix
            knn_cm_img = generate_confusion_matrix_image(knn_cm, unique_labels, "Confusion Matrix - KNN", 'Blues')
            dt_cm_img = generate_confusion_matrix_image(dt_cm, unique_labels, "Confusion Matrix - Decision Tree", 'Greens')
            
            # Buat grafik perbandingan akurasi
            plt.figure(figsize=(8, 5))
            algorithms = ['KNN', 'Decision Tree']
            accuracies = [knn_acc, dt_acc]
            plt.bar(algorithms, accuracies, color=['blue', 'green'])
            plt.ylim(0, 1)
            plt.title('Perbandingan Akurasi KNN vs Decision Tree')
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
            
            print("Processing completed successfully!")
            
            # Kirim hasil ke frontend
            return jsonify({
                'knn_accuracy': knn_acc,
                'dt_accuracy': dt_acc,
                'knn_cm_img': knn_cm_img,
                'dt_cm_img': dt_cm_img,
                'accuracy_img': accuracy_img,
                'results_table': results_table,
                'categories': unique_labels
            })
            
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
        
        # Kirim hasil ke frontend
        return jsonify({
            'title': title,
            'knn_prediction': knn_pred,
            'dt_prediction': dt_pred,
            'nearest_neighbors': nearest_titles
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
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
    # Jalankan server pada port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)