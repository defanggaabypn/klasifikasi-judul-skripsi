<?php
require_once 'config.php';
?>
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hasil Klasifikasi Judul Skripsi</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            padding-top: 50px;
            padding-bottom: 50px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1000px;
        }
        .section-card {
            margin-bottom: 30px;
        }
        .img-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .img-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        #noData {
            display: none;
        }
        #results {
            display: none;
        }
        #predictionForm {
            display: none;
        }
        #loadingPredict {
            display: none;
        }
        .card {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: none;
            border-radius: 10px;
            overflow: hidden;
        }
        .card-header {
            border-bottom: none;
            padding: 20px;
        }
        .badge-large {
            font-size: 1rem;
            padding: 0.5rem 0.7rem;
        }
        .step-container {
            display: flex;
            justify-content: center;
            margin: 30px 0;
        }
        .step {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: #dee2e6;
            color: #6c757d;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 30px;
            position: relative;
            font-weight: bold;
        }
        .step.active {
            background-color: #0d6efd;
            color: white;
        }
        .step:not(:last-child):after {
            content: '';
            position: absolute;
            width: 60px;
            height: 2px;
            background-color: #dee2e6;
            top: 50%;
            left: 30px;
        }
        .step.active:not(:last-child):after {
            background-color: #0d6efd;
        }
        .prediction-animation {
            animation: fadeInUp 0.5s ease-out;
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .table th, .table td {
            vertical-align: middle;
        }
        .highlight-row {
            background-color: rgba(13, 110, 253, 0.1);
        }
        .accordion-button:not(.collapsed) {
            background-color: rgba(13, 110, 253, 0.1);
            color: #0d6efd;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Tampilan ketika tidak ada data -->
        <div id="noData" class="card">
            <div class="card-header bg-warning text-dark">
                <h3 class="text-center"><i class="bi bi-exclamation-triangle me-2"></i>Data Tidak Ditemukan</h3>
            </div>
            <div class="card-body text-center">
                <p class="mb-4">Tidak ada hasil klasifikasi yang tersedia.</p>
                <p class="mb-4">Silakan upload file Excel terlebih dahulu untuk melihat hasil klasifikasi.</p>
                <a href="index.php" class="btn btn-primary">
                    <i class="bi bi-arrow-left me-2"></i>Kembali ke Halaman Upload
                </a>
            </div>
        </div>
        
        <!-- Tampilan hasil klasifikasi -->
        <div id="results">
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    <h2 class="text-center mb-0">Hasil Klasifikasi Judul Skripsi</h2>
                </div>
                <div class="card-body">
                    <div class="step-container">
                        <div class="step active">1</div>
                        <div class="step active">2</div>
                        <div class="step">3</div>
                    </div>
                    
                    <div class="alert alert-success">
                        <i class="bi bi-check-circle-fill me-2"></i>
                        <strong>Berhasil!</strong> Klasifikasi telah selesai diproses.
                    </div>
                    
                    <!-- Tombol untuk form prediksi -->
                    <div class="text-end mb-3">
                        <button id="showPredictionBtn" class="btn btn-primary">
                            <i class="bi bi-search me-1"></i> Prediksi Judul Baru
                        </button>
                    </div>
                    
                    <!-- Perbandingan Akurasi -->
                    <div class="section-card">
                        <div class="card border">
                            <div class="card-header bg-light">
                                <h5 class="mb-0"><i class="bi bi-bar-chart-line me-2"></i>Perbandingan Akurasi</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-8">
                                        <div class="img-container">
                                            <canvas id="accuracyChart" height="250"></canvas>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="card h-100">
                                            <div class="card-body">
                                                <h5 class="card-title">Hasil Akurasi:</h5>
                                                <div class="mt-4">
                                                    <div class="mb-3">
                                                        <div class="d-flex justify-content-between">
                                                            <span>KNN</span>
                                                            <span id="knnAccuracy" class="badge bg-primary badge-large">0%</span>
                                                        </div>
                                                        <div class="progress mt-2">
                                                            <div id="knnProgress" class="progress-bar" role="progressbar" style="width: 0%"></div>
                                                        </div>
                                                    </div>
                                                    <div class="mb-3">
                                                        <div class="d-flex justify-content-between">
                                                            <span>Decision Tree</span>
                                                            <span id="dtAccuracy" class="badge bg-success badge-large">0%</span>
                                                        </div>
                                                        <div class="progress mt-2">
                                                            <div id="dtProgress" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Confusion Matrix -->
                    <div class="section-card">
                        <div class="card border">
                            <div class="card-header bg-light">
                                <h5 class="mb-0"><i class="bi bi-grid-3x3 me-2"></i>Confusion Matrix</h5>
                            </div>
                            <div class="card-body">
                                <p>
                                    <small class="text-muted">
                                        <i class="bi bi-info-circle me-1"></i>
                                        Confusion matrix menunjukkan perbandingan antara label sebenarnya (actual) dan hasil prediksi model.
                                    </small>
                                </p>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="card">
                                            <div class="card-header bg-primary text-white">
                                                <h5 class="mb-0">KNN</h5>
                                            </div>
                                            <div class="card-body">
                                                <div class="img-container" id="knnMatrix">
                                                    <!-- Confusion Matrix KNN akan ditampilkan di sini -->
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="card">
                                            <div class="card-header bg-success text-white">
                                                <h5 class="mb-0">Decision Tree</h5>
                                            </div>
                                            <div class="card-body">
                                                <div class="img-container" id="dtMatrix">
                                                    <!-- Confusion Matrix Decision Tree akan ditampilkan di sini -->
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Hasil Detail -->
                    <div class="section-card">
                        <div class="card border">
                            <div class="card-header bg-light">
                                <h5 class="mb-0"><i class="bi bi-table me-2"></i>Detail Hasil Prediksi</h5>
                            </div>
                            <div class="card-body">
                                <div class="row mb-3">
                                    <div class="col-md-6">
                                        <div class="input-group">
                                            <span class="input-group-text"><i class="bi bi-search"></i></span>
                                            <input type="text" id="tableSearch" class="form-control" placeholder="Cari judul skripsi...">
                                        </div>
                                    </div>
                                    <div class="col-md-6 text-end">
                                        <div class="btn-group" role="group">
                                            <button type="button" class="btn btn-outline-primary filter-btn" data-filter="all">Semua</button>
                                            <button type="button" class="btn btn-outline-success filter-btn" data-filter="correct">Benar</button>
                                            <button type="button" class="btn btn-outline-danger filter-btn" data-filter="incorrect">Salah</button>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="table-responsive">
                                    <table class="table table-hover table-bordered" id="resultsTable">
                                        <thead class="table-dark">
                                            <tr>
                                                <th width="5%">No</th>
                                                <th width="45%">Judul Skripsi</th>
                                                <th width="15%">Label Sebenarnya</th>
                                                <th width="15%">Prediksi KNN</th>
                                                <th width="15%">Prediksi Decision Tree</th>
                                                <th width="5%">Detail</th>
                                            </tr>
                                        </thead>
                                        <tbody id="resultsTableBody">
                                            <!-- Hasil prediksi akan ditampilkan di sini -->
                                        </tbody>
                                    </table>
                                </div>
                                
                                <div id="noResultsMessage" style="display: none;" class="alert alert-warning mt-3">
                                    <i class="bi bi-exclamation-triangle me-2"></i>
                                    Tidak ada hasil yang sesuai dengan pencarian.
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Tombol Kembali -->
                    <div class="text-center mt-4">
                        <a href="index.php" class="btn btn-primary">
                            <i class="bi bi-arrow-left me-2"></i>Kembali ke Halaman Utama
                        </a>
                    </div>
                </div>
                <div class="card-footer text-center text-muted">
                    <small>Sistem Klasifikasi Judul Skripsi v<?php echo APP_VERSION; ?></small>
                </div>
            </div>
        </div>
        
        <!-- Form Prediksi Judul Baru -->
        <div id="predictionForm" class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h3 class="text-center mb-0">Prediksi Judul Skripsi Baru</h3>
                <p class="text-center mb-0 mt-2">Langkah 3: Uji Model dengan Judul Baru</p>
            </div>
            <div class="card-body">
                <div class="step-container">
                    <div class="step active">1</div>
                    <div class="step active">2</div>
                    <div class="step active">3</div>
                </div>
                
                <form id="predictForm" class="mb-4">
                    <div class="mb-3">
                        <label for="title" class="form-label">
                            <i class="bi bi-pencil me-1"></i>
                            Masukkan Judul Skripsi:
                        </label>
                        <textarea id="title" class="form-control" rows="3" required placeholder="Contoh: Sistem Informasi Manajemen Perpustakaan Berbasis Web"></textarea>
                    </div>
                    <div class="text-center mb-3">
                        <button type="submit" class="btn btn-primary">
                            <i class="bi bi-lightning me-1"></i> Prediksi
                        </button>
                        <button type="button" id="cancelPredict" class="btn btn-outline-secondary ms-2">
                            <i class="bi bi-x me-1"></i> Batal
                        </button>
                    </div>
                </form>
                
                <div id="loadingPredict" class="text-center my-4">
                    <div class="spinner-border text-primary" role="status"></div>
                    <p class="mt-2">Sedang memproses prediksi...</p>
                </div>
                
                <div id="predictionResult" class="prediction-animation" style="display: none;">
                    <div class="card bg-light">
                        <div class="card-header bg-light">
                            <h5 class="mb-0"><i class="bi bi-lightbulb me-2"></i>Hasil Prediksi</h5>
                        </div>
                        <div class="card-body">
                            <h6 class="card-title">Judul:</h6>
                            <p id="predictedTitle" class="mb-4 p-2 bg-white rounded"></p>
                            
                            <div class="row mt-4">
                                <div class="col-md-6">
                                    <div class="card bg-primary text-white">
                                        <div class="card-body text-center">
                                            <h5 class="card-title">
                                                <i class="bi bi-bullseye me-2"></i>KNN
                                            </h5>
                                            <h3 id="predictedKNN" class="my-3"></h3>
                                            <p class="mb-0 small" id="knnConfidence"></p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card bg-success text-white">
                                        <div class="card-body text-center">
                                            <h5 class="card-title">
                                                <i class="bi bi-diagram-3 me-2"></i>Decision Tree
                                            </h5>
                                            <h3 id="predictedDT" class="my-3"></h3>
                                            <p class="mb-0 small">Berdasarkan keputusan pohon</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="alert alert-info mt-4 mb-0">
                                <i class="bi bi-info-circle me-2"></i>
                                <span id="predictionMessage">Kedua model memberikan hasil prediksi yang sama, menunjukkan tingkat kepercayaan yang tinggi.</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="text-center mt-4">
                        <button type="button" id="newPredictionBtn" class="btn btn-outline-primary">
                            <i class="bi bi-plus-circle me-1"></i> Prediksi Judul Lain
                        </button>
                        <button type="button" id="backToResultsBtn" class="btn btn-primary ms-2">
                            <i class="bi bi-arrow-left me-1"></i> Kembali ke Hasil
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Modal untuk detail hasil -->
        <div class="modal fade" id="detailModal" tabindex="-1" aria-labelledby="detailModalLabel" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title" id="detailModalLabel">Detail Prediksi</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <h6>Judul Skripsi:</h6>
                        <p id="modalTitle" class="p-2 bg-light rounded"></p>
                        
                        <table class="table table-bordered mt-3">
                            <tr>
                                <th>Kategori</th>
                                <th>Label Sebenarnya</th>
                                <th>KNN</th>
                                <th>Decision Tree</th>
                            </tr>
                            <tr>
                                <td id="modalCategory"></td>
                                <td id="modalActual"></td>
                                <td id="modalKNN"></td>
                                <td id="modalDT"></td>
                            </tr>
                        </table>
                        
                        <div class="alert alert-secondary mt-3">
                            <small>
                                <i class="bi bi-info-circle me-1"></i>
                                Hasil ini berdasarkan model yang telah dilatih dengan data skripsi.
                                Nilai akurasi menunjukkan tingkat ketepatan model dalam klasifikasi.
                            </small>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Tutup</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Cek apakah ada data hasil klasifikasi di sessionStorage
            const resultsData = sessionStorage.getItem('classificationResults');
            
            if (!resultsData) {
                document.getElementById('noData').style.display = 'block';
                return;
            }
            
            const data = JSON.parse(resultsData);
            document.getElementById('results').style.display = 'block';
            
            // Tampilkan akurasi
            const knnAccuracy = (data.knn_accuracy * 100).toFixed(2);
            const dtAccuracy = (data.dt_accuracy * 100).toFixed(2);
            
            document.getElementById('knnAccuracy').textContent = knnAccuracy + '%';
            document.getElementById('dtAccuracy').textContent = dtAccuracy + '%';
            
            // Update progress bar
            document.getElementById('knnProgress').style.width = knnAccuracy + '%';
            document.getElementById('dtProgress').style.width = dtAccuracy + '%';
            
            // Tampilkan grafik akurasi menggunakan Chart.js
            const ctx = document.getElementById('accuracyChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['KNN', 'Decision Tree'],
                    datasets: [{
                        label: 'Akurasi (%)',
                        data: [knnAccuracy, dtAccuracy],
                        backgroundColor: [
                            'rgba(13, 110, 253, 0.7)',
                            'rgba(25, 135, 84, 0.7)'
                        ],
                        borderColor: [
                            'rgba(13, 110, 253, 1)',
                            'rgba(25, 135, 84, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return context.raw + '%';
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
            
            // Tampilkan confusion matrix
            document.getElementById('knnMatrix').innerHTML = 
                `<img src="data:image/png;base64,${data.knn_cm_img}" alt="KNN Confusion Matrix" class="img-fluid">`;
            document.getElementById('dtMatrix').innerHTML = 
                `<img src="data:image/png;base64,${data.dt_cm_img}" alt="Decision Tree Confusion Matrix" class="img-fluid">`;
            
            // Tampilkan tabel hasil
            const tableBody = document.getElementById('resultsTableBody');
            tableBody.innerHTML = '';
            
            data.results_table.forEach((row, index) => {
                const correctKNN = row.actual === row.knn_pred;
                const correctDT = row.actual === row.dt_pred;
                
                const tr = document.createElement('tr');
                tr.dataset.title = row.title;
                tr.dataset.actual = row.actual;
                tr.dataset.knn = row.knn_pred;
                tr.dataset.dt = row.dt_pred;
                tr.dataset.correct = (correctKNN && correctDT) ? 'both' : 
                                    (correctKNN || correctDT) ? 'partial' : 'none';
                
                const knnClass = correctKNN ? 'table-success' : 'table-danger';
                const dtClass = correctDT ? 'table-success' : 'table-danger';
                
                tr.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${row.title}</td>
                    <td><span class="badge bg-secondary">${row.actual}</span></td>
                    <td class="${knnClass}"><span class="badge ${correctKNN ? 'bg-success' : 'bg-danger'}">${row.knn_pred}</span></td>
                    <td class="${dtClass}"><span class="badge ${correctDT ? 'bg-success' : 'bg-danger'}">${row.dt_pred}</span></td>
                    <td>
                        <button class="btn btn-sm btn-primary detail-btn" data-bs-toggle="modal" data-bs-target="#detailModal" 
                            data-index="${index}">
                            <i class="bi bi-info-circle"></i>
                        </button>
                    </td>
                `;
                
                tableBody.appendChild(tr);
            });
            
            // Event handler untuk filter dan pencarian
            const filterButtons = document.querySelectorAll('.filter-btn');
            const searchInput = document.getElementById('tableSearch');
            
            // Fungsi untuk filter dan pencarian
            function filterTable() {
                const searchTerm = searchInput.value.toLowerCase();
                const activeFilter = document.querySelector('.filter-btn.active').dataset.filter;
                let visibleCount = 0;
                
                Array.from(tableBody.getElementsByTagName('tr')).forEach(row => {
                    const title = row.dataset.title.toLowerCase();
                    const correct = row.dataset.correct;
                    
                    // Filter berdasarkan teks
                    const matchesSearch = title.includes(searchTerm);
                    
                    // Filter berdasarkan status (benar/salah)
                    const matchesFilter = 
                        activeFilter === 'all' || 
                        (activeFilter === 'correct' && correct === 'both') ||
                        (activeFilter === 'incorrect' && correct !== 'both');
                    
                    if (matchesSearch && matchesFilter) {
                        row.style.display = '';
                        visibleCount++;
                    } else {
                        row.style.display = 'none';
                    }
                });
                
                // Tampilkan pesan jika tidak ada hasil
                document.getElementById('noResultsMessage').style.display = visibleCount > 0 ? 'none' : 'block';
            }
            
            // Aktifkan filter buttons
            filterButtons.forEach(button => {
                button.addEventListener('click', function() {
                    // Reset semua button
                    filterButtons.forEach(btn => btn.classList.remove('active'));
                    
                    // Aktifkan button yang diklik
                    this.classList.add('active');
                    
                    // Filter tabel
                    filterTable();
                });
            });
            
            // Aktifkan "all" filter by default
            document.querySelector('[data-filter="all"]').classList.add('active');
            
            // Aktifkan pencarian
            searchInput.addEventListener('input', filterTable);
            
            // Event handler untuk tombol detail
            const detailButtons = document.querySelectorAll('.detail-btn');
            detailButtons.forEach(button => {
                button.addEventListener('click', function() {
                    const index = this.dataset.index;
                    const row = data.results_table[index];
                    
                    document.getElementById('modalTitle').textContent = row.full_title || row.title;
                    document.getElementById('modalCategory').textContent = row.actual;
                    document.getElementById('modalActual').textContent = row.actual;
                    document.getElementById('modalKNN').textContent = row.knn_pred;
                    document.getElementById('modalDT').textContent = row.dt_pred;
                });
            });
            
            // Event handler untuk tombol prediksi
            document.getElementById('showPredictionBtn').addEventListener('click', function() {
                document.getElementById('predictionForm').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('predictionResult').style.display = 'none';
                document.getElementById('title').value = '';
                window.scrollTo(0, 0);
            });
            
            // Event handler untuk pembatalan prediksi
            document.getElementById('cancelPredict').addEventListener('click', function() {
                document.getElementById('predictionForm').style.display = 'none';
                document.getElementById('results').style.display = 'block';
                window.scrollTo(0, 0);
            });
            
            // Event handler untuk kembali ke hasil
            document.getElementById('backToResultsBtn').addEventListener('click', function() {
                document.getElementById('predictionForm').style.display = 'none';
                document.getElementById('results').style.display = 'block';
                window.scrollTo(0, 0);
            });
            
            // Event handler untuk prediksi baru
            document.getElementById('newPredictionBtn').addEventListener('click', function() {
                document.getElementById('predictionResult').style.display = 'none';
                document.getElementById('title').value = '';
                document.getElementById('predictForm').style.display = 'block';
            });
            
            // Event handler untuk form prediksi
            document.getElementById('predictForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const title = document.getElementById('title').value.trim();
                if (!title) {
                    alert('Silakan masukkan judul skripsi!');
                    return;
                }
                
                // Tampilkan loading
                document.getElementById('loadingPredict').style.display = 'block';
                document.getElementById('predictionResult').style.display = 'none';
                document.getElementById('predictForm').style.display = 'none';
                
                // Kirim request ke API
                fetch('<?php echo API_URL; ?>/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ title: title })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    // Sembunyikan loading
                    document.getElementById('loadingPredict').style.display = 'none';
                    
                    // Tampilkan hasil prediksi
                    document.getElementById('predictedTitle').textContent = data.title;
                    document.getElementById('predictedKNN').textContent = data.knn_prediction;
                    document.getElementById('predictedDT').textContent = data.dt_prediction;
                    
                    // Tampilkan confidence info jika ada
                    if (data.nearest_neighbors && data.nearest_neighbors.length > 0) {
                        document.getElementById('knnConfidence').textContent = "Berdasarkan " + data.nearest_neighbors[0];
                    }
                    
                    // Pesan tentang hasil
                    const predictionMessage = document.getElementById('predictionMessage');
                    if (data.knn_prediction === data.dt_prediction) {
                        predictionMessage.textContent = "Kedua model memberikan hasil prediksi yang sama, menunjukkan tingkat kepercayaan yang tinggi.";
                    } else {
                        predictionMessage.textContent = "Model memberikan prediksi yang berbeda. Anda dapat mempertimbangkan keduanya.";
                    }
                    
                    document.getElementById('predictionResult').style.display = 'block';
                })
                .catch(error => {
                    // Sembunyikan loading
                    document.getElementById('loadingPredict').style.display = 'none';
                    document.getElementById('predictForm').style.display = 'block';
                    
                    console.error('Error:', error);
                    alert('Terjadi kesalahan saat memproses prediksi. Silakan coba lagi.');
                });
            });
        });
    </script>
</body>
</html>