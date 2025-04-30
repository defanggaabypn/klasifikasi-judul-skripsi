<?php
require_once 'config.php';
require_once 'database_config.php';

// Inisialisasi database
$database = new Database();
$conn = $database->getConnection();

// Check database status
$dbStatus = false;
try {
    $dbTest = $database->fetch("SELECT COUNT(*) as count FROM categories");
    $dbStatus = ($dbTest !== false);
} catch (Exception $e) {
    $dbStatus = false;
}
?>
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Klasifikasi Judul Skripsi</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <style>
        body {
            padding-top: 50px;
            padding-bottom: 50px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 800px;
        }
        #loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
            z-index: 9999;
            text-align: center;
            padding-top: 200px;
            color: white;
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
        .feature-box {
            border-left: 4px solid #0d6efd;
            padding: 10px 15px;
            margin-bottom: 15px;
            background-color: #f8f9fa;
            border-radius: 0 5px 5px 0;
        }
        .feature-title {
            font-weight: bold;
            color: #0d6efd;
        }
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner-border text-light" role="status" style="width: 3rem; height: 3rem;"></div>
        <h3 class="mt-3 text-white">Sedang Memproses...</h3>
        <p class="text-white">Ini mungkin membutuhkan waktu beberapa menit karena IndoBERT sedang menganalisis judul-judul</p>
        <div class="progress mt-3" style="width: 50%; margin: 0 auto;">
            <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 100%"></div>
        </div>
    </div>

    <div class="container">
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h2 class="text-center mb-0">Sistem Klasifikasi Judul Skripsi</h2>
            </div>
            <div class="card-body">
                <!-- Progress Steps -->
                <div class="step-container">
                    <div class="step active">1</div>
                    <div class="step">2</div>
                    <div class="step">3</div>
                </div>
                
                <!-- Navigasi Menu -->
                <nav class="navbar navbar-expand-lg navbar-light bg-light mb-4">
                    <div class="container-fluid">
                        <span class="navbar-brand">Menu:</span>
                        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" 
                            aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                            <span class="navbar-toggler-icon"></span>
                        </button>
                        <div class="collapse navbar-collapse" id="navbarNav">
                            <ul class="navbar-nav">
                                <li class="nav-item">
                                    <a class="nav-link active" href="index.php">
                                        <i class="bi bi-house-door"></i> Beranda
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="result.php">
                                        <i class="bi bi-clipboard-data"></i> Hasil Klasifikasi
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="visualisasi.php">
                                        <i class="bi bi-bar-chart"></i> Visualisasi Data
                                    </a>
                                </li>
                                <li class="nav-item dropdown">
                                    <a class="nav-link dropdown-toggle" href="#" id="navbarDropdown" role="button" 
                                       data-bs-toggle="dropdown" aria-expanded="false">
                                        <i class="bi bi-download"></i> Export
                                    </a>
                                    <ul class="dropdown-menu" aria-labelledby="navbarDropdown">
                                        <li><a class="dropdown-item" href="export.php?type=excel">Excel</a></li>
                                        <li><a class="dropdown-item" href="export.php?type=pdf">PDF</a></li>
                                    </ul>
                                </li>
                            </ul>
                            <!-- API Status Indicator -->
                            <div class="ms-auto d-flex align-items-center">
                                <small class="badge rounded-pill bg-light text-dark me-2" id="apiStatus">
                                    <i class="bi bi-circle-fill text-secondary me-1" style="font-size: 0.5rem;"></i>
                                    API Status
                                </small>
                            </div>
                        </div>
                    </div>
                </nav>
                
                <!-- Database Status -->
                <div class="alert alert-<?php echo $dbStatus ? 'success' : 'danger'; ?> mb-3">
                    <i class="bi bi-<?php echo $dbStatus ? 'check-circle' : 'exclamation-triangle'; ?>-fill me-2"></i>
                    Status Database: <?php echo $dbStatus ? 'Terhubung' : 'Tidak Terhubung - Periksa konfigurasi database Anda'; ?>
                </div>
                
                <?php
                // Tampilkan pesan error jika ada
                if (isset($_SESSION['error'])) {
                    showAlert($_SESSION['error'], 'danger');
                    unset($_SESSION['error']);
                }
                
                // Tampilkan pesan sukses jika ada
                if (isset($_SESSION['success'])) {
                    showAlert($_SESSION['success'], 'success');
                    unset($_SESSION['success']);
                }
                ?>
                
                <!-- Alur Penggunaan -->
                <div class="alert alert-info">
                    <i class="bi bi-info-circle-fill me-2"></i>
                    <strong>Alur Penggunaan:</strong> 
                    <ol class="mb-0 ms-4">
                        <li>Upload file Excel berisi judul skripsi</li>
                        <li>Lihat hasil klasifikasi (Halaman Hasil)</li>
                        <li>Analisis visualisasi data (Halaman Visualisasi)</li>
                    </ol>
                </div>
                
                <div class="alert alert-info">
                    <i class="bi bi-info-circle-fill me-2"></i>
                    <strong>Petunjuk:</strong> Upload file Excel (.xlsx) yang berisi judul-judul skripsi. Sistem akan mengklasifikasikan judul ke dalam kategori yang sesuai menggunakan model IndoBERT dan algoritma machine learning.
                </div>
                
                <div class="card mb-4 border">
                    <div class="card-header bg-light">
                        <h5 class="mb-0"><i class="bi bi-upload me-2"></i>Upload File</h5>
                    </div>
                    <div class="card-body">
                        <form id="uploadForm">
                            <div class="mb-3">
                                <label for="file" class="form-label">Pilih File Excel (.xlsx):</label>
                                <input type="file" name="file" id="file" class="form-control" accept=".xlsx" required>
                                <div class="form-text">Ukuran maksimum file: <?php echo MAX_UPLOAD_SIZE; ?>MB</div>
                            </div>
                            
                            <div class="text-center mt-4">
                                <button type="button" id="downloadTemplateBtn" class="btn btn-outline-secondary me-2">
                                    <i class="bi bi-file-earmark-excel me-1"></i> Download Template
                                </button>
                                <button type="submit" class="btn btn-primary">
                                    <i class="bi bi-cloud-upload me-1"></i> Upload dan Proses
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
                
                <!-- Statistik Database (jika ada) -->
                <?php
                $totalTitles = $database->fetch("SELECT COUNT(*) as count FROM thesis_titles");
                $totalPredictions = $database->fetch("SELECT COUNT(*) as count FROM predictions");
                
                if ($totalTitles && $totalTitles['count'] > 0): 
                ?>
                <div class="card mb-4 border">
                    <div class="card-header bg-light">
                        <h5 class="mb-0"><i class="bi bi-graph-up me-2"></i>Statistik Database</h5>
                    </div>
                    <div class="card-body">
                        <div class="row text-center">
                            <div class="col-md-6">
                                <div class="card h-100 border-primary">
                                    <div class="card-body">
                                        <h1 class="display-4 text-primary"><?= number_format($totalTitles['count']) ?></h1>
                                        <p class="lead">Judul Skripsi</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card h-100 border-success">
                                    <div class="card-body">
                                        <h1 class="display-4 text-success"><?= number_format($totalPredictions['count']) ?></h1>
                                        <p class="lead">Prediksi</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="text-center mt-3">
                            <a href="visualisasi.php" class="btn btn-outline-primary">
                                <i class="bi bi-bar-chart me-1"></i> Lihat Visualisasi Data
                            </a>
                        </div>
                    </div>
                </div>
                <?php endif; ?>
                
                <div class="card mb-4">
                    <div class="card-header bg-light">
                        <h5 class="mb-0"><i class="bi bi-info-circle me-2"></i>Informasi Sistem</h5>
                    </div>
                    <div class="card-body">
                        <p>Sistem ini akan mengklasifikasikan judul skripsi ke dalam kategori:</p>
                        
                        <div class="row">
                            <div class="col-md-4 mb-3">
                                <div class="feature-box">
                                    <div class="feature-title">RPL</div>
                                    <small>Rekayasa Perangkat Lunak</small>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="feature-box">
                                    <div class="feature-title">Jaringan</div>
                                    <small>Jaringan Komputer</small>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="feature-box">
                                    <div class="feature-title">Multimedia</div>
                                    <small>Desain dan Multimedia</small>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-3">
                            <h6><i class="bi bi-lightning-charge me-2"></i>Fitur Utama:</h6>
                            <ul>
                                <li>Pemrosesan bahasa alami menggunakan IndoBERT</li>
                                <li>Klasifikasi dengan algoritma KNN dan Decision Tree</li>
                                <li>Visualisasi hasil dengan grafik perbandingan dan confusion matrix</li>
                                <li>Prediksi kategori untuk judul baru</li>
                                <li>Export hasil ke Excel dan PDF</li>
                                <li>Analisis performa detail KNN vs Decision Tree</li>
                            </ul>
                        </div>
                        
                        <div class="alert alert-secondary mt-3 mb-0">
                            <small>
                                <i class="bi bi-code-slash me-1"></i> 
                                <strong>Tech Stack:</strong> Python Flask + IndoBERT + Scikit-learn | PHP + MySQL + Bootstrap
                            </small>
                        </div>
                    </div>
                </div>
                
            </div>
            <div class="card-footer text-center text-muted">
                <small>Sistem Klasifikasi Judul Skripsi v<?php echo APP_VERSION; ?> | <?php echo date('Y'); ?></small>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const uploadForm = document.getElementById('uploadForm');
            const fileInput = document.getElementById('file');
            const loadingEl = document.getElementById('loading');
            const downloadTemplateBtn = document.getElementById('downloadTemplateBtn');
            const apiStatusEl = document.getElementById('apiStatus');
            
            // Cek status API
            checkApiStatus();
            
            function checkApiStatus() {
                fetch('<?php echo API_URL; ?>/template', { 
                    method: 'GET',
                    headers: { 'Accept': 'application/json' },
                    // Timeout 5 detik
                    signal: AbortSignal.timeout(5000)
                })
                .then(response => {
                    if (response.ok) {
                        apiStatusEl.innerHTML = '<i class="bi bi-circle-fill text-success me-1" style="font-size: 0.5rem;"></i> API Connected';
                        apiStatusEl.classList.replace('text-dark', 'text-success');
                    } else {
                        throw new Error('API response not OK');
                    }
                })
                .catch(error => {
                    apiStatusEl.innerHTML = '<i class="bi bi-circle-fill text-danger me-1" style="font-size: 0.5rem;"></i> API Disconnected';
                    apiStatusEl.classList.replace('text-dark', 'text-danger');
                    console.error('API Status Check Error:', error);
                });
            }
            
            // Validasi file sebelum upload
            function validateFile() {
                const file = fileInput.files[0];
                if (!file) {
                    showCustomAlert('Silakan pilih file terlebih dahulu!', 'warning');
                    return false;
                }
                
                // Validasi ekstensi file
                if (!file.name.endsWith('.xlsx')) {
                    showCustomAlert('Hanya file Excel (.xlsx) yang diperbolehkan!', 'warning');
                    return false;
                }
                
                // Validasi ukuran file
                const maxSize = <?php echo MAX_UPLOAD_SIZE; ?> * 1024 * 1024; // MB to bytes
                if (file.size > maxSize) {
                    showCustomAlert(`Ukuran file terlalu besar (maksimum <?php echo MAX_UPLOAD_SIZE; ?>MB)!`, 'warning');
                    return false;
                }
                
                return true;
            }
            
            // Tampilkan alert dengan Bootstrap
            function showCustomAlert(message, type = 'danger') {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
                alertDiv.innerHTML = `
                    ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                `;
                
                // Tambahkan alert ke dalam form
                uploadForm.insertAdjacentElement('beforebegin', alertDiv);
                
                // Auto-dismiss setelah 5 detik
                setTimeout(() => {
                    alertDiv.classList.remove('show');
                    setTimeout(() => alertDiv.remove(), 300);
                }, 5000);
            }
            
            // Handle form submit
            uploadForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                if (!validateFile()) {
                    return;
                }
                
                // Tampilkan loading spinner
                loadingEl.style.display = 'block';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                // Kirim file ke API
                fetch('<?php echo API_URL; ?>/process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    console.log("API response status:", response.status);
                    if (!response.ok) {
                        throw new Error('Network response was not ok: ' + response.status);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log("API response data:", data);
                    // Sembunyikan loading spinner
                    loadingEl.style.display = 'none';
                    
                    // Cek jika ada error dari API
                    if (data.error) {
                        showCustomAlert(`Error: ${data.error}`, 'danger');
                        return;
                    }
                    
                    // Simpan data hasil ke sessionStorage
                    sessionStorage.setItem('classificationResults', JSON.stringify(data));
                    
                    // Simpan file info ke database (optional)
                    saveUploadInfo(fileInput.files[0], data.categories)
                        .then(() => {
                            // Redirect ke halaman hasil
                            window.location.href = 'result.php';
                        })
                        .catch(error => {
                            console.warn('Warning: Could not save upload info to database', error);
                            // Tetap redirect ke halaman hasil
                            window.location.href = 'result.php';
                        });
                })
                .catch(error => {
                    // Sembunyikan loading spinner
                    loadingEl.style.display = 'none';
                    
                    console.error('Error details:', error);
                    showCustomAlert('Terjadi kesalahan saat memproses file. Silakan coba lagi. Error: ' + error.message, 'danger');
                });
            });
            
            // Simpan info upload ke database
            function saveUploadInfo(file, categories) {
                const fileInfo = {
                    filename: file.name,
                    size: file.size,
                    categories: categories || []
                };
                
                return fetch('save_upload.php', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(fileInfo)
                })
                .then(response => response.json());
            }
            
            // Handle download template
            downloadTemplateBtn.addEventListener('click', function() {
                fetch('<?php echo API_URL; ?>/template')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        showCustomAlert(`Error: ${data.error}`, 'danger');
                        return;
                    }
                    
                    // Buat link download dan klik otomatis
                    const link = document.createElement('a');
                    link.href = 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + data.data;
                    link.download = data.filename;
                    link.style.display = 'none';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                })
                .catch(error => {
                    console.error('Error:', error);
                    showCustomAlert('Gagal mengunduh template. Silakan coba lagi. Error: ' + error.message, 'danger');
                });
            });
        });
    </script>
</body>
</html>