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
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    <style>
        :root {
          --primary-color: #4361ee;
          --secondary-color: #3f37c9;
          --success-color: #4cc9f0;
          --info-color: #4895ef;
          --warning-color: #f72585;
          --light-color: #f8f9fa;
          --dark-color: #212529;
        }
        
        body {
            font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #f0f2f5;
            padding-top: 50px;
            padding-bottom: 50px;
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
            background-color: rgba(0,0,0,0.8);
            backdrop-filter: blur(5px);
            z-index: 9999;
            text-align: center;
            padding-top: 200px;
            color: white;
        }
        
        .card {
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.05);
            border: none;
            transition: transform 0.3s, box-shadow 0.3s;
            overflow: hidden;
            margin-bottom: 30px;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 20px rgba(0, 0, 0, 0.1);
        }
        
        .card-header {
            border-bottom: none;
            padding: 20px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
        }
        
        .btn {
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.3s;
        }

        .btn-primary {
            background: var(--primary-color);
            border-color: var(--primary-color);
        }

        .btn-primary:hover {
            background: var(--secondary-color);
            border-color: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .step-container {
            display: flex;
            justify-content: center;
            margin: 40px 0;
        }
        
        .step {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: #dee2e6;
            color: #6c757d;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 40px;
            position: relative;
            font-weight: bold;
            font-size: 18px;
        }
        
        .step.active {
            background-color: var(--primary-color);
            color: white;
        }
        
        .step:not(:last-child):after {
            content: '';
            position: absolute;
            width: 80px;
            height: 2px;
            background-color: #dee2e6;
            top: 50%;
            left: 40px;
        }
        
        .step.active:not(:last-child):after {
            background-color: var(--primary-color);
        }
        
        .feature-card {
            border-left: 4px solid var(--primary-color);
            background-color: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 0 10px 10px 0;
            transition: all 0.3s;
        }
        
        .feature-card:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }
        
        .feature-title {
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 5px;
            display: flex;
            align-items: center;
        }
        
        .feature-title i {
            margin-right: 8px;
        }
        
        .count-up {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .bg-pattern {
            background-color: var(--primary-color);
            background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            padding: 50px 0;
            margin-bottom: 30px;
            border-radius: 0 0 20px 20px;
        }
        
        .navbar {
            border-radius: 10px;
            padding: 10px 15px;
            margin-bottom: 25px;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .alert {
            border-radius: 10px;
            border: none;
            padding: 15px;
        }
        
        .fade-in {
            opacity: 0;
            animation: fadeIn 0.5s forwards;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 0 15px;
            }
            
            .card-header {
                padding: 15px;
            }
            
            .step {
                width: 30px;
                height: 30px;
                margin: 0 20px;
                font-size: 14px;
            }
            
            .step:not(:last-child):after {
                width: 40px;
                left: 30px;
            }
        }
        
        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.7; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.7; }
        }

        .loading-pulse {
            animation: pulse 1.5s infinite ease-in-out;
        }
        
        .stat-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }
        
        .stat-icon {
            font-size: 2rem;
            width: 70px;
            height: 70px;
            line-height: 70px;
            background: rgba(67, 97, 238, 0.1);
            color: var(--primary-color);
            margin: 0 auto 15px;
            border-radius: 50%;
        }
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner-border text-light loading-pulse" role="status" style="width: 3rem; height: 3rem;"></div>
        <h3 class="mt-3 text-white">Sedang Memproses...</h3>
        <p class="text-white">Ini mungkin membutuhkan waktu beberapa menit karena IndoBERT sedang menganalisis judul-judul</p>
        <div class="progress mt-3" style="width: 50%; margin: 0 auto;">
            <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 100%"></div>
        </div>
    </div>

    <div class="bg-pattern text-center d-none d-md-block">
        <h1 class="display-4 fw-bold text-white mb-0">Sistem Klasifikasi Judul Skripsi</h1>
        <p class="lead text-white-50">Powered by IndoBERT & Machine Learning</p>
    </div>

    <div class="container">
        <div class="card mb-4 fade-in">
            <div class="card-header">
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
                <nav class="navbar navbar-expand-lg navbar-light">
                    <div class="container-fluid">
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
                                <li class="nav-item">
                    <a class="nav-link" href="history.php">
                        <i class="bi bi-clock-history"></i> Riwayat Klasifikasi
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
                                <span class="badge rounded-pill bg-light text-dark me-2" id="apiStatus">
                                    <i class="bi bi-circle-fill text-secondary me-1" style="font-size: 0.5rem;"></i>
                                    API Status
                                </span>
                            </div>
                        </div>
                    </div>
                </nav>
                
                <!-- Database Status -->
                <div class="alert alert-<?php echo $dbStatus ? 'success' : 'danger'; ?> mb-3 fade-in" style="animation-delay: 0.1s">
                    <i class="bi bi-<?php echo $dbStatus ? 'check-circle' : 'exclamation-triangle'; ?>-fill me-2"></i>
                    Status Database: <?php echo $dbStatus ? 'Terhubung' : 'Tidak Terhubung - Periksa konfigurasi database Anda'; ?>
                </div>
                
                <?php
                // Tampilkan pesan error jika ada
                if (isset($_SESSION['error'])) {
                    echo '<div class="alert alert-danger fade-in" style="animation-delay: 0.2s">';
                    echo '<i class="bi bi-exclamation-triangle-fill me-2"></i>'.$_SESSION['error'];
                    echo '</div>';
                    unset($_SESSION['error']);
                }
                
                // Tampilkan pesan sukses jika ada
                if (isset($_SESSION['success'])) {
                    echo '<div class="alert alert-success fade-in" style="animation-delay: 0.2s">';
                    echo '<i class="bi bi-check-circle-fill me-2"></i>'.$_SESSION['success'];
                    echo '</div>';
                    unset($_SESSION['success']);
                }
                ?>
                
                <!-- Alur Penggunaan -->
                <div class="alert alert-info fade-in" style="animation-delay: 0.3s">
                    <i class="bi bi-info-circle-fill me-2"></i>
                    <strong>Alur Penggunaan:</strong> 
                    <ol class="mb-0 ms-4">
                        <li>Upload file Excel berisi judul skripsi</li>
                        <li>Lihat hasil klasifikasi (Halaman Hasil)</li>
                        <li>Analisis visualisasi data (Halaman Visualisasi)</li>
                    </ol>
                </div>
                
                <div class="alert alert-info fade-in" style="animation-delay: 0.4s">
                    <i class="bi bi-info-circle-fill me-2"></i>
                    <strong>Petunjuk:</strong> Upload file Excel (.xlsx) yang berisi judul-judul skripsi. Sistem akan mengklasifikasikan judul ke dalam kategori yang sesuai menggunakan model IndoBERT dan algoritma machine learning.
                </div>
                
                <div class="card mb-4 border-0 shadow-sm fade-in" style="animation-delay: 0.5s">
                    <div class="card-header bg-white">
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
<div class="card mb-4 border-0 shadow-sm fade-in" style="animation-delay: 0.6s">
    <div class="card-header bg-white">
        <h5 class="mb-0"><i class="bi bi-graph-up me-2"></i>Statistik Database</h5>
    </div>
    <div class="card-body">
        <div class="row text-center g-3">
            <div class="col-md-6">
                <div class="stat-card h-100">
                    <div class="stat-icon">
                        <i class="bi bi-file-earmark-text"></i>
                    </div>
                    <h2 class="count-up" id="titleCount"><?= number_format($totalTitles['count']) ?></h2>
                    <p class="text-muted mb-0">Judul Skripsi</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="stat-card h-100">
                    <div class="stat-icon">
                        <i class="bi bi-lightning"></i>
                    </div>
                    <h2 class="count-up" id="predCount"><?= number_format($totalPredictions['count']) ?></h2>
                    <p class="text-muted mb-0">Prediksi</p>
                </div>
            </div>
        </div>
        
        <div class="text-center mt-4">
            <a href="visualisasi.php" class="btn btn-outline-primary me-2">
                <i class="bi bi-bar-chart me-1"></i> Lihat Visualisasi Data
            </a>
            <a href="history.php" class="btn btn-outline-primary">
                <i class="bi bi-clock-history me-1"></i> Lihat Riwayat Klasifikasi
            </a>
        </div>
    </div>
</div>
<?php endif; ?>
                
                <div class="card mb-4 fade-in" style="animation-delay: 0.7s">
                    <div class="card-header bg-white">
                        <h5 class="mb-0"><i class="bi bi-info-circle me-2"></i>Informasi Sistem</h5>
                    </div>
                    <div class="card-body">
                        <p>Sistem ini akan mengklasifikasikan judul skripsi ke dalam kategori:</p>
                        
                        <div class="row">
                            <div class="col-md-4 mb-3">
                                <div class="feature-card">
                                    <div class="feature-title"><i class="bi bi-code-slash"></i> RPL</div>
                                    <div>Rekayasa Perangkat Lunak</div>
                                    <small class="text-muted">Web, Mobile, Desktop</small>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="feature-card">
                                    <div class="feature-title"><i class="bi bi-hdd-network"></i> Jaringan</div>
                                    <div>Jaringan Komputer</div>
                                    <small class="text-muted">Networking, Security</small>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="feature-card">
                                    <div class="feature-title"><i class="bi bi-camera"></i> Multimedia</div>
                                    <div>Desain dan Multimedia</div>
                                    <small class="text-muted">Grafis, Animasi, Video</small>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-4">
                            <h6 class="fw-bold"><i class="bi bi-lightning-charge me-2"></i>Fitur Utama:</h6>
                            <div class="row g-3 mt-2">
                                <div class="col-md-6">
                                    <div class="d-flex">
                                        <div class="me-3 text-primary"><i class="bi bi-cpu"></i></div>
                                        <div>Pemrosesan bahasa alami menggunakan IndoBERT</div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="d-flex">
                                        <div class="me-3 text-primary"><i class="bi bi-graph-up"></i></div>
                                        <div>Klasifikasi dengan algoritma KNN dan Decision Tree</div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="d-flex">
                                        <div class="me-3 text-primary"><i class="bi bi-bar-chart"></i></div>
                                        <div>Visualisasi hasil dengan grafik perbandingan</div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="d-flex">
                                        <div class="me-3 text-primary"><i class="bi bi-search"></i></div>
                                        <div>Prediksi kategori untuk judul baru</div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="d-flex">
                                        <div class="me-3 text-primary"><i class="bi bi-file-arrow-down"></i></div>
                                        <div>Export hasil ke Excel dan PDF</div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="d-flex">
                                        <div class="me-3 text-primary"><i class="bi bi-file-earmark-bar-graph"></i></div>
                                        <div>Analisis performa detail algoritma</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="alert alert-secondary mt-4 mb-0">
                            <div class="d-flex align-items-center">
                                <i class="bi bi-code-slash me-3 fs-4"></i>
                                <div>
                                    <strong>Tech Stack:</strong> Python Flask + IndoBERT + Scikit-learn | PHP + MySQL + Bootstrap
                                </div>
                            </div>
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
            // Animasi elemen saat halaman dimuat
            const animatedElements = document.querySelectorAll('.fade-in');
            animatedElements.forEach((element, index) => {
                element.style.animationDelay = (index * 0.1) + 's';
            });
            
            const uploadForm = document.getElementById('uploadForm');
            const fileInput = document.getElementById('file');
            const loadingEl = document.getElementById('loading');
            const downloadTemplateBtn = document.getElementById('downloadTemplateBtn');
            const apiStatusEl = document.getElementById('apiStatus');
            
            // Cek status API
            checkApiStatus();
            
            // Animasi statistik jika ada
            if (document.getElementById('titleCount')) {
                animateValue('titleCount', 0, <?= $totalTitles ? $totalTitles['count'] : 0 ?>, 1500);
            }
            if (document.getElementById('predCount')) {
                animateValue('predCount', 0, <?= $totalPredictions ? $totalPredictions['count'] : 0 ?>, 1500);
            }
            
            function animateValue(id, start, end, duration) {
                const obj = document.getElementById(id);
                if (!obj) return;
                
                let startTimestamp = null;
                const step = (timestamp) => {
                    if (!startTimestamp) startTimestamp = timestamp;
                    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
                    obj.innerHTML = Math.floor(progress * (end - start) + start).toLocaleString();
                    if (progress < 1) {
                        window.requestAnimationFrame(step);
                    }
                };
                window.requestAnimationFrame(step);
            }
            
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