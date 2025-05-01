<?php
require_once 'config.php';
require_once 'database_config.php';

// Inisialisasi database
$database = new Database();
$connection = $database->getConnection();

// Check database status
$dbStatus = false;
try {
    $dbTest = $database->fetch("SELECT COUNT(*) as count FROM categories");
    $dbStatus = ($dbTest !== false);
} catch (Exception $e) {
    $dbStatus = false;
}

// Ambil data dari database
$results = [];
$upload_id = isset($_GET['upload_id']) ? intval($_GET['upload_id']) : null;
$fileInfo = null;

if ($dbStatus) {
    if ($upload_id) {
        // Ambil hasil berdasarkan upload_id
        $results = $database->fetchAll("
            SELECT p.id, p.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
                   p.confidence, p.prediction_date
            FROM predictions p
            LEFT JOIN categories c1 ON p.actual_category_id = c1.id
            LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
            LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
            WHERE p.upload_file_id = ?
            ORDER BY p.id DESC
        ", [$upload_id]);
        
        // Dapatkan info file
        $fileInfo = $database->fetch("
            SELECT id, original_filename, file_size, upload_date
            FROM uploaded_files
            WHERE id = ?
        ", [$upload_id]);
    } else {
        // Ambil 100 hasil terbaru
        $results = $database->fetchAll("
            SELECT p.id, p.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
                   p.confidence, p.prediction_date
            FROM predictions p
            LEFT JOIN categories c1 ON p.actual_category_id = c1.id
            LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
            LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
            ORDER BY p.prediction_date DESC
            LIMIT 100
        ");
    }

    // Hitung statistik
    $totalCount = count($results);
    $correctKNN = 0;
    $correctDT = 0;

    foreach ($results as $row) {
        if ($row['actual'] == $row['knn_pred']) $correctKNN++;
        if ($row['actual'] == $row['dt_pred']) $correctDT++;
    }

    $knnAccuracy = $totalCount > 0 ? ($correctKNN / $totalCount) * 100 : 0;
    $dtAccuracy = $totalCount > 0 ? ($correctDT / $totalCount) * 100 : 0;
}
?>
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hasil Klasifikasi Judul Skripsi</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
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
        
        .badge-large {
            font-size: 1rem;
            padding: 0.5rem 0.7rem;
            border-radius: 8px;
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
        
        .fade-in {
            opacity: 0;
            animation: fadeIn 0.5s forwards;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .results-table {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        }
        
        .results-table thead th {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            color: var(--dark-color);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
            border: none;
        }
        
        .results-table tbody tr {
            transition: all 0.2s;
        }
        
        .results-table tbody tr:hover {
            background-color: rgba(67, 97, 238, 0.05);
            transform: scale(1.01);
        }
        
        .results-table .badge {
            padding: 6px 10px;
            font-weight: 500;
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
        
        .bg-pattern {
            background-color: var(--primary-color);
            background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            padding: 50px 0;
            margin-bottom: 30px;
            border-radius: 0 0 20px 20px;
        }
        
        .model-detail-card {
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        
        .model-icon {
            position: absolute;
            right: -20px;
            bottom: -20px;
            font-size: 8rem;
            opacity: 0.05;
            transform: rotate(15deg);
        }
        
        .knn-border {
            border-left: 4px solid var(--primary-color);
        }
        
        .dt-border {
            border-left: 4px solid #198754;
        }
        
        .metric-pill {
            font-size: 0.85rem;
            padding: 0.25rem 0.5rem;
            border-radius: 20px;
        }
        
        .highlight-row {
            background-color: rgba(13, 110, 253, 0.1);
        }
        
        .input-group {
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .input-group-text {
            background-color: var(--primary-color);
            color: white;
            border: none;
        }
        
        .filter-btn.active {
            background-color: var(--primary-color);
            color: white;
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
            
            .badge-large {
                font-size: 0.85rem;
            }
        }
    </style>
</head>
<body>
    <div class="bg-pattern text-center d-none d-md-block">
        <h1 class="display-4 fw-bold text-white mb-0">Hasil Klasifikasi Judul Skripsi</h1>
        <p class="lead text-white-50">Analisis dan Perbandingan Model Machine Learning</p>
    </div>

    <div class="container">
        <!-- Tampilan ketika tidak ada data -->
        <div id="noData" class="card fade-in" style="<?php echo (empty($results)) ? '' : 'display: none;'; ?>">
            <div class="card-header">
                <h3 class="text-center mb-0"><i class="bi bi-exclamation-triangle me-2"></i>Data Tidak Ditemukan</h3>
            </div>
            <div class="card-body text-center">
                <img src="https://cdn-icons-png.flaticon.com/512/6134/6134065.png" alt="No Data" class="img-fluid mb-4" style="max-width: 150px;">
                <p class="mb-4">Tidak ada hasil klasifikasi yang tersedia.</p>
                <p class="mb-4">Silakan upload file Excel terlebih dahulu untuk melihat hasil klasifikasi.</p>
                <a href="index.php" class="btn btn-primary">
                    <i class="bi bi-arrow-left me-2"></i>Kembali ke Halaman Upload
                </a>
            </div>
        </div>
        
        <!-- Tampilan hasil klasifikasi -->
        <div id="results" style="<?php echo (!empty($results)) ? '' : 'display: none;'; ?>">
            <div class="card mb-4 fade-in">
                <div class="card-header">
                    <h2 class="text-center mb-0">Hasil Klasifikasi Judul Skripsi</h2>
                </div>
                <div class="card-body">
                    <div class="step-container">
                        <div class="step active">1</div>
                        <div class="step active">2</div>
                        <div class="step">3</div>
                    </div>
                    
                    <!-- Breadcrumb -->
                    <nav aria-label="breadcrumb" class="mb-3">
                        <ol class="breadcrumb">
                            <li class="breadcrumb-item"><a href="index.php" class="text-decoration-none">Beranda</a></li>
                            <li class="breadcrumb-item active" aria-current="page">Hasil Klasifikasi</li>
                        </ol>
                    </nav>
                    
                    <div class="alert alert-success fade-in" style="animation-delay: 0.1s">
                        <div class="d-flex align-items-center">
                            <div class="me-3">
                                <i class="bi bi-check-circle-fill fs-3"></i>
                            </div>
                            <div>
                                <strong>Berhasil!</strong> Klasifikasi telah selesai diproses.
                                <div class="text-success small">Model machine learning telah melakukan prediksi kategori pada judul-judul skripsi.</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Database Status -->
                    <div class="alert alert-<?php echo $dbStatus ? 'success' : 'danger'; ?> mb-3 fade-in" style="animation-delay: 0.2s">
                        <i class="bi bi-<?php echo $dbStatus ? 'check-circle' : 'exclamation-triangle'; ?>-fill me-2"></i>
                        Status Database: <?php echo $dbStatus ? 'Terhubung' : 'Tidak Terhubung - Periksa konfigurasi database Anda'; ?>
                    </div>
                    
                    <?php if ($fileInfo): ?>
                    <div class="alert alert-info mb-3 fade-in" style="animation-delay: 0.25s">
                        <div class="d-flex align-items-center">
                            <i class="bi bi-file-earmark-excel me-3 fs-4"></i>
                            <div>
                                <strong>Informasi File:</strong>
                                <p class="mb-0">Nama: <?= $fileInfo['original_filename'] ?>, 
                                Ukuran: <?= number_format($fileInfo['file_size'] / 1024, 2) ?> KB, 
                                Diupload: <?= date('d/m/Y H:i', strtotime($fileInfo['upload_date'])) ?></p>
                            </div>
                        </div>
                    </div>
                    <?php endif; ?>
                    
                    <!-- Menu Navigasi -->
                    <nav class="navbar navbar-expand-lg navbar-light fade-in" style="animation-delay: 0.3s">
                        <div class="container-fluid">
                            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" 
                                aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                                <span class="navbar-toggler-icon"></span>
                            </button>
                            <div class="collapse navbar-collapse" id="navbarNav">
                                <ul class="navbar-nav">
                                    <li class="nav-item">
                                        <a class="nav-link" href="index.php">
                                            <i class="bi bi-house-door"></i> Beranda
                                        </a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link active" href="result.php">
                                            <i class="bi bi-clipboard-data"></i> Hasil Klasifikasi
                                        </a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link" href="visualisasi.php<?= $upload_id ? '?upload_id='.$upload_id : '' ?>">
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
                                            <li><a class="dropdown-item" href="export.php?type=excel<?= $upload_id ? '&id='.$upload_id : '' ?>">Excel</a></li>
                                            <li><a class="dropdown-item" href="export.php?type=pdf<?= $upload_id ? '&id='.$upload_id : '' ?>">PDF</a></li>
                                        </ul>
                                    </li>
                                </ul>
                                <div class="ms-auto">
                                    <button id="showPredictionBtn" class="btn btn-primary btn-sm">
                                        <i class="bi bi-search me-1"></i> Prediksi Judul Baru
                                    </button>
                                </div>
                                <!-- API Status Indicator -->
                                <div class="ms-2 d-flex align-items-center">
                                    <span class="badge rounded-pill bg-light text-dark me-2" id="apiStatus">
                                        <i class="bi bi-circle-fill text-secondary me-1" style="font-size: 0.5rem;"></i>
                                        API Status
                                    </span>
                                </div>
                            </div>
                        </div>
                    </nav>
                    
                    <!-- Perbandingan Akurasi -->
                    <div class="section-card fade-in" style="animation-delay: 0.4s">
                        <div class="card border-0 bg-transparent">
                            <div class="card-body p-0">
                                <div class="row g-4">
                                    <div class="col-md-8">
                                        <div class="card h-100">
                                            <div class="card-header bg-white">
                                                <div class="d-flex align-items-center">
                                                    <i class="bi bi-bar-chart-line fs-4 me-2 text-primary"></i>
                                                    <h5 class="mb-0">Perbandingan Akurasi</h5>
                                                </div>
                                            </div>
                                            <div class="card-body">
                                                <div class="img-container">
                                                    <canvas id="accuracyChart" height="250"></canvas>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="card h-100">
                                            <div class="card-header bg-white">
                                                <div class="d-flex align-items-center">
                                                    <i class="bi bi-award fs-4 me-2 text-primary"></i>
                                                    <h5 class="mb-0">Hasil Akurasi</h5>
                                                </div>
                                            </div>
                                            <div class="card-body">
                                                <div class="mt-2">
                                                    <div class="mb-4">
                                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                                            <span class="fw-bold">KNN</span>
                                                            <span id="knnAccuracy" class="badge bg-primary badge-large">0%</span>
                                                        </div>
                                                        <div class="progress" style="height: 10px;">
                                                            <div id="knnProgress" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
                                                        </div>
                                                    </div>
                                                    <div class="mb-4">
                                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                                            <span class="fw-bold">Decision Tree</span>
                                                            <span id="dtAccuracy" class="badge bg-success badge-large">0%</span>
                                                        </div>
                                                        <div class="progress" style="height: 10px;">
                                                            <div id="dtProgress" class="progress-bar progress-bar-striped progress-bar-animated bg-success" role="progressbar" style="width: 0%"></div>
                                                        </div>
                                                    </div>
                                                </div>
                                                
                                                <div class="mt-4">
                                                    <h6 class="text-muted fw-bold mb-3">Model Terbaik:</h6>
                                                    <div class="alert alert-primary py-3">
                                                        <div class="d-flex align-items-center">
                                                            <i class="bi bi-trophy-fill fs-4 me-3"></i>
                                                            <div>
                                                                <strong id="bestModel" class="fs-5">-</strong>
                                                                <div class="small">Akurasi tertinggi</div>
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
                    </div>
                    
                    <!-- Model Detail Cards -->
                    <div class="section-card fade-in" style="animation-delay: 0.5s">
                        <div class="card">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-info-circle fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Detail Model</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <!-- KNN Detail Card -->
                                    <div class="col-md-6 mb-4">
                                        <div class="card model-detail-card knn-border h-100">
                                            <div class="card-body position-relative">
                                                <i class="bi bi-bullseye model-icon"></i>
                                                <div class="d-flex align-items-center mb-3">
                                                    <div class="me-3 bg-primary bg-opacity-10 p-3 rounded-circle">
                                                        <i class="bi bi-bullseye fs-3 text-primary"></i>
                                                    </div>
                                                    <div>
                                                        <h5 class="card-title mb-0 text-primary">K-Nearest Neighbors</h5>
                                                        <div class="text-muted small">Algoritma berbasis jarak</div>
                                                    </div>
                                                </div>
                                                <div id="knnDetailMetrics">
                                                    <div class="d-flex gap-2 mt-3 mb-2 flex-wrap">
                                                        <span class="badge bg-primary-subtle text-primary metric-pill">
                                                            <i class="bi bi-people me-1"></i>
                                                            n_neighbors: <span id="knnNeighbors">3</span>
                                                        </span>
                                                        <span class="badge bg-secondary metric-pill">
                                                            <i class="bi bi-rulers me-1"></i>
                                                            metric: minkowski
                                                        </span>
                                                    </div>
                                                    
                                                    <div class="table-responsive mt-3">
                                                        <table class="table table-sm table-bordered border-primary">
                                                            <thead class="table-primary">
                                                                <tr>
                                                                    <th>Metric</th>
                                                                    <th>Score</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody id="knnMetricsTableBody">
                                                                <!-- KNN metrics will be inserted here -->
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- Decision Tree Detail Card -->
                                    <div class="col-md-6 mb-4">
                                        <div class="card model-detail-card dt-border h-100">
                                            <div class="card-body position-relative">
                                                <i class="bi bi-diagram-3 model-icon"></i>
                                                <div class="d-flex align-items-center mb-3">
                                                    <div class="me-3 bg-success bg-opacity-10 p-3 rounded-circle">
                                                        <i class="bi bi-diagram-3 fs-3 text-success"></i>
                                                    </div>
                                                    <div>
                                                        <h5 class="card-title mb-0 text-success">Decision Tree</h5>
                                                        <div class="text-muted small">Algoritma berbasis pohon keputusan</div>
                                                    </div>
                                                </div>
                                                <div id="dtDetailMetrics">
                                                    <div class="d-flex gap-2 mt-3 mb-2 flex-wrap">
                                                        <span class="badge bg-success-subtle text-success metric-pill">
                                                            <i class="bi bi-tree me-1"></i>
                                                            Depth: <span id="dtDepth">0</span>
                                                        </span>
                                                        <span class="badge bg-success-subtle text-success metric-pill">
                                                            <i class="bi bi-journal me-1"></i>
                                                            Leaves: <span id="dtLeaves">0</span>
                                                        </span>
                                                        <span class="badge bg-secondary metric-pill">
                                                            <i class="bi bi-calculator me-1"></i>
                                                            criterion: <span id="dtCriterion">gini</span>
                                                        </span>
                                                    </div>
                                                    
                                                    <div class="table-responsive mt-3">
                                                        <table class="table table-sm table-bordered border-success">
                                                            <thead class="table-success">
                                                                <tr>
                                                                    <th>Metric</th>
                                                                    <th>Score</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody id="dtMetricsTableBody">
                                                                <!-- DT metrics will be inserted here -->
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Hasil Detail -->
                    <div class="section-card fade-in" style="animation-delay: 0.7s">
                        <div class="card">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-table fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Detail Hasil Prediksi</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <div class="row mb-4">
                                    <div class="col-md-6 mb-3 mb-md-0">
                                        <div class="input-group">
                                            <span class="input-group-text"><i class="bi bi-search"></i></span>
                                            <input type="text" id="tableSearch" class="form-control" placeholder="Cari judul skripsi...">
                                        </div>
                                    </div>
                                    <div class="col-md-6 text-md-end">
                                        <div class="btn-group" role="group">
                                            <button type="button" class="btn btn-outline-primary filter-btn active" data-filter="all">Semua</button>
                                            <button type="button" class="btn btn-outline-success filter-btn" data-filter="correct">Benar</button>
                                            <button type="button" class="btn btn-outline-danger filter-btn" data-filter="incorrect">Salah</button>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="table-responsive">
                                    <table class="table table-hover results-table" id="resultsTable">
                                        <thead>
                                            <tr>
                                                <th width="5%" class="text-center">No</th>
                                                <th width="45%">Judul Skripsi</th>
                                                <th width="15%" class="text-center">Label Sebenarnya</th>
                                                <th width="15%" class="text-center">Prediksi KNN</th>
                                                <th width="15%" class="text-center">Prediksi DT</th>
                                                <th width="5%" class="text-center">Detail</th>
                                            </tr>
                                        </thead>
                                        <tbody id="resultsTableBody">
                                            <!-- Hasil prediksi akan ditampilkan di sini -->
                                        </tbody>
                                    </table>
                                </div>
                                
                                <div id="noResultsMessage" style="display: none;" class="alert alert-warning mt-3">
                                    <div class="d-flex align-items-center">
                                        <i class="bi bi-exclamation-triangle-fill me-3 fs-4"></i>
                                        <div>
                                            <strong>Tidak ada hasil</strong><br>
                                            <span>Tidak ada hasil yang sesuai dengan pencarian.</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Navigation Buttons -->
                    <div class="d-flex justify-content-between mt-4 fade-in" style="animation-delay: 0.8s">
                        <a href="index.php" class="btn btn-outline-secondary">
                            <i class="bi bi-arrow-left me-1"></i> Kembali ke Upload
                        </a>
                        <a href="history.php" class="btn btn-outline-primary">
                            <i class="bi bi-clock-history me-1"></i> Riwayat Klasifikasi
                        </a>
                        <a href="visualisasi.php<?= $upload_id ? '?upload_id='.$upload_id : '' ?>" class="btn btn-primary">
                            <i class="bi bi-bar-chart me-1"></i> Lihat Visualisasi
                        </a>
                    </div>
                </div>
                <div class="card-footer text-center text-muted">
                    <small>Sistem Klasifikasi Judul Skripsi v<?php echo APP_VERSION; ?> | <?php echo date('Y'); ?></small>
                </div>
            </div>
        </div>
       
        <!-- Form Prediksi Judul Baru -->
        <div id="predictionForm" class="card mb-4 fade-in">
            <div class="card-header">
                <h3 class="text-center mb-0">Prediksi Judul Skripsi Baru</h3>
                <p class="text-center mb-0 mt-2">Langkah 3: Uji Model dengan Judul Baru</p>
            </div>
            <div class="card-body">
                <div class="step-container">
                    <div class="step active">1</div>
                    <div class="step active">2</div>
                    <div class="step active">3</div>
                </div>
                
                <!-- Breadcrumb -->
                <nav aria-label="breadcrumb" class="mb-3">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="index.php" class="text-decoration-none">Beranda</a></li>
                        <li class="breadcrumb-item"><a href="result.php" class="text-decoration-none">Hasil Klasifikasi</a></li>
                        <li class="breadcrumb-item active" aria-current="page">Prediksi Judul Baru</li>
                    </ol>
                </nav>
                
                <form id="predictForm" class="mb-4">
                    <div class="mb-3">
                        <label for="title" class="form-label fw-bold">
                            <i class="bi bi-pencil me-1"></i>
                            Masukkan Judul Skripsi:
                        </label>
                        <textarea id="title" class="form-control" rows="3" required placeholder="Contoh: Sistem Informasi Manajemen Perpustakaan Berbasis Web"></textarea>
                        <div class="form-text">Masukkan judul skripsi lengkap untuk mendapatkan hasil prediksi yang lebih akurat.</div>
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
                    <p class="text-muted small">IndoBERT sedang menganalisis judul yang Anda masukkan</p>
                </div>
                
                <div id="predictionResult" class="prediction-animation" style="display: none;">
                    <div class="card bg-light">
                        <div class="card-header bg-white">
                            <h5 class="mb-0"><i class="bi bi-lightbulb me-2"></i>Hasil Prediksi</h5>
                        </div>
                        <div class="card-body">
                            <h6 class="card-title fw-bold">Judul:</h6>
                            <p id="predictedTitle" class="mb-4 p-3 bg-white rounded border"></p>
                            
                            <div class="row mt-4 g-4">
                                <div class="col-md-6">
                                    <div class="card bg-primary text-white h-100">
                                        <div class="card-body text-center">
                                            <h5 class="card-title">
                                                <i class="bi bi-bullseye me-2"></i>KNN
                                            </h5>
                                            <h3 id="predictedKNN" class="display-6 my-3 fw-bold"></h3>
                                            <p class="mb-0 small" id="knnConfidence"></p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card bg-success text-white h-100">
                                        <div class="card-body text-center">
                                            <h5 class="card-title">
                                                <i class="bi bi-diagram-3 me-2"></i>Decision Tree
                                            </h5>
                                            <h3 id="predictedDT" class="display-6 my-3 fw-bold"></h3>
                                            <p class="mb-0 small">Berdasarkan pohon keputusan</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="alert alert-info mt-4 mb-0">
                                <div class="d-flex">
                                    <i class="bi bi-info-circle-fill fs-4 me-3"></i>
                                    <div>
                                        <strong>Hasil Analisis</strong>
                                        <p class="mb-0" id="predictionMessage"></p>
                                    </div>
                                </div>
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
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title" id="detailModalLabel">Detail Prediksi</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <h6 class="fw-bold">Judul Skripsi:</h6>
                        <p id="modalTitle" class="p-3 bg-light rounded border mb-4"></p>
                        
                        <div class="table-responsive">
                            <table class="table table-bordered">
                                <thead class="table-light">
                                    <tr>
                                        <th>Kategori</th>
                                        <th>Label Sebenarnya</th>
                                        <th>KNN</th>
                                        <th>Decision Tree</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td id="modalCategory" class="fw-bold text-primary"></td>
                                        <td id="modalActual"></td>
                                        <td id="modalKNN"></td>
                                        <td id="modalDT"></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        
                        <div class="alert alert-light border mt-4 mb-0">
                            <div class="d-flex">
                                <i class="bi bi-info-circle me-3 fs-4 text-primary"></i>
                                <div>
                                    <strong>Informasi</strong>
                                    <p class="mb-0 small">Hasil ini berdasarkan model yang telah dilatih dengan data skripsi. Nilai akurasi menunjukkan tingkat ketepatan model dalam klasifikasi. Kategori utama untuk judul ini adalah <span id="modalMainCategory" class="fw-bold"></span>.</p>
                                </div>
                            </div>
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
            // Ambil data dari PHP
            const databaseResults = <?= json_encode($results) ?>;
            const knnAccuracy = <?= json_encode($knnAccuracy) ?>;
            const dtAccuracy = <?= json_encode($dtAccuracy) ?>;
            
            // Animasi elemen saat halaman dimuat
            const animatedElements = document.querySelectorAll('.fade-in');
            animatedElements.forEach((element, index) => {
                element.style.animationDelay = (index * 0.1) + 's';
            });
            
            // Cek status API
            const apiStatusEl = document.getElementById('apiStatus');
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
            
            if (databaseResults && databaseResults.length > 0) {
                // Tampilkan data dari database
                document.getElementById('results').style.display = 'block';
                document.getElementById('noData').style.display = 'none';
                
                // Tampilkan akurasi
                document.getElementById('knnAccuracy').textContent = knnAccuracy.toFixed(2) + '%';
                document.getElementById('dtAccuracy').textContent = dtAccuracy.toFixed(2) + '%';
                
                // Update progress bars
                setTimeout(() => {
                    document.getElementById('knnProgress').style.width = knnAccuracy + '%';
                    document.getElementById('dtProgress').style.width = dtAccuracy + '%';
                }, 500);
                
                // Tentukan model terbaik
                if (knnAccuracy >= dtAccuracy) {
                    document.getElementById('bestModel').textContent = 'KNN (' + knnAccuracy.toFixed(2) + '%)';
                } else {
                    document.getElementById('bestModel').textContent = 'Decision Tree (' + dtAccuracy.toFixed(2) + '%)';
                }
                
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
                                'rgba(67, 97, 238, 0.7)',
                                'rgba(25, 135, 84, 0.7)'
                            ],
                            borderColor: [
                                'rgba(67, 97, 238, 1)',
                                'rgba(25, 135, 84, 1)'
                            ],
                            borderWidth: 1,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        animation: {
                            duration: 2000,
                            easing: 'easeOutQuart'
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return context.raw.toFixed(2) + '%';
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
                            },
                            x: {
                                grid: {
                                    display: false
                                }
                            }
                        }
                    }
                });
                
                // Hitung metrik precision, recall, dan F1 score untuk KNN
                const knnMetricsTable = document.getElementById('knnMetricsTableBody');
                knnMetricsTable.innerHTML = '';
                
                // Hitung untuk setiap kategori
                const categories = [...new Set(databaseResults.map(r => r.actual))];
                
                // Hitung confusion matrix dan metrik turunannya
                let knnTruePositives = {}, knnFalsePositives = {}, knnFalseNegatives = {}, knnTrueNegatives = {};
                
                // Inisialisasi perhitungan untuk setiap kategori
                categories.forEach(category => {
                    knnTruePositives[category] = 0;
                    knnFalsePositives[category] = 0;
                    knnFalseNegatives[category] = 0;
                    knnTrueNegatives[category] = 0;
                });
                
                // Hitung confusion matrix
                databaseResults.forEach(row => {
                    categories.forEach(category => {
                        if (row.actual === category && row.knn_pred === category) {
                            knnTruePositives[category]++;
                        } else if (row.actual !== category && row.knn_pred === category) {
                            knnFalsePositives[category]++;
                        } else if (row.actual === category && row.knn_pred !== category) {
                            knnFalseNegatives[category]++;
                        } else if (row.actual !== category && row.knn_pred !== category) {
                            knnTrueNegatives[category]++;
                        }
                    });
                });
                
                // Hitung metrik
                let totalKnnPrecision = 0, totalKnnRecall = 0, totalKnnF1 = 0;
                let weightedKnnPrecision = 0, weightedKnnRecall = 0, weightedKnnF1 = 0;
                let totalSamples = databaseResults.length;
                
                categories.forEach(category => {
                    const tp = knnTruePositives[category];
                    const fp = knnFalsePositives[category];
                    const fn = knnFalseNegatives[category];
                    const precision = tp / (tp + fp) || 0;
                    const recall = tp / (tp + fn) || 0;
                    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
                    
                    const categoryCount = databaseResults.filter(r => r.actual === category).length;
                    const weight = categoryCount / totalSamples;
                    
                    weightedKnnPrecision += precision * weight;
                    weightedKnnRecall += recall * weight;
                    weightedKnnF1 += f1 * weight;
                    
                    totalKnnPrecision += precision;
                    totalKnnRecall += recall;
                    totalKnnF1 += f1;
                });
                
                // Rata-rata metrik (macro)
                const macroKnnPrecision = totalKnnPrecision / categories.length;
                const macroKnnRecall = totalKnnRecall / categories.length;
                const macroKnnF1 = totalKnnF1 / categories.length;
                
                // Tampilkan metrik KNN
                knnMetricsTable.innerHTML += `
                    <tr>
                        <td>Accuracy</td>
                        <td>${knnAccuracy.toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Precision (weighted)</td>
                        <td>${(weightedKnnPrecision * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Recall (weighted)</td>
                        <td>${(weightedKnnRecall * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>F1-Score (weighted)</td>
                        <td>${(weightedKnnF1 * 100).toFixed(2)}%</td>
                    </tr>
                `;
                
                // Hitung metrik precision, recall, dan F1 score untuk Decision Tree
                const dtMetricsTable = document.getElementById('dtMetricsTableBody');
                dtMetricsTable.innerHTML = '';
                
                // Hitung confusion matrix dan metrik turunannya
                let dtTruePositives = {}, dtFalsePositives = {}, dtFalseNegatives = {}, dtTrueNegatives = {};
                
                // Inisialisasi perhitungan untuk setiap kategori
                categories.forEach(category => {
                    dtTruePositives[category] = 0;
                    dtFalsePositives[category] = 0;
                    dtFalseNegatives[category] = 0;
                    dtTrueNegatives[category] = 0;
                });
                
                // Hitung confusion matrix
                databaseResults.forEach(row => {
                    categories.forEach(category => {
                        if (row.actual === category && row.dt_pred === category) {
                            dtTruePositives[category]++;
                        } else if (row.actual !== category && row.dt_pred === category) {
                            dtFalsePositives[category]++;
                        } else if (row.actual === category && row.dt_pred !== category) {
                            dtFalseNegatives[category]++;
                        } else if (row.actual !== category && row.dt_pred !== category) {
                            dtTrueNegatives[category]++;
                        }
                    });
                });
                
                // Hitung metrik
                let totalDtPrecision = 0, totalDtRecall = 0, totalDtF1 = 0;
                let weightedDtPrecision = 0, weightedDtRecall = 0, weightedDtF1 = 0;
                
                categories.forEach(category => {
                    const tp = dtTruePositives[category];
                    const fp = dtFalsePositives[category];
                    const fn = dtFalseNegatives[category];
                    const precision = tp / (tp + fp) || 0;
                    const recall = tp / (tp + fn) || 0;
                    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
                    
                    const categoryCount = databaseResults.filter(r => r.actual === category).length;
                    const weight = categoryCount / totalSamples;
                    
                    weightedDtPrecision += precision * weight;
                    weightedDtRecall += recall * weight;
                    weightedDtF1 += f1 * weight;
                    
                    totalDtPrecision += precision;
                    totalDtRecall += recall;
                    totalDtF1 += f1;
                });
                
                // Rata-rata metrik (macro)
                const macroDtPrecision = totalDtPrecision / categories.length;
                const macroDtRecall = totalDtRecall / categories.length;
                const macroDtF1 = totalDtF1 / categories.length;
                
                // Tampilkan metrik Decision Tree
                dtMetricsTable.innerHTML += `
                    <tr>
                        <td>Accuracy</td>
                        <td>${dtAccuracy.toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Precision (weighted)</td>
                        <td>${(weightedDtPrecision * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Recall (weighted)</td>
                        <td>${(weightedDtRecall * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>F1-Score (weighted)</td>
                        <td>${(weightedDtF1 * 100).toFixed(2)}%</td>
                    </tr>
                `;
                
                // Default values for tree parameters
                document.getElementById('dtDepth').textContent = "N/A";
                document.getElementById('dtLeaves').textContent = "N/A";
                
                // Tampilkan tabel hasil
                const tableBody = document.getElementById('resultsTableBody');
                tableBody.innerHTML = '';
                
                databaseResults.forEach((row, index) => {
                    const correctKNN = row.actual === row.knn_pred;
                    const correctDT = row.actual === row.dt_pred;
                    
                    const tr = document.createElement('tr');
                    tr.dataset.title = row.title;
                    tr.dataset.actual = row.actual;
                    tr.dataset.knn = row.knn_pred;
                    tr.dataset.dt = row.dt_pred;
                    
                    // Determine overall correctness
                    const correctCount = (correctKNN ? 1 : 0) + (correctDT ? 1 : 0);
                    
                    tr.dataset.correct = (correctCount === 2) ? 'all' : 
                                        (correctCount > 0) ? 'partial' : 'none';
                    
                    if (correctCount === 2) {
                        tr.classList.add('table-success');
                    } else if (correctCount === 0) {
                        tr.classList.add('table-danger');
                    }
                    
                    const knnClass = correctKNN ? 'bg-success' : 'bg-danger';
                    const dtClass = correctDT ? 'bg-success' : 'bg-danger';
                    
                    tr.innerHTML = `
                        <td class="text-center">${index + 1}</td>
                        <td>${row.title}</td>
                        <td class="text-center"><span class="badge bg-secondary rounded-pill">${row.actual}</span></td>
                        <td class="text-center"><span class="badge ${knnClass} rounded-pill">${row.knn_pred}</span></td>
                        <td class="text-center"><span class="badge ${dtClass} rounded-pill">${row.dt_pred}</span></td>
                        <td class="text-center">
                            <button class="btn btn-sm btn-primary detail-btn" data-bs-toggle="modal" data-bs-target="#detailModal" 
                                data-index="${index}">
                                <i class="bi bi-info-circle"></i>
                            </button>
                        </td>
                    `;
                    
                    tableBody.appendChild(tr);
                });
                
                // Tambahkan hover effect
                const tableRows = document.querySelectorAll('#resultsTableBody tr');
                tableRows.forEach(row => {
                    row.addEventListener('mouseenter', function() {
                        this.classList.add('highlight-row');
                    });
                    row.addEventListener('mouseleave', function() {
                        this.classList.remove('highlight-row');
                    });
                });
                
                // Event handler untuk filter dan pencarian
                const filterButtons = document.querySelectorAll('.filter-btn');
                const searchInput = document.getElementById('tableSearch');
                
                // Fungsi untuk filter dan pencarian
                function filterTable() {
                    const searchTerm = searchInput.value.toLowerCase();
                    const activeFilter = document.querySelector('.filter-btn.active')?.dataset.filter || 'all';
                    let visibleCount = 0;
                    
                    Array.from(tableBody.getElementsByTagName('tr')).forEach(row => {
                        const title = row.dataset.title.toLowerCase();
                        const correct = row.dataset.correct;
                        
                        // Filter berdasarkan teks
                        const matchesSearch = title.includes(searchTerm);
                        
                        // Filter berdasarkan status (benar/salah)
                        const matchesFilter = 
                            activeFilter === 'all' || 
                            (activeFilter === 'correct' && correct === 'all') ||
                            (activeFilter === 'incorrect' && correct !== 'all');
                        
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
                
                // Aktifkan pencarian
                if (searchInput) {
                    searchInput.addEventListener('input', filterTable);
                }
                
                // Aktifkan handler untuk tombol detail
                const detailButtons = document.querySelectorAll('.detail-btn');
                detailButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        const index = this.dataset.index;
                        const row = databaseResults[index];
                        
                        document.getElementById('modalTitle').textContent = row.title;
                        document.getElementById('modalCategory').textContent = row.actual;
                        document.getElementById('modalActual').textContent = row.actual;
                        document.getElementById('modalKNN').textContent = row.knn_pred;
                        document.getElementById('modalDT').textContent = row.dt_pred;
                        document.getElementById('modalMainCategory').textContent = row.actual;
                    });
                });
            } else {
                document.getElementById('results').style.display = 'none';
                document.getElementById('noData').style.display = 'block';
            }
            
            // Event handler untuk tombol prediksi
            const showPredictionBtn = document.getElementById('showPredictionBtn');
            if (showPredictionBtn) {
                showPredictionBtn.addEventListener('click', function() {
                    document.getElementById('predictionForm').style.display = 'block';
                    document.getElementById('results').style.display = 'none';
                    document.getElementById('predictionResult').style.display = 'none';
                    document.getElementById('title').value = '';
                    window.scrollTo(0, 0);
                });
            }
            
            // Event handler untuk pembatalan prediksi
            const cancelPredict = document.getElementById('cancelPredict');
            if (cancelPredict) {
                cancelPredict.addEventListener('click', function() {
                    document.getElementById('predictionForm').style.display = 'none';
                    document.getElementById('results').style.display = 'block';
                    window.scrollTo(0, 0);
                });
            }
            
            // Event handler untuk kembali ke hasil
            const backToResultsBtn = document.getElementById('backToResultsBtn');
            if (backToResultsBtn) {
                backToResultsBtn.addEventListener('click', function() {
                    document.getElementById('predictionForm').style.display = 'none';
                    document.getElementById('results').style.display = 'block';
                    window.scrollTo(0, 0);
                });
            }
            
            // Event handler untuk prediksi baru
            const newPredictionBtn = document.getElementById('newPredictionBtn');
            if (newPredictionBtn) {
                newPredictionBtn.addEventListener('click', function() {
                    document.getElementById('predictionResult').style.display = 'none';
                    document.getElementById('title').value = '';
                    document.getElementById('predictForm').style.display = 'block';
                });
            }
            
            // Event handler untuk form prediksi
            const predictForm = document.getElementById('predictForm');
            if (predictForm) {
                predictForm.addEventListener('submit', function(e) {
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
                    
                    // Tambahkan upload_id jika ada
                    const upload_id = <?= $upload_id ? $upload_id : 'null' ?>;
                    
                    // Kirim request ke API
                    fetch('<?php echo API_URL; ?>/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ 
                            title: title,
                            upload_id: upload_id // Tambahkan upload_id
                        })
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
                        const knnMatch = data.knn_prediction === data.dt_prediction;
                        
                        if (knnMatch) {
                            predictionMessage.textContent = "Kedua model memberikan hasil prediksi yang sama (" + data.knn_prediction + "), menunjukkan tingkat kepercayaan yang tinggi terhadap hasil klasifikasi judul ini.";
                        } else {
                            predictionMessage.textContent = "Model memberikan prediksi yang berbeda. Anda dapat mempertimbangkan hasil KNN (" + data.knn_prediction + ") atau Decision Tree (" + data.dt_prediction + ") berdasarkan akurasi yang telah ditunjukkan pada data pengujian.";
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
            }
        });
    </script>
</body>
</html>