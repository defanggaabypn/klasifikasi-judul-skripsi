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
$trainingResults = [];
$testingResults = [];
$upload_id = isset($_GET['upload_id']) ? intval($_GET['upload_id']) : null;
$fileInfo = null;

// Ambil data confusion matrix jika tersedia
$knnCmImg = null;
$dtCmImg = null;
$performanceComparisonImg = null;
$accuracyImg = null;
$trainTestComparisonImg = null;
$combinedCmImg = null;

if ($dbStatus) {
    if ($upload_id) {
        // Coba ambil data dari training_data terlebih dahulu
        $trainingResults = $database->fetchAll("
            SELECT td.id, td.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
                   td.is_correct_knn, td.is_correct_dt, td.created_at as prediction_date
            FROM training_data td
            LEFT JOIN categories c1 ON td.actual_category_id = c1.id
            LEFT JOIN categories c2 ON td.knn_prediction_id = c2.id
            LEFT JOIN categories c3 ON td.dt_prediction_id = c3.id
            WHERE td.upload_file_id = ? AND td.data_type = 'training'
            ORDER BY td.id ASC
        ", [$upload_id]);
        
        $testingResults = $database->fetchAll("
            SELECT td.id, td.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
                   td.is_correct_knn, td.is_correct_dt, td.created_at as prediction_date
            FROM training_data td
            LEFT JOIN categories c1 ON td.actual_category_id = c1.id
            LEFT JOIN categories c2 ON td.knn_prediction_id = c2.id
            LEFT JOIN categories c3 ON td.dt_prediction_id = c3.id
            WHERE td.upload_file_id = ? AND td.data_type = 'testing'
            ORDER BY td.id ASC
        ", [$upload_id]);
        
        // Jika training_data kosong, gunakan data dari predictions sebagai fallback
        if (empty($trainingResults) && empty($testingResults)) {
            // Ambil data dari predictions dan split secara manual
            $allPredictions = $database->fetchAll("
                SELECT p.id, p.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
                       p.confidence, p.prediction_date
                FROM predictions p
                LEFT JOIN categories c1 ON p.actual_category_id = c1.id
                LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
                LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
                WHERE p.upload_file_id = ?
                ORDER BY p.id ASC
            ", [$upload_id]);
            
            // Split data 80:20 untuk training:testing
            $totalCount = count($allPredictions);
            $trainingCount = (int)($totalCount * 0.8);
            
            $trainingResults = array_slice($allPredictions, 0, $trainingCount);
            $testingResults = array_slice($allPredictions, $trainingCount);
        }
        
        // Dapatkan info file
        $fileInfo = $database->fetch("
            SELECT id, original_filename, file_size, upload_date
            FROM uploaded_files
            WHERE id = ?
        ", [$upload_id]);
        
        // Ambil visualisasi berdasarkan upload_id
        $vizData = $database->fetch("
            SELECT knn_cm_img, dt_cm_img, performance_comparison_img, accuracy_img, 
                   train_test_comparison_img, combined_cm_img
            FROM model_visualizations 
            WHERE upload_file_id = ?
            ORDER BY id DESC LIMIT 1
        ", [$upload_id]);
    } else {
        // Coba ambil data training terbaru dari training_data
        $trainingResults = $database->fetchAll("
            SELECT td.id, td.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
                   td.is_correct_knn, td.is_correct_dt, td.created_at as prediction_date
            FROM training_data td
            LEFT JOIN categories c1 ON td.actual_category_id = c1.id
            LEFT JOIN categories c2 ON td.knn_prediction_id = c2.id
            LEFT JOIN categories c3 ON td.dt_prediction_id = c3.id
            WHERE td.data_type = 'training'
            ORDER BY td.created_at DESC
            LIMIT 100
        ");
        
        $testingResults = $database->fetchAll("
            SELECT td.id, td.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
                   td.is_correct_knn, td.is_correct_dt, td.created_at as prediction_date
            FROM training_data td
            LEFT JOIN categories c1 ON td.actual_category_id = c1.id
            LEFT JOIN categories c2 ON td.knn_prediction_id = c2.id
            LEFT JOIN categories c3 ON td.dt_prediction_id = c3.id
            WHERE td.data_type = 'testing'
            ORDER BY td.created_at DESC
            LIMIT 100
        ");
        
        // Jika training_data kosong, gunakan data dari predictions
        if (empty($trainingResults) && empty($testingResults)) {
            $allPredictions = $database->fetchAll("
                SELECT p.id, p.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
                       p.confidence, p.prediction_date
                FROM predictions p
                LEFT JOIN categories c1 ON p.actual_category_id = c1.id
                LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
                LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
                ORDER BY p.prediction_date DESC
                LIMIT 100
            ");
            
            // Split data untuk simulasi training/testing
            $totalCount = count($allPredictions);
            $trainingCount = (int)($totalCount * 0.8);
            
            $trainingResults = array_slice($allPredictions, 0, $trainingCount);
            $testingResults = array_slice($allPredictions, $trainingCount);
        }
        
        // Ambil visualisasi terbaru
        $vizData = $database->fetch("
            SELECT knn_cm_img, dt_cm_img, performance_comparison_img, accuracy_img,
                   train_test_comparison_img, combined_cm_img
            FROM model_visualizations 
            ORDER BY id DESC LIMIT 1
        ");
    }
    
    if ($vizData) {
        $knnCmImg = $vizData['knn_cm_img'];
        $dtCmImg = $vizData['dt_cm_img'];
        $performanceComparisonImg = $vizData['performance_comparison_img'];
        $accuracyImg = $vizData['accuracy_img'];
        $trainTestComparisonImg = $vizData['train_test_comparison_img'];
        $combinedCmImg = $vizData['combined_cm_img'];
    }

    // Hitung statistik untuk training data
    $trainingTotalCount = count($trainingResults);
    $trainingCorrectKNN = 0;
    $trainingCorrectDT = 0;
    $trainingCategoryStats = [];

    foreach ($trainingResults as $row) {
        if ($row['actual'] == $row['knn_pred']) $trainingCorrectKNN++;
        if ($row['actual'] == $row['dt_pred']) $trainingCorrectDT++;
        
        // Hitung statistik per kategori
        $category = $row['actual'];
        if (!isset($trainingCategoryStats[$category])) {
            $trainingCategoryStats[$category] = ['total' => 0, 'knn_correct' => 0, 'dt_correct' => 0];
        }
        $trainingCategoryStats[$category]['total']++;
        if ($row['actual'] == $row['knn_pred']) $trainingCategoryStats[$category]['knn_correct']++;
        if ($row['actual'] == $row['dt_pred']) $trainingCategoryStats[$category]['dt_correct']++;
    }

    $trainingKnnAccuracy = $trainingTotalCount > 0 ? ($trainingCorrectKNN / $trainingTotalCount) * 100 : 0;
    $trainingDtAccuracy = $trainingTotalCount > 0 ? ($trainingCorrectDT / $trainingTotalCount) * 100 : 0;

    // Hitung statistik untuk testing data
    $testingTotalCount = count($testingResults);
    $testingCorrectKNN = 0;
    $testingCorrectDT = 0;
    $testingCategoryStats = [];

    foreach ($testingResults as $row) {
        if ($row['actual'] == $row['knn_pred']) $testingCorrectKNN++;
        if ($row['actual'] == $row['dt_pred']) $testingCorrectDT++;
        
        // Hitung statistik per kategori
        $category = $row['actual'];
        if (!isset($testingCategoryStats[$category])) {
            $testingCategoryStats[$category] = ['total' => 0, 'knn_correct' => 0, 'dt_correct' => 0];
        }
        $testingCategoryStats[$category]['total']++;
        if ($row['actual'] == $row['knn_pred']) $testingCategoryStats[$category]['knn_correct']++;
        if ($row['actual'] == $row['dt_pred']) $testingCategoryStats[$category]['dt_correct']++;
    }

    $testingKnnAccuracy = $testingTotalCount > 0 ? ($testingCorrectKNN / $testingTotalCount) * 100 : 0;
    $testingDtAccuracy = $testingTotalCount > 0 ? ($testingCorrectDT / $testingTotalCount) * 100 : 0;
}

// Gabungkan untuk kompatibilitas dengan kode lama
$allResults = array_merge($trainingResults, $testingResults);
$hasData = !empty($allResults);

// Hitung distribusi kategori
$allCategories = [];

// Kumpulkan semua kategori unik dari training results
foreach ($trainingResults as $row) {
    if (!empty($row['actual']) && !in_array($row['actual'], $allCategories)) {
        $allCategories[] = $row['actual'];
    }
}

// Kumpulkan semua kategori unik dari testing results
foreach ($testingResults as $row) {
    if (!empty($row['actual']) && !in_array($row['actual'], $allCategories)) {
        $allCategories[] = $row['actual'];
    }
}

// Pastikan allCategories adalah array valid
$allCategories = array_values(array_unique(array_filter($allCategories)));

// Debug: tampilkan kategori yang ditemukan
error_log("All categories found: " . print_r($allCategories, true));
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
            max-width: 1200px;
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
        
        /* Tambahan style untuk tabs */
        .nav-tabs {
            border-bottom: 2px solid #dee2e6;
        }
        
        .nav-tabs .nav-link {
            border: none;
            color: #6c757d;
            font-weight: 500;
            padding: 12px 20px;
            border-radius: 8px 8px 0 0;
            margin-right: 5px;
        }
        
        .nav-tabs .nav-link.active {
            background-color: var(--primary-color);
            color: white;
        }
        
        .nav-tabs .nav-link:hover {
            background-color: rgba(67, 97, 238, 0.1);
            color: var(--primary-color);
        }
        
        .nav-tabs .nav-link.active:hover {
            background-color: var(--secondary-color);
            color: white;
        }
        
        .tab-content {
            padding-top: 20px;
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
        
        /* Styles untuk statistik kategori */
        .category-stats {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .category-card {
            border: 2px solid #e9ecef;
            border-radius: 10px;
            transition: all 0.3s;
        }
        
        .category-card:hover {
            border-color: var(--primary-color);
            transform: translateY(-2px);
        }
        
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            transition: all 0.3s;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        /* Chart container styles */
        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }
        
        .chart-small {
            height: 250px;
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
            
            .metric-value {
                font-size: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Tampilan ketika tidak ada data -->
        <div id="noData" class="card fade-in" style="<?php echo (!$hasData) ? '' : 'display: none;'; ?>">
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
        <div id="results" style="<?php echo ($hasData) ? '' : 'display: none;'; ?>">
            <div class="card fade-in">
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
                    <nav aria-label="breadcrumb" class="mb-3 fade-in" style="animation-delay: 0.1s">
                        <ol class="breadcrumb">
                            <li class="breadcrumb-item"><a href="index.php" class="text-decoration-none">Beranda</a></li>
                            <li class="breadcrumb-item active" aria-current="page">Hasil Klasifikasi</li>
                        </ol>
                    </nav>
                    
                    <div class="alert alert-success fade-in" style="animation-delay: 0.2s">
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
                    <div class="alert alert-<?php echo $dbStatus ? 'success' : 'danger'; ?> mb-3 fade-in" style="animation-delay: 0.3s">
                        <i class="bi bi-<?php echo $dbStatus ? 'check-circle' : 'exclamation-triangle'; ?>-fill me-2"></i>
                        Status Database: <?php echo $dbStatus ? 'Terhubung' : 'Tidak Terhubung - Periksa konfigurasi database Anda'; ?>
                    </div>
                    
                    <?php if ($fileInfo): ?>
                    <div class="alert alert-info mb-3 fade-in" style="animation-delay: 0.4s">
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
                    <nav class="navbar navbar-expand-lg navbar-light fade-in" style="animation-delay: 0.5s">
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
                    
                    <!-- Ringkasan Data -->
                    <div class="section-card fade-in" style="animation-delay: 0.6s">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-pie-chart fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Ringkasan Dataset</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-3 mb-3">
                                        <div class="metric-card bg-primary text-white">
                                            <div class="metric-value"><?= $trainingTotalCount + $testingTotalCount ?></div>
                                            <div class="metric-label">Total Data</div>
                                        </div>
                                    </div>
                                    <div class="col-md-3 mb-3">
                                        <div class="metric-card bg-info text-white">
                                            <div class="metric-value"><?= $trainingTotalCount ?></div>
                                            <div class="metric-label">Data Training</div>
                                        </div>
                                    </div>
                                    <div class="col-md-3 mb-3">
                                        <div class="metric-card bg-success text-white">
                                            <div class="metric-value"><?= $testingTotalCount ?></div>
                                            <div class="metric-label">Data Testing</div>
                                        </div>
                                    </div>
                                    <div class="col-md-3 mb-3">
                                        <div class="metric-card bg-warning text-white">
                                            <div class="metric-value"><?= count($allCategories) ?></div>
                                            <div class="metric-label">Kategori</div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Distribusi Data per Kategori -->
                                <div class="mt-4">
                                    <h6 class="fw-bold mb-3">Distribusi Data per Kategori</h6>
                                    <div class="chart-container chart-small">
                                        <canvas id="categoryDistributionChart"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Statistik Training vs Testing -->
                    <div class="section-card fade-in" style="animation-delay: 0.7s">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-graph-up fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Perbandingan Akurasi Training vs Testing</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <!-- Training Stats -->
                                    <div class="col-md-6 mb-4">
                                        <div class="card h-100 border-primary">
                                            <div class="card-header bg-primary text-white">
                                                <h6 class="mb-0"><i class="bi bi-mortarboard me-2"></i>Data Training</h6>
                                            </div>
                                            <div class="card-body">
                                                <div class="row text-center">
                                                    <div class="col-6">
                                                        <h5 class="text-primary"><?= number_format($trainingKnnAccuracy, 2) ?>%</h5>
                                                        <small class="text-muted">KNN</small>
                                                    </div>
                                                    <div class="col-6">
                                                        <h5 class="text-success"><?= number_format($trainingDtAccuracy, 2) ?>%</h5>
                                                        <small class="text-muted">Decision Tree</small>
                                                    </div>
                                                </div>
                                                <hr>
                                                <div class="row">
                                                    <div class="col-6 text-center">
                                                        <small class="text-muted">Benar: <?= $trainingCorrectKNN ?></small>
                                                    </div>
                                                    <div class="col-6 text-center">
                                                        <small class="text-muted">Benar: <?= $trainingCorrectDT ?></small>
                                                    </div>
                                                </div>
                                                <small class="text-muted">Total data: <?= $trainingTotalCount ?> judul</small>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- Testing Stats -->
                                    <div class="col-md-6 mb-4">
                                        <div class="card h-100 border-success">
                                            <div class="card-header bg-success text-white">
                                                <h6 class="mb-0"><i class="bi bi-clipboard-check me-2"></i>Data Testing</h6>
                                            </div>
                                            <div class="card-body">
                                                <div class="row text-center">
                                                    <div class="col-6">
                                                        <h5 class="text-primary"><?= number_format($testingKnnAccuracy, 2) ?>%</h5>
                                                        <small class="text-muted">KNN</small>
                                                    </div>
                                                    <div class="col-6">
                                                        <h5 class="text-success"><?= number_format($testingDtAccuracy, 2) ?>%</h5>
                                                        <small class="text-muted">Decision Tree</small>
                                                    </div>
                                                </div>
                                                <hr>
                                                <div class="row">
                                                    <div class="col-6 text-center">
                                                        <small class="text-muted">Benar: <?= $testingCorrectKNN ?></small>
                                                    </div>
                                                    <div class="col-6 text-center">
                                                        <small class="text-muted">Benar: <?= $testingCorrectDT ?></small>
                                                    </div>
                                                </div>
                                                <small class="text-muted">Total data: <?= $testingTotalCount ?> judul</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Chart Perbandingan Training vs Testing -->
                                <div class="mt-4">
                                    <h6 class="fw-bold mb-3">Grafik Perbandingan Akurasi</h6>
                                    <?php if ($trainTestComparisonImg): ?>
                                        <div class="img-container">
                                            <img src="data:image/png;base64,<?php echo $trainTestComparisonImg; ?>" class="img-fluid" alt="Training vs Testing Comparison">
                                        </div>
                                    <?php else: ?>
                                        <div class="chart-container chart-small">
                                            <canvas id="trainTestComparisonChart"></canvas>
                                        </div>
                                    <?php endif; ?>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Performa per Kategori -->
                    <div class="section-card fade-in" style="animation-delay: 0.8s">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-bullseye fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Performa Model per Kategori</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <!-- Tabs untuk Training dan Testing -->
                                <ul class="nav nav-tabs" id="categoryPerformanceTabs" role="tablist">
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link active" id="training-category-tab" data-bs-toggle="tab" data-bs-target="#training-category" type="button" role="tab">
                                            <i class="bi bi-mortarboard me-2"></i>Training Performance
                                        </button>
                                    </li>
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link" id="testing-category-tab" data-bs-toggle="tab" data-bs-target="#testing-category" type="button" role="tab">
                                            <i class="bi bi-clipboard-check me-2"></i>Testing Performance
                                        </button>
                                    </li>
                                </ul>
                                
                                <div class="tab-content" id="categoryPerformanceTabContent">
                                    <!-- Training Category Performance -->
                                    <div class="tab-pane fade show active" id="training-category" role="tabpanel">
                                        <div class="row mt-4">
                                            <?php foreach ($trainingCategoryStats as $category => $stats): ?>
                                            <div class="col-md-6 col-lg-4 mb-3">
                                                <div class="category-card">
                                                    <div class="card-body">
                                                        <h6 class="card-title text-primary"><?= $category ?></h6>
                                                        <div class="row text-center">
                                                            <div class="col-6">
                                                                <div class="metric-value text-primary" style="font-size: 1.2rem;">
                                                                    <?= $stats['total'] > 0 ? number_format(($stats['knn_correct'] / $stats['total']) * 100, 1) : 0 ?>%
                                                                </div>
                                                                <div class="metric-label">KNN</div>
                                                                <small class="text-muted"><?= $stats['knn_correct'] ?>/<?= $stats['total'] ?></small>
                                                            </div>
                                                            <div class="col-6">
                                                                <div class="metric-value text-success" style="font-size: 1.2rem;">
                                                                    <?= $stats['total'] > 0 ? number_format(($stats['dt_correct'] / $stats['total']) * 100, 1) : 0 ?>%
                                                                </div>
                                                                <div class="metric-label">DT</div>
                                                                <small class="text-muted"><?= $stats['dt_correct'] ?>/<?= $stats['total'] ?></small>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            <?php endforeach; ?>
                                        </div>
                                    </div>
                                    
                                    <!-- Testing Category Performance -->
                                    <div class="tab-pane fade" id="testing-category" role="tabpanel">
                                        <div class="row mt-4">
                                            <?php foreach ($testingCategoryStats as $category => $stats): ?>
                                            <div class="col-md-6 col-lg-4 mb-3">
                                                <div class="category-card">
                                                    <div class="card-body">
                                                        <h6 class="card-title text-primary"><?= $category ?></h6>
                                                        <div class="row text-center">
                                                            <div class="col-6">
                                                                <div class="metric-value text-primary" style="font-size: 1.2rem;">
                                                                    <?= $stats['total'] > 0 ? number_format(($stats['knn_correct'] / $stats['total']) * 100, 1) : 0 ?>%
                                                                </div>
                                                                <div class="metric-label">KNN</div>
                                                                <small class="text-muted"><?= $stats['knn_correct'] ?>/<?= $stats['total'] ?></small>
                                                            </div>
                                                            <div class="col-6">
                                                                <div class="metric-value text-success" style="font-size: 1.2rem;">
                                                                    <?= $stats['total'] > 0 ? number_format(($stats['dt_correct'] / $stats['total']) * 100, 1) : 0 ?>%
                                                                </div>
                                                                <div class="metric-label">DT</div>
                                                                <small class="text-muted"><?= $stats['dt_correct'] ?>/<?= $stats['total'] ?></small>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            <?php endforeach; ?>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Perbandingan Akurasi KNN vs DT -->
                    <div class="section-card fade-in" style="animation-delay: 0.9s">
                        <div class="card border-0 shadow-sm">
                            <div class="card-body p-0">
                                <div class="row g-4">
                                    <div class="col-md-8">
                                        <div class="card h-100">
                                            <div class="card-header bg-white">
                                                <div class="d-flex align-items-center">
                                                    <i class="bi bi-bar-chart-line fs-4 me-2 text-primary"></i>
                                                    <h5 class="mb-0">Perbandingan Akurasi Model</h5>
                                                </div>
                                            </div>
                                            <div class="card-body">
                                                <div class="img-container">
                                                <?php if ($accuracyImg): ?>
                                                    <img src="data:image/png;base64,<?php echo $accuracyImg; ?>" class="img-fluid" alt="Perbandingan Akurasi">
                                                <?php else: ?>
                                                    <canvas id="accuracyChart" height="250"></canvas>
                                                <?php endif; ?>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="card h-100">
                                            <div class="card-header bg-white">
                                                <div class="d-flex align-items-center">
                                                    <i class="bi bi-award fs-4 me-2 text-primary"></i>
                                                    <h5 class="mb-0">Hasil Akurasi Testing</h5>
                                                </div>
                                            </div>
                                            <div class="card-body">
                                                <div class="mt-2">
                                                    <div class="mb-4">
                                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                                            <span class="fw-bold">KNN</span>
                                                            <span id="knnAccuracy" class="badge bg-primary badge-large"><?= number_format($testingKnnAccuracy, 1) ?>%</span>
                                                        </div>
                                                        <div class="progress" style="height: 10px;">
                                                            <div id="knnProgress" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: <?= $testingKnnAccuracy ?>%"></div>
                                                        </div>
                                                    </div>
                                                    <div class="mb-4">
                                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                                            <span class="fw-bold">Decision Tree</span>
                                                            <span id="dtAccuracy" class="badge bg-success badge-large"><?= number_format($testingDtAccuracy, 1) ?>%</span>
                                                        </div>
                                                        <div class="progress" style="height: 10px;">
                                                            <div id="dtProgress" class="progress-bar progress-bar-striped progress-bar-animated bg-success" role="progressbar" style="width: <?= $testingDtAccuracy ?>%"></div>
                                                        </div>
                                                    </div>
                                                </div>
                                                
                                                <div class="mt-4">
                                                    <h6 class="text-muted fw-bold mb-3">Model Terbaik:</h6>
                                                    <div class="alert alert-<?= $testingKnnAccuracy > $testingDtAccuracy ? 'primary' : 'success' ?> py-3">
                                                        <div class="d-flex align-items-center">
                                                            <i class="bi bi-trophy-fill fs-4 me-3"></i>
                                                            <div>
                                                                <strong id="bestModel" class="fs-5"><?= $testingKnnAccuracy > $testingDtAccuracy ? 'KNN' : 'Decision Tree' ?></strong>
                                                                <div class="small">Akurasi: <?= number_format(max($testingKnnAccuracy, $testingDtAccuracy), 2) ?>%</div>
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
                    <div class="section-card fade-in" style="animation-delay: 1.0s">
                        <div class="card border-0 shadow-sm">
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
                                                                    <th>Training</th>
                                                                    <th>Testing</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody>
                                                                <tr>
                                                                    <td>Accuracy</td>
                                                                    <td><?= number_format($trainingKnnAccuracy, 2) ?>%</td>
                                                                    <td><?= number_format($testingKnnAccuracy, 2) ?>%</td>
                                                                </tr>
                                                                <tr>
                                                                    <td>Correct Predictions</td>
                                                                    <td><?= $trainingCorrectKNN ?>/<?= $trainingTotalCount ?></td>
                                                                    <td><?= $testingCorrectKNN ?>/<?= $testingTotalCount ?></td>
                                                                </tr>
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
                                                            Max Depth: <span id="dtDepth">Auto</span>
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
                                                                    <th>Training</th>
                                                                    <th>Testing</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody>
                                                                <tr>
                                                                    <td>Accuracy</td>
                                                                    <td><?= number_format($trainingDtAccuracy, 2) ?>%</td>
                                                                    <td><?= number_format($testingDtAccuracy, 2) ?>%</td>
                                                                </tr>
                                                                <tr>
                                                                    <td>Correct Predictions</td>
                                                                    <td><?= $trainingCorrectDT ?>/<?= $trainingTotalCount ?></td>
                                                                    <td><?= $testingCorrectDT ?>/<?= $testingTotalCount ?></td>
                                                                </tr>
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
                    
                    <!-- Confusion Matrix Section -->
                    <div class="section-card fade-in" style="animation-delay: 1.1s">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-grid-3x3 fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Confusion Matrix</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <?php if ($knnCmImg && $dtCmImg): ?>
                                <div class="row">
                                    <!-- KNN Confusion Matrix -->
                                    <div class="col-md-6 mb-4">
                                        <div class="card h-100">
                                            <div class="card-header bg-white">
                                                <div class="d-flex align-items-center">
                                                    <i class="bi bi-bullseye fs-4 me-2 text-primary"></i>
                                                    <h5 class="mb-0">KNN Confusion Matrix</h5>
                                                </div>
                                            </div>
                                            <div class="card-body text-center">
                                                <div id="knnConfusionMatrix" class="img-container">
                                                    <img src="data:image/png;base64,<?php echo $knnCmImg; ?>" class="img-fluid" alt="KNN Confusion Matrix">
                                                </div>
                                                <p class="text-muted mt-2 small">Confusion matrix menunjukkan jumlah prediksi yang benar dan salah untuk setiap kategori</p>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- Decision Tree Confusion Matrix -->
                                    <div class="col-md-6 mb-4">
                                        <div class="card h-100">
                                            <div class="card-header bg-white">
                                                <div class="d-flex align-items-center">
                                                    <i class="bi bi-diagram-3 fs-4 me-2 text-success"></i>
                                                    <h5 class="mb-0">Decision Tree Confusion Matrix</h5>
                                                </div>
                                            </div>
                                            <div class="card-body text-center">
                                                <div id="dtConfusionMatrix" class="img-container">
                                                    <img src="data:image/png;base64,<?php echo $dtCmImg; ?>" class="img-fluid" alt="Decision Tree Confusion Matrix">
                                                </div>
                                                <p class="text-muted mt-2 small">Visualisasi performa model decision tree dalam memprediksi setiap kategori</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <?php if ($combinedCmImg): ?>
                                <!-- Combined Confusion Matrix -->
                                <div class="mt-4">
                                    <h6 class="fw-bold mb-3">Perbandingan Confusion Matrix (Training vs Testing)</h6>
                                    <div class="img-container">
                                        <img src="data:image/png;base64,<?php echo $combinedCmImg; ?>" class="img-fluid" alt="Combined Confusion Matrix">
                                    </div>
                                </div>
                                <?php endif; ?>
                                
                                <div class="alert alert-info mt-3">
                                    <div class="d-flex">
                                        <i class="bi bi-info-circle-fill fs-4 me-3"></i>
                                        <div>
                                            <strong>Cara Membaca Confusion Matrix</strong>
                                            <p class="mb-0">Baris menunjukkan kategori sebenarnya, kolom menunjukkan kategori yang diprediksi. Angka diagonal (dari kiri atas ke kanan bawah) menunjukkan prediksi yang benar, sedangkan angka lainnya menunjukkan kesalahan klasifikasi.</p>
                                        </div>
                                    </div>
                                </div>
                                <?php else: ?>
                                <div class="alert alert-warning">
                                    <div class="d-flex">
                                        <i class="bi bi-exclamation-triangle-fill fs-4 me-3"></i>
                                        <div>
                                            <strong>Visualisasi tidak tersedia</strong>
                                            <p class="mb-0">Confusion matrix tidak tersedia untuk hasil klasifikasi ini. Coba proses ulang file atau unggah file baru.</p>
                                        </div>
                                    </div>
                                </div>
                                <?php endif; ?>
                            </div>
                        </div>
                    </div>

                    <?php if ($performanceComparisonImg): ?>
                    <!-- Performance Comparison Section -->
                    <div class="section-card fade-in" style="animation-delay: 1.2s">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-bar-chart-line fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Perbandingan Performa Detail</h5>
                                </div>
                            </div>
                            <div class="card-body text-center">
                                <div class="img-container">
                                    <img src="data:image/png;base64,<?php echo $performanceComparisonImg; ?>" class="img-fluid" alt="Performance Comparison">
                                </div>
                                <p class="text-muted mt-2">Perbandingan metrik performa (accuracy, precision, recall, F1-score) antara model KNN dan Decision Tree</p>
                            </div>
                        </div>
                    </div>
                    <?php endif; ?>
                    
                    <!-- Hasil Detail dengan Tabs -->
                    <div class="section-card fade-in" style="animation-delay: 1.3s">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-table fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Detail Hasil Prediksi</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <!-- Tabs Navigation -->
                                <ul class="nav nav-tabs" id="dataTypeTabs" role="tablist">
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link active" id="training-tab" data-bs-toggle="tab" data-bs-target="#training" type="button" role="tab" aria-controls="training" aria-selected="true">
                                            <i class="bi bi-mortarboard me-2"></i>Data Training (<?= $trainingTotalCount ?>)
                                        </button>
                                    </li>
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link" id="testing-tab" data-bs-toggle="tab" data-bs-target="#testing" type="button" role="tab" aria-controls="testing" aria-selected="false">
                                            <i class="bi bi-clipboard-check me-2"></i>Data Testing (<?= $testingTotalCount ?>)
                                        </button>
                                    </li>
                                </ul>
                                
                                <!-- Tab Content -->
                                <div class="tab-content" id="dataTypeTabContent">
                                    <!-- Training Data Tab -->
                                    <div class="tab-pane fade show active" id="training" role="tabpanel" aria-labelledby="training-tab">
                                        <div class="row mb-4">
                                            <div class="col-md-6 mb-3 mb-md-0">
                                                <div class="input-group">
                                                    <span class="input-group-text"><i class="bi bi-search"></i></span>
                                                    <input type="text" id="trainingTableSearch" class="form-control" placeholder="Cari judul skripsi di data training...">
                                                </div>
                                            </div>
                                            <div class="col-md-6 text-md-end">
                                                <div class="btn-group" role="group">
                                                    <button type="button" class="btn btn-outline-primary training-filter-btn active" data-filter="all">Semua</button>
                                                    <button type="button" class="btn btn-outline-success training-filter-btn" data-filter="correct">Benar</button>
                                                    <button type="button" class="btn btn-outline-danger training-filter-btn" data-filter="incorrect">Salah</button>
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div class="table-responsive">
                                            <table class="table table-hover results-table" id="trainingResultsTable">
                                                <thead>
                                                    <tr>
                                                        <th width="5%" class="text-center">No</th>
                                                        <th width="40%">Judul Skripsi</th>
                                                        <th width="15%" class="text-center">Label Sebenarnya</th>
                                                        <th width="15%" class="text-center">Prediksi KNN</th>
                                                        <th width="15%" class="text-center">Prediksi DT</th>
                                                        <th width="10%" class="text-center">Status</th>
                                                    </tr>
                                                </thead>
                                                <tbody id="trainingResultsTableBody">
                                                    <!-- Training results akan ditampilkan di sini -->
                                                </tbody>
                                            </table>
                                        </div>
                                        
                                        <div id="noTrainingResultsMessage" style="display: none;" class="alert alert-warning mt-3">
                                            <div class="d-flex align-items-center">
                                                <i class="bi bi-exclamation-triangle-fill me-3 fs-4"></i>
                                                <div>
                                                    <strong>Tidak ada hasil</strong><br>
                                                    <span>Tidak ada hasil training yang sesuai dengan pencarian.</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- Testing Data Tab -->
                                    <div class="tab-pane fade" id="testing" role="tabpanel" aria-labelledby="testing-tab">
                                        <div class="row mb-4">
                                            <div class="col-md-6 mb-3 mb-md-0">
                                                <div class="input-group">
                                                    <span class="input-group-text"><i class="bi bi-search"></i></span>
                                                    <input type="text" id="testingTableSearch" class="form-control" placeholder="Cari judul skripsi di data testing...">
                                                </div>
                                            </div>
                                            <div class="col-md-6 text-md-end">
                                                <div class="btn-group" role="group">
                                                    <button type="button" class="btn btn-outline-primary testing-filter-btn active" data-filter="all">Semua</button>
                                                    <button type="button" class="btn btn-outline-success testing-filter-btn" data-filter="correct">Benar</button>
                                                    <button type="button" class="btn btn-outline-danger testing-filter-btn" data-filter="incorrect">Salah</button>
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div class="table-responsive">
                                            <table class="table table-hover results-table" id="testingResultsTable">
                                                <thead>
                                                    <tr>
                                                        <th width="5%" class="text-center">No</th>
                                                        <th width="40%">Judul Skripsi</th>
                                                        <th width="15%" class="text-center">Label Sebenarnya</th>
                                                        <th width="15%" class="text-center">Prediksi KNN</th>
                                                        <th width="15%" class="text-center">Prediksi DT</th>
                                                        <th width="10%" class="text-center">Status</th>
                                                    </tr>
                                                </thead>
                                                <tbody id="testingResultsTableBody">
                                                    <!-- Testing results akan ditampilkan di sini -->
                                                </tbody>
                                            </table>
                                        </div>
                                        
                                        <div id="noTestingResultsMessage" style="display: none;" class="alert alert-warning mt-3">
                                            <div class="d-flex align-items-center">
                                                <i class="bi bi-exclamation-triangle-fill me-3 fs-4"></i>
                                                <div>
                                                    <strong>Tidak ada hasil</strong><br>
                                                    <span>Tidak ada hasil testing yang sesuai dengan pencarian.</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Navigation Buttons -->
                    <div class="d-flex justify-content-between mt-4 fade-in" style="animation-delay: 1.4s">
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
                    <small>ANALISIS PERBANDINGAN ALGORITMA K-NEAREST NEIGHBORS (KNN) DAN DECISION TREE BERDASARKAN HASIL SEMATIC SIMILARITY  JUDUL SKRIPSI DAN BIDANG KONSENSTRASI (STUDI KASUS : JURUSAN  PENDIDIKAN TEKNOLOGI INFORMASI DAN KOMUNIKASI) v<?php echo APP_VERSION; ?> | <?php echo date('Y'); ?></small>
                </div>
            </div>
        </div>
        
        <!-- Form Prediksi Judul Baru -->
        <div id="predictionForm" class="card fade-in">
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
                <nav aria-label="breadcrumb" class="mb-3 fade-in" style="animation-delay: 0.1s">
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
                            <div class="d-flex align-items-center">
                                <i class="bi bi-lightbulb fs-4 me-2 text-primary"></i>
                                <h5 class="mb-0">Hasil Prediksi</h5>
                            </div>
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
// Perbaikan untuk script JavaScript di result.php
document.addEventListener('DOMContentLoaded', function() {
    // Ambil data dari PHP dengan validasi
    const trainingResults = <?= json_encode($trainingResults) ?> || [];
    const testingResults = <?= json_encode($testingResults) ?> || [];
    const trainingKnnAccuracy = <?= json_encode($trainingKnnAccuracy) ?> || 0;
    const trainingDtAccuracy = <?= json_encode($trainingDtAccuracy) ?> || 0;
    const testingKnnAccuracy = <?= json_encode($testingKnnAccuracy) ?> || 0;
    const testingDtAccuracy = <?= json_encode($testingDtAccuracy) ?> || 0;
    const trainingCategoryStats = <?= json_encode($trainingCategoryStats) ?> || {};
    const testingCategoryStats = <?= json_encode($testingCategoryStats) ?> || {};
    
    // Perbaikan untuk allCategories - pastikan berupa array
    let allCategories = <?= json_encode($allCategories) ?> || [];
    
    // Validasi dan normalisasi allCategories
    if (!Array.isArray(allCategories)) {
        console.warn('allCategories is not an array, converting...');
        if (typeof allCategories === 'object' && allCategories !== null) {
            allCategories = Object.keys(allCategories);
        } else {
            allCategories = [];
        }
    }
    
    // Jika allCategories masih kosong, buat dari data yang ada
    if (allCategories.length === 0) {
        const categoriesSet = new Set();
        
        // Ambil kategori dari training results
        trainingResults.forEach(row => {
            if (row.actual) categoriesSet.add(row.actual);
        });
        
        // Ambil kategori dari testing results
        testingResults.forEach(row => {
            if (row.actual) categoriesSet.add(row.actual);
        });
        
        allCategories = Array.from(categoriesSet);
    }
    
    console.log('Categories found:', allCategories);
    
    // Buat chart distribusi kategori
    function createCategoryDistributionChart() {
        const ctx = document.getElementById('categoryDistributionChart');
        if (!ctx) {
            console.log('categoryDistributionChart canvas not found');
            return;
        }
        
        // Validasi allCategories sebelum forEach
        if (!Array.isArray(allCategories) || allCategories.length === 0) {
            console.warn('No categories available for chart');
            // Tampilkan pesan di canvas
            const canvasContainer = ctx.parentElement;
            canvasContainer.innerHTML = '<div class="alert alert-warning">Tidak ada data kategori untuk ditampilkan</div>';
            return;
        }
        
        // Hitung distribusi kategori
        const categoryData = {};
        allCategories.forEach(category => {
            const trainingCount = trainingCategoryStats[category] ? trainingCategoryStats[category].total : 0;
            const testingCount = testingCategoryStats[category] ? testingCategoryStats[category].total : 0;
            categoryData[category] = {
                training: trainingCount,
                testing: testingCount,
                total: trainingCount + testingCount
            };
        });
        
        try {
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: allCategories,
                    datasets: [{
                        label: 'Training',
                        data: allCategories.map(cat => categoryData[cat].training),
                        backgroundColor: 'rgba(67, 97, 238, 0.8)',
                        borderColor: 'rgba(67, 97, 238, 1)',
                        borderWidth: 1
                    }, {
                        label: 'Testing',
                        data: allCategories.map(cat => categoryData[cat].testing),
                        backgroundColor: 'rgba(76, 201, 240, 0.8)',
                        borderColor: 'rgba(76, 201, 240, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Distribusi Data per Kategori'
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating category distribution chart:', error);
            const canvasContainer = ctx.parentElement;
            canvasContainer.innerHTML = '<div class="alert alert-danger">Gagal membuat grafik distribusi kategori</div>';
        }
    }
    
    // Buat chart perbandingan training vs testing
    function createTrainTestComparisonChart() {
        const ctx = document.getElementById('trainTestComparisonChart');
        if (!ctx) {
            console.log('trainTestComparisonChart canvas not found');
            return;
        }
        
        try {
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['KNN', 'Decision Tree'],
                    datasets: [{
                        label: 'Training Accuracy (%)',
                        data: [trainingKnnAccuracy, trainingDtAccuracy],
                        backgroundColor: 'rgba(67, 97, 238, 0.8)',
                        borderColor: 'rgba(67, 97, 238, 1)',
                        borderWidth: 2
                    }, {
                        label: 'Testing Accuracy (%)',
                        data: [testingKnnAccuracy, testingDtAccuracy],
                        backgroundColor: 'rgba(40, 167, 69, 0.8)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Perbandingan Akurasi Training vs Testing'
                        },
                        legend: {
                            position: 'top'
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
        } catch (error) {
            console.error('Error creating train test comparison chart:', error);
            const canvasContainer = ctx.parentElement;
            canvasContainer.innerHTML = '<div class="alert alert-danger">Gagal membuat grafik perbandingan training vs testing</div>';
        }
    }
    
    // Buat chart akurasi jika tidak ada gambar
    function createAccuracyChart() {
        const ctx = document.getElementById('accuracyChart');
        if (!ctx) {
            console.log('accuracyChart canvas not found');
            return;
        }
        
        try {
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['KNN', 'Decision Tree'],
                    datasets: [{
                        data: [testingKnnAccuracy, testingDtAccuracy],
                        backgroundColor: [
                            'rgba(67, 97, 238, 0.8)',
                            'rgba(40, 167, 69, 0.8)'
                        ],
                        borderColor: [
                            'rgba(67, 97, 238, 1)',
                            'rgba(40, 167, 69, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Akurasi Testing (%)'
                        },
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating accuracy chart:', error);
            const canvasContainer = ctx.parentElement;
            canvasContainer.innerHTML = '<div class="alert alert-danger">Gagal membuat grafik akurasi</div>';
        }
    }
    
    // Fungsi untuk populate table
    function populateTable(results, tableBodyId, tableId, filterClass, searchId, noResultsId) {
        const tableBody = document.getElementById(tableBodyId);
        if (!tableBody) {
            console.log(`Table body ${tableBodyId} not found`);
            return;
        }
        
        if (!Array.isArray(results) || results.length === 0) {
            console.log(`No results for ${tableBodyId}`);
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">Tidak ada data</td></tr>';
            return;
        }
        
        tableBody.innerHTML = '';
        
        results.forEach((row, index) => {
            if (!row.actual || !row.knn_pred || !row.dt_pred) {
                console.warn('Invalid row data:', row);
                return;
            }
            
            const correctKNN = row.actual === row.knn_pred;
            const correctDT = row.actual === row.dt_pred;
            
            const tr = document.createElement('tr');
            tr.dataset.title = row.title || '';
            tr.dataset.actual = row.actual || '';
            tr.dataset.knn = row.knn_pred || '';
            tr.dataset.dt = row.dt_pred || '';
            
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
            
            // Status badge
            let statusBadge = '';
            if (correctCount === 2) {
                statusBadge = '<span class="badge bg-success">Semua Benar</span>';
            } else if (correctCount === 1) {
                statusBadge = '<span class="badge bg-warning">Sebagian Benar</span>';
            } else {
                statusBadge = '<span class="badge bg-danger">Semua Salah</span>';
            }
            
            tr.innerHTML = `
                <td class="text-center">${index + 1}</td>
                <td>${row.title || 'N/A'}</td>
                <td class="text-center"><span class="badge bg-secondary rounded-pill">${row.actual || 'N/A'}</span></td>
                <td class="text-center"><span class="badge ${knnClass} rounded-pill">${row.knn_pred || 'N/A'}</span></td>
                <td class="text-center"><span class="badge ${dtClass} rounded-pill">${row.dt_pred || 'N/A'}</span></td>
                <td class="text-center">${statusBadge}</td>
            `;
            
            tableBody.appendChild(tr);
        });
        
        // Setup filter dan search untuk tabel ini
        setupTableControls(tableBodyId, filterClass, searchId, noResultsId);
    }
    
    // Fungsi untuk setup kontrol filter dan search
    function setupTableControls(tableBodyId, filterClass, searchId, noResultsId) {
        const filterButtons = document.querySelectorAll(`.${filterClass}`);
        const searchInput = document.getElementById(searchId);
        const tableBody = document.getElementById(tableBodyId);
        
        if (!tableBody) return;
        
        // Fungsi untuk filter dan pencarian
        function filterTable() {
            const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
            const activeFilter = document.querySelector(`.${filterClass}.active`)?.dataset.filter || 'all';
            let visibleCount = 0;
            
            Array.from(tableBody.getElementsByTagName('tr')).forEach(row => {
                if (!row.dataset.title) return; // Skip rows without data
                
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
            const noResultsElement = document.getElementById(noResultsId);
            if (noResultsElement) {
                noResultsElement.style.display = visibleCount > 0 ? 'none' : 'block';
            }
        }
        
        // Aktifkan filter buttons
        filterButtons.forEach(button => {
            button.addEventListener('click', function() {
                // Reset semua button di grup ini
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
    }
    
    // Event handler untuk detail modal
    document.addEventListener('click', function(e) {
        if (e.target.closest('.detail-btn')) {
            const btn = e.target.closest('.detail-btn');
            const title = btn.dataset.title;
            const actual = btn.dataset.actual;
            const knn = btn.dataset.knn;
            const dt = btn.dataset.dt;
            
            document.getElementById('modalTitle').textContent = title;
            document.getElementById('modalCategory').textContent = actual;
            document.getElementById('modalActual').textContent = actual;
            document.getElementById('modalKNN').textContent = knn;
            document.getElementById('modalDT').textContent = dt;
            document.getElementById('modalMainCategory').textContent = actual;
        }
    });
    
    // Initialize charts dengan error handling
    try {
        createCategoryDistributionChart();
    } catch (error) {
        console.error('Failed to create category distribution chart:', error);
    }
    
    try {
        createTrainTestComparisonChart();
    } catch (error) {
        console.error('Failed to create train test comparison chart:', error);
    }
    
    try {
        createAccuracyChart();
    } catch (error) {
        console.error('Failed to create accuracy chart:', error);
    }
    
    // Tampilkan hasil jika ada data
    if ((trainingResults && trainingResults.length > 0) || (testingResults && testingResults.length > 0)) {
        document.getElementById('results').style.display = 'block';
        document.getElementById('noData').style.display = 'none';
        
        // Populate kedua tabel
        if (trainingResults && trainingResults.length > 0) {
            populateTable(trainingResults, 'trainingResultsTableBody', 'trainingResultsTable', 
                        'training-filter-btn', 'trainingTableSearch', 'noTrainingResultsMessage');
        }
        
        if (testingResults && testingResults.length > 0) {
            populateTable(testingResults, 'testingResultsTableBody', 'testingResultsTable', 
                        'testing-filter-btn', 'testingTableSearch', 'noTestingResultsMessage');
        }
    } else {
        document.getElementById('results').style.display = 'none';
        document.getElementById('noData').style.display = 'block';
    }
    
    console.log('Result page initialized successfully');
    console.log('Training results:', trainingResults.length);
    console.log('Testing results:', testingResults.length);
    console.log('Categories:', allCategories.length);
        });
        
        // Global functions for external access
        window.refreshResults = function() {
            location.reload();
        };
        
        window.showPredictionForm = function() {
            document.getElementById('showPredictionBtn').click();
        };
        
        // Error handling for uncaught errors
        window.addEventListener('error', function(e) {
            console.error('Global error:', e.error);
            // Optionally show user-friendly error message
        });
        
        // Handle page visibility change
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'visible') {
                // Page became visible, optionally refresh data
                console.log('Page became visible');
            }
        });
    </script>
</body>
</html>


