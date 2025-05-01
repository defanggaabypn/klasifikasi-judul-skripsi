<?php
// File: web/visualisasi.php
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

// Ambil statistik kategori
$categoryStats = $database->fetchAll("
    SELECT c.name, COUNT(t.id) as count
    FROM categories c
    LEFT JOIN thesis_titles t ON c.id = t.category_id
    GROUP BY c.id
    ORDER BY count DESC
");

// Hitung total
$totalCount = 0;
foreach ($categoryStats as $stat) {
    $totalCount += $stat['count'];
}

// Ambil top keywords per kategori
$keywordStats = [];
$categories = $database->fetchAll("SELECT id, name FROM categories");

foreach ($categories as $category) {
    $keywords = $database->fetchAll("
        SELECT keyword, frequency
        FROM keyword_analysis
        WHERE category_id = ?
        ORDER BY frequency DESC
        LIMIT 10
    ", [$category['id']]);
    
    $keywordStats[$category['name']] = $keywords;
}

// Ambil data performa model
$modelPerformance = $database->fetchAll("
    SELECT model_name, accuracy, training_date
    FROM model_performances
    WHERE model_name IN ('KNN', 'Decision Tree')
    ORDER BY training_date DESC
    LIMIT 10
");
?>

<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualisasi dan Analisis - Sistem Klasifikasi Judul Skripsi</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
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
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        
        .keyword-tag {
            display: inline-block;
            background-color: #f1f1f1;
            border-radius: 20px;
            padding: 5px 15px;
            margin: 5px;
            font-size: 14px;
        }
        
        .keyword-tag-1 { font-size: 22px; background-color: #e3f2fd; }
        .keyword-tag-2 { font-size: 20px; background-color: #e8f5e9; }
        .keyword-tag-3 { font-size: 18px; background-color: #fff3e0; }
        .keyword-tag-4 { font-size: 16px; background-color: #f3e5f5; }
        .keyword-tag-5 { font-size: 14px; background-color: #e1f5fe; }
        
        .tab-content {
            padding: 20px;
        }
        
        .model-progress {
            height: 10px;
            margin-bottom: 10px;
        }
        
        .title-search {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        #similarResults {
            display: none;
        }
        
        .no-data-message {
            text-align: center;
            padding: 30px;
        }
        
        .no-data-icon {
            font-size: 40px;
            color: #6c757d;
            margin-bottom: 15px;
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
        
        .comparison-container {
            border-left: 4px solid #0d6efd;
            padding-left: 15px;
            margin-bottom: 20px;
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
    <div class="container">
        <div class="card fade-in">
            <div class="card-header">
                <h2 class="text-center mb-0">Visualisasi dan Analisis Data</h2>
            </div>
            <div class="card-body">
                <!-- Progress Steps -->
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
                        <li class="breadcrumb-item active" aria-current="page">Visualisasi Data</li>
                    </ol>
                </nav>
                
                <!-- Database Status -->
                <div class="alert alert-<?php echo $dbStatus ? 'success' : 'danger'; ?> mb-3 fade-in" style="animation-delay: 0.2s">
                    <i class="bi bi-<?php echo $dbStatus ? 'check-circle' : 'exclamation-triangle'; ?>-fill me-2"></i>
                    Status Database: <?php echo $dbStatus ? 'Terhubung' : 'Tidak Terhubung - Periksa konfigurasi database Anda'; ?>
                </div>
                
                <!-- Navigasi Menu -->
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
                                    <a class="nav-link" href="result.php">
                                        <i class="bi bi-clipboard-data"></i> Hasil Klasifikasi
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link active" href="visualisasi.php">
                                        <i class="bi bi-bar-chart"></i> Visualisasi Data
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link" href="history.php">
                                        <i class="bi bi-clock-history"></i> Riwayat Klasifikasi
                                    </a>
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
                
                <!-- Navigasi Tab -->
                <ul class="nav nav-tabs fade-in" role="tablist" style="animation-delay: 0.4s">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button" role="tab" aria-controls="overview" aria-selected="true">
                            <i class="bi bi-pie-chart me-1"></i> Overview
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="keywords-tab" data-bs-toggle="tab" data-bs-target="#keywords" type="button" role="tab" aria-controls="keywords" aria-selected="false">
                            <i class="bi bi-tags me-1"></i> Analisis Kata Kunci
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="performance-tab" data-bs-toggle="tab" data-bs-target="#performance" type="button" role="tab" aria-controls="performance" aria-selected="false">
                            <i class="bi bi-graph-up me-1"></i> Performa Model
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="similar-tab" data-bs-toggle="tab" data-bs-target="#similar" type="button" role="tab" aria-controls="similar" aria-selected="false">
                            <i class="bi bi-search me-1"></i> Judul Serupa
                        </button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="comparison-tab" data-bs-toggle="tab" data-bs-target="#comparison" type="button" role="tab" aria-controls="comparison" aria-selected="false">
                            <i class="bi bi-arrow-left-right me-1"></i> Perbandingan Model
                        </button>
                    </li>
                </ul>
                
                <!-- Konten Tab -->
                <div class="tab-content fade-in" style="animation-delay: 0.5s">
                    <!-- Tab Overview -->
                    <div class="tab-pane fade show active" id="overview" role="tabpanel" aria-labelledby="overview-tab">
                        <div class="card border-0 shadow-sm mt-3">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-bar-chart-line fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Distribusi Kategori Judul Skripsi</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <?php if ($totalCount == 0): ?>
                                <div class="no-data-message">
                                    <div class="no-data-icon">
                                        <i class="bi bi-exclamation-circle"></i>
                                    </div>
                                    <h5>Belum Ada Data</h5>
                                    <p>Belum ada data judul skripsi untuk divisualisasikan.</p>
                                    <p>Silakan upload file Excel yang berisi judul skripsi terlebih dahulu.</p>
                                    <a href="index.php" class="btn btn-primary mt-3">
                                        <i class="bi bi-upload me-1"></i> Upload Data
                                    </a>
                                </div>
                                <?php else: ?>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="chart-container">
                                            <canvas id="categoryChart"></canvas>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="table-responsive">
                                            <table class="table table-hover table-bordered">
                                                <thead class="table-primary">
                                                    <tr>
                                                        <th>Kategori</th>
                                                        <th>Jumlah Judul</th>
                                                        <th>Persentase</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    <?php 
                                                    foreach ($categoryStats as $stat): 
                                                        $percentage = ($totalCount > 0) ? ($stat['count'] / $totalCount) * 100 : 0;
                                                    ?>
                                                    <tr>
                                                        <td><?= $stat['name'] ?></td>
                                                        <td><?= $stat['count'] ?></td>
                                                        <td><?= number_format($percentage, 2) ?>%</td>
                                                    </tr>
                                                    <?php endforeach; ?>
                                                </tbody>
                                            </table>
                                        </div>
                                        
                                        <div class="mt-4">
                                            <div class="d-flex justify-content-between align-items-center mb-2">
                                                <h6 class="fw-bold">Ekspor Data</h6>
                                            </div>
                                            <div class="btn-group" role="group">
                                                <a href="export.php?type=excel" class="btn btn-success">
                                                    <i class="bi bi-file-earmark-excel me-1"></i> Ekspor Excel
                                                </a>
                                                <a href="export.php?type=pdf" class="btn btn-danger">
                                                    <i class="bi bi-file-earmark-pdf me-1"></i> Ekspor PDF
                                                </a>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <?php endif; ?>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Tab Analisis Kata Kunci -->
                    <div class="tab-pane fade" id="keywords" role="tabpanel" aria-labelledby="keywords-tab">
                        <div class="card border-0 shadow-sm mt-3">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-tags fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Kata Kunci Populer per Kategori</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <?php if (empty($keywordStats) || array_sum(array_map(function($keywords) { return count($keywords); }, $keywordStats)) == 0): ?>
                                <div class="no-data-message">
                                    <div class="no-data-icon">
                                        <i class="bi bi-tag"></i>
                                    </div>
                                    <h5>Belum Ada Analisis Kata Kunci</h5>
                                    <p>Belum ada analisis kata kunci untuk ditampilkan.</p>
                                    <p>Silakan proses data judul skripsi terlebih dahulu untuk melihat analisis kata kunci.</p>
                                    <a href="index.php" class="btn btn-primary mt-3">
                                        <i class="bi bi-upload me-1"></i> Upload Data
                                    </a>
                                </div>
                                <?php else: ?>
                                <div class="row">
                                    <?php foreach ($keywordStats as $category => $keywords): ?>
                                    <?php if (!empty($keywords)): ?>
                                    <div class="col-md-4 mb-4">
                                        <div class="card h-100">
                                            <div class="card-header bg-light">
                                                <h5 class="mb-0"><?= $category ?></h5>
                                            </div>
                                            <div class="card-body">
                                                <div class="keyword-cloud">
                                                    <?php 
                                                    $count = 1;
                                                    foreach ($keywords as $keyword): 
                                                        $tagClass = 'keyword-tag-' . min(5, ceil($count / 2));
                                                    ?>
                                                    <span class="keyword-tag <?= $tagClass ?>" title="Frekuensi: <?= $keyword['frequency'] ?>">
                                                        <?= $keyword['keyword'] ?>
                                                    </span>
                                                    <?php 
                                                        $count++;
                                                    endforeach; 
                                                    ?>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <?php endif; ?>
                                    <?php endforeach; ?>
                                </div>
                                <?php endif; ?>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Tab Performa Model -->
                    <div class="tab-pane fade" id="performance" role="tabpanel" aria-labelledby="performance-tab">
                        <div class="card border-0 shadow-sm mt-3">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-graph-up fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Performa Model Klasifikasi</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <?php if (empty($modelPerformance)): ?>
                                <div class="no-data-message">
                                    <div class="no-data-icon">
                                        <i class="bi bi-graph-up"></i>
                                    </div>
                                    <h5>Belum Ada Data Performa Model</h5>
                                    <p>Belum ada data performa model untuk ditampilkan.</p>
                                    <p>Silakan latih model dengan data judul skripsi terlebih dahulu.</p>
                                    <a href="index.php" class="btn btn-primary mt-3">
                                        <i class="bi bi-upload me-1"></i> Upload Data
                                    </a>
                                </div>
                                <?php else: ?>
                                <div class="row">
                                    <div class="col-md-7">
                                        <div class="chart-container">
                                            <canvas id="performanceChart"></canvas>
                                        </div>
                                    </div>
                                    <div class="col-md-5">
                                        <div class="card">
                                            <div class="card-header bg-light">
                                                <h5 class="mb-0">Detail Performa Model</h5>
                                            </div>
                                            <div class="card-body">
                                                <?php 
                                                $latestPerformance = [];
                                                foreach ($modelPerformance as $perf) {
                                                    if (!isset($latestPerformance[$perf['model_name']])) {
                                                        $latestPerformance[$perf['model_name']] = $perf;
                                                    }
                                                }
                                                
                                                foreach ($latestPerformance as $model => $perf): 
                                                    $percentage = $perf['accuracy'] * 100;
                                                    $bgClass = $percentage >= 80 ? 'bg-success' : ($percentage >= 60 ? 'bg-warning' : 'bg-danger');
                                                    $colorClass = $model == 'KNN' ? 'text-primary' : 'text-success';
                                                ?>
                                                <div class="mb-3">
                                                    <div class="d-flex justify-content-between">
                                                        <span class="<?= $colorClass ?>"><?= $model ?></span>
                                                        <span class="badge bg-primary"><?= number_format($percentage, 2) ?>%</span>
                                                    </div>
                                                    <div class="progress model-progress">
                                                        <div class="progress-bar <?= $bgClass ?>" role="progressbar" style="width: <?= $percentage ?>%" aria-valuenow="<?= $percentage ?>" aria-valuemin="0" aria-valuemax="100"></div>
                                                    </div>
                                                    <small class="text-muted">Diperbarui: <?= date('d/m/Y H:i', strtotime($perf['training_date'])) ?></small>
                                                </div>
                                                <?php endforeach; ?>
                                                
                                                <div class="alert alert-info mt-3 mb-0">
                                                    <i class="bi bi-info-circle me-2"></i>
                                                    <small>Performa model diukur berdasarkan akurasi pada data pengujian.</small>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <?php endif; ?>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Tab Judul Serupa -->
                    <div class="tab-pane fade" id="similar" role="tabpanel" aria-labelledby="similar-tab">
                        <div class="card border-0 shadow-sm mt-3">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-search fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Cari Judul Skripsi Serupa</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <?php if ($totalCount == 0): ?>
                                <div class="alert alert-warning">
                                    <i class="bi bi-exclamation-triangle me-2"></i>
                                    <strong>Perhatian:</strong> Pencarian judul serupa membutuhkan data judul skripsi yang sudah tersimpan dalam database. Silakan upload data terlebih dahulu.
                                </div>
                                <?php endif; ?>
                                
                                <div class="title-search">
                                    <form id="similarTitleForm">
                                        <div class="mb-3">
                                            <label for="searchTitle" class="form-label fw-bold">Masukkan Judul Skripsi:</label>
                                            <textarea id="searchTitle" class="form-control" rows="3" required placeholder="Contoh: Sistem Informasi Manajemen Perpustakaan Berbasis Web"></textarea>
                                        </div>
                                        <div class="row">
                                            <div class="col-md-4">
                                                <div class="mb-3">
                                                    <label for="limitResults" class="form-label">Jumlah Hasil:</label>
                                                    <select id="limitResults" class="form-select">
                                                        <option value="5" selected>5 Judul</option>
                                                        <option value="10">10 Judul</option>
                                                        <option value="15">15 Judul</option>
                                                    </select>
                                                </div>
                                            </div>
                                            <div class="col-md-8">
                                                <div class="d-grid gap-2 d-md-flex justify-content-md-end mt-4">
                                                    <button type="submit" class="btn btn-primary" <?= $totalCount == 0 ? 'disabled' : '' ?>>
                                                        <i class="bi bi-search me-1"></i> Cari Judul Serupa
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </form>
                                </div>
                                
                                <div id="loadingSearch" class="text-center my-5" style="display: none;">
                                    <div class="spinner-border text-primary" role="status"></div>
                                    <p class="mt-2">Sedang mencari judul serupa...</p>
                                    <p class="text-muted small">Proses ini mungkin membutuhkan waktu beberapa detik karena menggunakan IndoBERT</p>
                                </div>
                                
                                <div id="similarResults">
                                    <div class="alert alert-info">
                                        <div class="d-flex align-items-center">
                                            <i class="bi bi-info-circle-fill me-2 fs-4"></i>
                                            <div>
                                                <strong>Hasil Pencarian Judul Serupa</strong>
                                                <p class="mb-0" id="similarTitle">Judul yang dicari: </p>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="table-responsive">
                                        <table class="table table-hover table-bordered">
                                            <thead class="table-primary">
                                                <tr>
                                                    <th width="5%">No</th>
                                                    <th width="55%">Judul Skripsi</th>
                                                    <th width="20%">Kategori</th>
                                                    <th width="20%">Kemiripan</th>
                                                </tr>
                                            </thead>
                                            <tbody id="similarTableBody">
                                                <!-- Hasil pencarian akan ditampilkan di sini -->
                                            </tbody>
                                        </table>
                                    </div>
                                    
                                    <div class="mt-4">
                                        <div class="d-flex align-items-center">
                                            <div class="me-3">
                                                <span class="me-2">Prediksi Kategori:</span>
                                                <span id="predictedCategory" class="badge bg-primary fs-6"></span>
                                            </div>
                                            <button id="newSearchBtn" class="btn btn-outline-primary ms-auto">
                                                <i class="bi bi-search me-1"></i> Pencarian Baru
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Tab Perbandingan Model -->
                    <div class="tab-pane fade" id="comparison" role="tabpanel" aria-labelledby="comparison-tab">
                        <div class="card border-0 shadow-sm mt-3">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-arrow-left-right fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Perbandingan KNN dan Decision Tree</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-12">
                                        <div class="card mb-4">
                                            <div class="card-header bg-light">
                                                <h5 class="mb-0">Karakteristik Model</h5>
                                            </div>
                                            <div class="card-body">
                                                <div class="row">
                                                    <div class="col-md-6">
                                                        <div class="comparison-container">
                                                            <h5 class="text-primary"><i class="bi bi-bullseye me-2"></i>KNN (K-Nearest Neighbors)</h5>
                                                            <ul class="mt-3">
                                                                <li>Algoritma <strong>berbasis contoh/instance</strong> yang melakukan klasifikasi berdasarkan jarak</li>
                                                                <li>Prediksi menggunakan <strong>voting dari k tetangga terdekat</strong> dalam ruang fitur</li>
                                                                <li>Kelebihan: Sederhana, mudah diimplementasikan, tidak perlu training eksplisit</li>
                                                                <li>Kekurangan: Komputasi berat saat prediksi, sensitif terhadap fitur yang tidak relevan</li>
                                                                <li>Parameter utama: Jumlah tetangga (k), metrik jarak</li>
                                                            </ul>
                                                        </div>
                                                    </div>
                                                    <div class="col-md-6">
                                                        <div class="comparison-container" style="border-left-color: #198754;">
                                                            <h5 class="text-success"><i class="bi bi-diagram-3 me-2"></i>Decision Tree</h5>
                                                            <ul class="mt-3">
                                                                <li>Algoritma <strong>berbasis aturan</strong> yang membagi data dengan partisi rekursif</li>
                                                                <li>Prediksi menggunakan <strong>serangkaian keputusan biner</strong> dari akar ke daun</li>
                                                                <li>Kelebihan: Interpretabilitas tinggi, dapat menangani data campuran</li>
                                                                <li>Kekurangan: Rentan overfitting, kurang stabil terhadap variasi data</li>
                                                                <li>Parameter utama: Kedalaman pohon, kriteria pemisahan (gini, entropy)</li>
                                                            </ul>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div class="card mb-4">
                                            <div class="card-header bg-light">
                                                <h5 class="mb-0">Perbandingan Performa pada Klasifikasi Judul Skripsi</h5>
                                            </div>
                                            <div class="card-body">
                                                <div class="table-responsive">
                                                    <table class="table table-bordered">
                                                        <thead class="table-light">
                                                            <tr>
                                                                <th>Aspek</th>
                                                                <th class="text-primary">KNN</th>
                                                                <th class="text-success">Decision Tree</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td>Kecepatan Training</td>
                                                                <td>Sangat cepat (hanya menyimpan data)</td>
                                                                <td>Cepat (membangun pohon keputusan)</td>
                                                            </tr>
                                                            <tr>
                                                                <td>Kecepatan Prediksi</td>
                                                                <td>Lambat (harus menghitung jarak ke semua data training)</td>
                                                                <td>Sangat cepat (hanya mengikuti jalur pohon)</td>
                                                            </tr>
                                                            <tr>
                                                                <td>Interpretabilitas</td>
                                                                <td>Rendah (black box, sulit dijelaskan)</td>
                                                                <td>Tinggi (aturan keputusan jelas dan bisa divisualisasikan)</td>
                                                            </tr>
                                                            <tr>
                                                                <td>Kemampuan Menangani Noise</td>
                                                                <td>Sedang (dapat dipengaruhi oleh outlier)</td>
                                                                <td>Rentan terhadap noise (dapat menyebabkan cabang yang tidak perlu)</td>
                                                            </tr>
                                                            <tr>
                                                                <td>Performa dengan Judul Panjang</td>
                                                                <td>Baik (jarak Euclidean pada embedding mampu menangkap kemiripan semantik)</td>
                                                                <td>Sedang (hanya menggunakan fitur-fitur embedding yang diskriminatif)</td>
                                                            </tr>
                                                            <tr>
                                                                <td>Kebutuhan Data</td>
                                                                <td>Membutuhkan data training yang cukup besar dan representatif</td>
                                                                <td>Dapat bekerja dengan dataset yang lebih kecil</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                                
                                                <div class="alert alert-info mt-4">
                                                    <i class="bi bi-lightbulb me-2"></i>
                                                    <strong>Insight:</strong> Dengan fitur-fitur embedding IndoBERT, KNN dapat mengungguli Decision Tree dalam klasifikasi judul karena kemampuannya menangkap kemiripan semantik melalui jarak, namun Decision Tree memberikan interpretabilitas yang lebih baik tentang fitur mana yang paling penting dalam klasifikasi.
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div class="card">
                                            <div class="card-header bg-light">
                                                <h5 class="mb-0">Rekomendasi Penggunaan</h5>
                                            </div>
                                            <div class="card-body">
                                                <div class="row">
                                                    <div class="col-md-6">
                                                        <div class="alert alert-primary">
                                                            <h6 class="alert-heading"><i class="bi bi-bullseye me-2"></i>Gunakan KNN Ketika:</h6>
                                                            <ul class="mb-0 mt-2">
                                                                <li>Mencari judul-judul yang secara semantik mirip</li>
                                                                <li>Kemiripan antar judul lebih penting daripada aturan spesifik</li>
                                                                <li>Dataset relatif bersih dari noise dan outlier</li>
                                                                <li>Kecepatan prediksi bukan prioritas utama</li>
                                                            </ul>
                                                        </div>
                                                    </div>
                                                    <div class="col-md-6">
                                                        <div class="alert alert-success">
                                                            <h6 class="alert-heading"><i class="bi bi-diagram-3 me-2"></i>Gunakan Decision Tree Ketika:</h6>
                                                            <ul class="mb-0 mt-2">
                                                                <li>Membutuhkan penjelasan mengapa judul diklasifikasikan ke kategori tertentu</li>
                                                                <li>Kecepatan prediksi sangat penting</li>
                                                                <li>Ingin mengidentifikasi fitur paling penting dalam klasifikasi</li>
                                                                <li>Akan digunakan untuk membuat aturan manual</li>
                                                            </ul>
                                                        </div>
                                                    </div>
                                                </div>
                                                
                                                <div class="mt-4">
                                                    <h6 class="fw-bold">Pendekatan Ensembel:</h6>
                                                    <p>Untuk hasil terbaik, pertimbangkan menggunakan pendekatan <strong>voting ensembel</strong> dari kedua model. Ketika KNN dan Decision Tree memberikan prediksi yang sama, tingkat kepercayaan terhadap hasil klasifikasi biasanya lebih tinggi.</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Navigation Buttons -->
                <div class="d-flex justify-content-between mt-4 fade-in" style="animation-delay: 0.8s">
                    <a href="result.php" class="btn btn-outline-secondary">
                        <i class="bi bi-arrow-left me-1"></i> Kembali ke Hasil
                    </a>
                    <a href="index.php" class="btn btn-primary">
                        <i class="bi bi-house-door me-1"></i> Kembali ke Beranda
                    </a>
                </div>
            </div>
            <div class="card-footer text-center text-muted">
                <small>ANALISIS PERBANDINGAN ALGORITMA K-NEAREST NEIGHBORS (KNN) DAN DECISION TREE BERDASARKAN HASIL SEMATIC SIMILARITY  JUDUL SKRIPSI DAN BIDANG KONSENSTRASI (STUDI KASUS : JURUSAN  PENDIDIKAN TEKNOLOGI INFORMASI DAN KOMUNIKASI) v<?php echo APP_VERSION; ?> | <?php echo date('Y'); ?></small>
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
            
            // Chart.js Registration
            Chart.register(ChartDataLabels);
            
            <?php if ($totalCount > 0): ?>
            // Data untuk chart
            const categoryData = {
                labels: <?= json_encode(array_column($categoryStats, 'name')) ?>,
                counts: <?= json_encode(array_column($categoryStats, 'count')) ?>
            };
            
            // Render category chart
            const categoryCtx = document.getElementById('categoryChart').getContext('2d');
            new Chart(categoryCtx, {
                type: 'pie',
                data: {
                    labels: categoryData.labels,
                    datasets: [{
                        data: categoryData.counts,
                        backgroundColor: [
                            'rgba(54, 162, 235, 0.7)',
                            'rgba(255, 99, 132, 0.7)',
                            'rgba(75, 192, 192, 0.7)',
                            'rgba(255, 206, 86, 0.7)',
                            'rgba(153, 102, 255, 0.7)'
                        ],
                        borderColor: [
                            'rgba(54, 162, 235, 1)',
                            'rgba(255, 99, 132, 1)',
                            'rgba(75, 192, 192, 1)',
                            'rgba(255, 206, 86, 1)',
                            'rgba(153, 102, 255, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right',
                        },
                        title: {
                            display: true,
                            text: 'Distribusi Judul Skripsi per Kategori'
                        },
                        datalabels: {
                            formatter: (value, ctx) => {
                                const total = ctx.dataset.data.reduce((acc, data) => acc + data, 0);
                                const percentage = (total > 0) ? ((value * 100) / total).toFixed(1) + '%' : '0%';
                                return value + '\n' + percentage;
                            },
                            color: '#fff',
                            font: {
                                weight: 'bold'
                            }
                        }
                    }
                }
            });
            <?php endif; ?>
            
            <?php if (!empty($modelPerformance)): ?>
            const performanceData = {
                models: [],
                accuracies: [],
                dates: []
            };
            
            <?php foreach ($modelPerformance as $perf): ?>
                performanceData.models.push('<?= $perf['model_name'] ?>');
                performanceData.accuracies.push(<?= $perf['accuracy'] * 100 ?>);
                performanceData.dates.push('<?= date('d/m/Y', strtotime($perf['training_date'])) ?>');
            <?php endforeach; ?>
            
            // Render performance chart
            const perfCtx = document.getElementById('performanceChart').getContext('2d');
            new Chart(perfCtx, {
                type: 'bar',
                data: {
                    labels: performanceData.dates,
                    datasets: [{
                        label: 'KNN',
                        data: performanceData.accuracies.filter((_, i) => performanceData.models[i] === 'KNN'),
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Decision Tree',
                        data: performanceData.accuracies.filter((_, i) => performanceData.models[i] === 'Decision Tree'),
                        backgroundColor: 'rgba(75, 192, 192, 0.7)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
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
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Performa Model Klasifikasi'
                        },
                        datalabels: {
                            formatter: (value) => value.toFixed(1) + '%',
                            color: '#fff',
                            font: {
                                weight: 'bold'
                            }
                        }
                    }
                }
            });
            <?php endif; ?>
            
            // Handler untuk form pencarian judul serupa
            const similarTitleForm = document.getElementById('similarTitleForm');
            const searchTitle = document.getElementById('searchTitle');
            const limitResults = document.getElementById('limitResults');
            const loadingSearch = document.getElementById('loadingSearch');
            const similarResults = document.getElementById('similarResults');
            const similarTableBody = document.getElementById('similarTableBody');
            const similarTitle = document.getElementById('similarTitle');
            const predictedCategory = document.getElementById('predictedCategory');
            const newSearchBtn = document.getElementById('newSearchBtn');
            
            // Hide similar results initially
            similarResults.style.display = 'none';
            
            if (similarTitleForm) {
                similarTitleForm.addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    const title = searchTitle.value.trim();
                    const limit = limitResults.value;
                    
                    if (!title) {
                        alert('Silakan masukkan judul skripsi terlebih dahulu!');
                        return;
                    }
                    
                    // Tampilkan loading
                    loadingSearch.style.display = 'block';
                    similarResults.style.display = 'none';
                    
                    // Kirim request ke API
                    fetch('<?php echo API_URL; ?>/similar', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ title: title, limit: parseInt(limit) })
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        // Sembunyikan loading
                        loadingSearch.style.display = 'none';
                        
                        // Tampilkan hasil
                        similarTitle.textContent = 'Judul yang dicari: ' + data.query;
                        
                        // Kosongkan tabel
                        similarTableBody.innerHTML = '';
                        
                        // Isi tabel dengan hasil
                        if (data.similar_titles && data.similar_titles.length > 0) {
                            const similarTitles = data.similar_titles;
                            
                            // Prediksi kategori berdasarkan judul terdekat
                            const categories = {};
                            let maxCount = 0;
                            let predictedCat = '';
                            
                            similarTitles.forEach((item, index) => {
                                // Hitung frekuensi kategori
                                categories[item.category] = (categories[item.category] || 0) + 1;
                                if (categories[item.category] > maxCount) {
                                    maxCount = categories[item.category];
                                    predictedCat = item.category;
                                }
                                
                                // Format similarity
                                const similarity = (item.similarity * 100).toFixed(2) + '%';
                                const similarityClass = item.similarity >= 0.7 ? 'bg-success' : 
                                                        (item.similarity >= 0.5 ? 'bg-warning' : 'bg-danger');
                                
                                // Tambahkan ke tabel
                                const tr = document.createElement('tr');
                                tr.innerHTML = `
                                    <td>${index + 1}</td>
                                    <td>${item.title}</td>
                                    <td><span class="badge bg-primary">${item.category}</span></td>
                                    <td>
                                        <div class="progress">
                                            <div class="progress-bar ${similarityClass}" role="progressbar" 
                                                style="width: ${item.similarity * 100}%" 
                                                aria-valuenow="${item.similarity * 100}" 
                                                aria-valuemin="0" aria-valuemax="100">
                                                ${similarity}
                                            </div>
                                        </div>
                                    </td>
                                `;
                                
                                similarTableBody.appendChild(tr);
                            });
                            
                            // Tampilkan prediksi kategori
                            predictedCategory.textContent = data.predicted_category || predictedCat;
                        } else {
                            // Tidak ada hasil
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td colspan="4" class="text-center">Tidak ditemukan judul serupa</td>
                            `;
                            similarTableBody.appendChild(tr);
                            
                            predictedCategory.textContent = 'Tidak dapat diprediksi';
                        }
                        
                        // Tampilkan hasil
                        similarResults.style.display = 'block';
                    })
                    .catch(error => {
                        // Sembunyikan loading
                        loadingSearch.style.display = 'none';
                        
                        console.error('Error:', error);
                        alert('Terjadi kesalahan saat mencari judul serupa. Silakan coba lagi.');
                    });
                });
            }
            
            // Handler untuk tombol pencarian baru
            if (newSearchBtn) {
                newSearchBtn.addEventListener('click', function() {
                    searchTitle.value = '';
                    similarResults.style.display = 'none';
                    searchTitle.focus();
                });
            }
        });
    </script>
</body>
</html>