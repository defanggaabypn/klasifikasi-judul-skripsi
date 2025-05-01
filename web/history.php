<?php
// File: web/history.php
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

// Ambil daftar file yang sudah diupload
$uploadedFiles = [];
if ($dbStatus) {
    $uploadedFiles = $database->fetchAll("
        SELECT uf.id, uf.original_filename, uf.file_size, uf.upload_date, 
               COUNT(p.id) as prediction_count
        FROM uploaded_files uf
        LEFT JOIN predictions p ON uf.id = p.upload_file_id
        WHERE uf.processed = 1
        GROUP BY uf.id
        ORDER BY uf.upload_date DESC
    ");
}
?>

<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Riwayat Klasifikasi - Sistem Klasifikasi Judul Skripsi</title>
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
        
        .no-data-message {
            text-align: center;
            padding: 40px 20px;
        }
        
        .no-data-icon {
            font-size: 48px;
            color: #ccc;
            margin-bottom: 20px;
        }
        
        .table-responsive {
            border-radius: 10px;
            overflow: hidden;
        }
        
        .table {
            margin-bottom: 0;
        }
        
        .table thead th {
            background-color: rgba(0, 0, 0, 0.03);
            border-bottom: none;
        }
        
        .fade-in {
            opacity: 0;
            animation: fadeIn 0.5s forwards;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .file-card {
            transition: all 0.3s;
            border-radius: 12px;
            overflow: hidden;
        }
        
        .file-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
        
        .file-icon {
            font-size: 2rem;
            margin-bottom: 15px;
        }
        
        .breadcrumb {
            background-color: transparent;
            padding: 10px 0;
        }
        
        .breadcrumb-item a {
            color: var(--primary-color);
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card fade-in">
            <div class="card-header">
                <h2 class="text-center mb-0">Riwayat Klasifikasi Judul Skripsi</h2>
            </div>
            <div class="card-body">
                <!-- Breadcrumb -->
                <nav aria-label="breadcrumb" class="mb-3">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="index.php">Beranda</a></li>
                        <li class="breadcrumb-item active" aria-current="page">Riwayat Klasifikasi</li>
                    </ol>
                </nav>
                
                <!-- Tampilkan pesan jika ada -->
                <?php if (isset($_GET['success']) && isset($_GET['message'])): 
                    $success = $_GET['success'] == '1';
                    $message = urldecode($_GET['message']);
                    $alertClass = $success ? 'alert-success' : 'alert-danger';
                    $icon = $success ? 'check-circle' : 'exclamation-triangle';
                ?>
                <div class="alert <?= $alertClass ?> alert-dismissible fade show mb-3" role="alert">
                    <i class="bi bi-<?= $icon ?>-fill me-2"></i><?= htmlspecialchars($message) ?>
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
                <?php endif; ?>
                
                <!-- Database Status -->
                <div class="alert alert-<?php echo $dbStatus ? 'success' : 'danger'; ?> mb-3 fade-in" style="animation-delay: 0.2s">
                    <i class="bi bi-<?php echo $dbStatus ? 'check-circle' : 'exclamation-triangle'; ?>-fill me-2"></i>
                    Status Database: <?php echo $dbStatus ? 'Terhubung' : 'Tidak Terhubung - Periksa konfigurasi database Anda'; ?>
                </div>
                
                <!-- Menu Navigasi -->
                <nav class="navbar navbar-expand-lg navbar-light bg-light mb-4 fade-in" style="animation-delay: 0.3s">
                    <div class="container-fluid">
                        <span class="navbar-brand">Menu:</span>
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
                                    <a class="nav-link" href="visualisasi.php">
                                        <i class="bi bi-bar-chart"></i> Visualisasi Data
                                    </a>
                                </li>
                                <li class="nav-item">
                                    <a class="nav-link active" href="history.php">
                                        <i class="bi bi-clock-history"></i> Riwayat Klasifikasi
                                    </a>
                                </li>
                            </ul>
                        </div>
                    </div>
                </nav>
                
                <?php if (empty($uploadedFiles)): ?>
                <div class="no-data-message fade-in" style="animation-delay: 0.4s">
                    <div class="no-data-icon">
                        <i class="bi bi-folder"></i>
                    </div>
                    <h4>Belum Ada Data Riwayat</h4>
                    <p class="text-muted">Belum ada file yang diupload dan diproses.</p>
                    <a href="index.php" class="btn btn-primary mt-3">
                        <i class="bi bi-upload me-2"></i>Upload File
                    </a>
                </div>
                <?php else: ?>
                <div class="row fade-in" style="animation-delay: 0.4s">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header bg-white">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-clock-history fs-4 me-2 text-primary"></i>
                                    <h5 class="mb-0">Daftar File yang Sudah Diproses</h5>
                                </div>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-hover">
                                        <thead>
                                            <tr>
                                                <th>#</th>
                                                <th>Nama File</th>
                                                <th>Ukuran</th>
                                                <th>Tanggal Upload</th>
                                                <th>Jumlah Prediksi</th>
                                                <th>Aksi</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <?php foreach ($uploadedFiles as $index => $file): ?>
                                            <tr class="align-middle">
                                                <td><?= $index + 1 ?></td>
                                                <td>
                                                    <div class="d-flex align-items-center">
                                                        <i class="bi bi-file-earmark-excel text-success me-2"></i>
                                                        <?= htmlspecialchars($file['original_filename']) ?>
                                                    </div>
                                                </td>
                                                <td><?= number_format($file['file_size'] / 1024, 2) ?> KB</td>
                                                <td><?= date('d/m/Y H:i', strtotime($file['upload_date'])) ?></td>
                                                <td>
                                                    <span class="badge bg-primary rounded-pill">
                                                        <?= $file['prediction_count'] ?> hasil
                                                    </span>
                                                </td>
                                                <td>
                                                    <div class="btn-group" role="group">
                                                        <a href="result.php?upload_id=<?= $file['id'] ?>" class="btn btn-sm btn-primary">
                                                            <i class="bi bi-eye"></i> Lihat
                                                        </a>
                                                        <a href="export.php?type=excel&id=<?= $file['id'] ?>" class="btn btn-sm btn-success">
                                                            <i class="bi bi-file-excel"></i> Excel
                                                        </a>
                                                        <a href="export.php?type=pdf&id=<?= $file['id'] ?>" class="btn btn-sm btn-danger">
                                                            <i class="bi bi-file-pdf"></i> PDF
                                                        </a>
                                                        <!-- Tombol Delete -->
                                                        <button type="button" class="btn btn-sm btn-danger delete-btn" 
                                                            data-id="<?= $file['id'] ?>" 
                                                            data-filename="<?= htmlspecialchars($file['original_filename']) ?>">
                                                            <i class="bi bi-trash"></i> Hapus
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                            <?php endforeach; ?>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <?php endif; ?>
                
                <!-- Navigation Buttons -->
                <div class="d-flex justify-content-between mt-4 fade-in" style="animation-delay: 0.5s">
                    <a href="index.php" class="btn btn-outline-secondary">
                        <i class="bi bi-arrow-left me-1"></i> Kembali ke Beranda
                    </a>
                    <a href="result.php" class="btn btn-primary">
                        <i class="bi bi-clipboard-data me-1"></i> Lihat Hasil Klasifikasi
                    </a>
                </div>
            </div>
            <div class="card-footer text-center text-muted">
                <small>Sistem Klasifikasi Judul Skripsi v<?php echo APP_VERSION; ?> | <?php echo date('Y'); ?></small>
            </div>
        </div>
    </div>
    
    <!-- Modal Konfirmasi Hapus -->
    <div class="modal fade" id="deleteModal" tabindex="-1" aria-labelledby="deleteModalLabel" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header bg-danger text-white">
                    <h5 class="modal-title" id="deleteModalLabel">Konfirmasi Hapus</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p><i class="bi bi-exclamation-triangle-fill text-danger me-2"></i> Anda yakin ingin menghapus file <strong id="deleteFileName"></strong>?</p>
                    <p class="text-danger mb-0">Semua data prediksi terkait file ini juga akan dihapus. Tindakan ini tidak dapat dibatalkan.</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Batal</button>
                    <a href="#" id="confirmDelete" class="btn btn-danger">Hapus</a>
                </div>
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
            
            // Script untuk modal delete
            const deleteButtons = document.querySelectorAll('.delete-btn');
            
            if (deleteButtons.length > 0) {
                const deleteModal = new bootstrap.Modal(document.getElementById('deleteModal'));
                
                deleteButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        const id = this.getAttribute('data-id');
                        const filename = this.getAttribute('data-filename');
                        
                        // Set nilai ke dalam modal
                        document.getElementById('deleteFileName').textContent = filename;
                        document.getElementById('confirmDelete').href = 'delete_history.php?id=' + id;
                        
                        // Tampilkan modal
                        deleteModal.show();
                    });
                });
            }
        });
    </script>
</body>
</html>