<?php
// File: web/install.php
// Script untuk instalasi database dan setup awal
require_once 'config.php';

$success = true;
$messages = [];

function addMessage($message, $type = 'info') {
    global $messages;
    $messages[] = ['message' => $message, 'type' => $type];
}

// Fungsi untuk mengeksekusi SQL dari file
function executeSQLFile($filename) {
    try {
        $pdo = getPDO();
        
        // Baca file SQL
        $sql = file_get_contents($filename);
        
        // Split berdasarkan delimiter ;
        $queries = explode(';', $sql);
        
        foreach ($queries as $query) {
            $query = trim($query);
            if (!empty($query)) {
                $pdo->exec($query);
            }
        }
        
        return true;
    } catch (PDOException $e) {
        addMessage('Error mengeksekusi SQL: ' . $e->getMessage(), 'danger');
        return false;
    }
}

// Cek eksistensi folder dan file
if (!is_dir('vendor')) {
    addMessage('Folder vendor tidak ditemukan. Silakan jalankan "composer install" terlebih dahulu.', 'warning');
    $success = false;
}

// Cek koneksi database
try {
    $pdo = new PDO(
        'mysql:host=' . DB_HOST,
        DB_USER,
        DB_PASS,
        [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]
    );
    
    addMessage('Koneksi ke MySQL berhasil.', 'success');
    
    // Cek apakah database sudah ada
    $result = $pdo->query("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '" . DB_NAME . "'");
    
    if ($result->rowCount() > 0) {
        addMessage('Database ' . DB_NAME . ' sudah ada.', 'info');
        
        // Konfirmasi untuk reset database
        $reset = isset($_GET['reset']) && $_GET['reset'] === 'true';
        
        if ($reset) {
            // Drop database
            $pdo->exec("DROP DATABASE IF EXISTS `" . DB_NAME . "`");
            addMessage('Database ' . DB_NAME . ' telah dihapus.', 'warning');
            
            // Buat database baru
            $pdo->exec("CREATE DATABASE `" . DB_NAME . "` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci");
            addMessage('Database ' . DB_NAME . ' telah dibuat ulang.', 'success');
            
            // Install struktur database
            if (file_exists('database.sql')) {
                $success = executeSQLFile('database.sql');
                if ($success) {
                    addMessage('Struktur database berhasil diinstall.', 'success');
                }
            } else {
                addMessage('File database.sql tidak ditemukan.', 'danger');
                $success = false;
            }
        } else {
            addMessage('Database tidak di-reset. Gunakan parameter ?reset=true untuk reset database.', 'info');
        }
    } else {
        // Buat database baru
        $pdo->exec("CREATE DATABASE `" . DB_NAME . "` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci");
        addMessage('Database ' . DB_NAME . ' telah dibuat.', 'success');
        
        // Install struktur database
        if (file_exists('database.sql')) {
            $success = executeSQLFile('database.sql');
            if ($success) {
                addMessage('Struktur database berhasil diinstall.', 'success');
            }
        } else {
            addMessage('File database.sql tidak ditemukan.', 'danger');
            $success = false;
        }
    }
    
} catch (PDOException $e) {
    addMessage('Koneksi database gagal: ' . $e->getMessage(), 'danger');
    $success = false;
}

// Cek folder upload dan struktur lainnya
$folders = [
    '../python-api/uploads',
    '../python-api/models',
    '../python-api/cache',
    'uploads',
    'temp'
];

foreach ($folders as $folder) {
    if (!is_dir($folder)) {
        if (mkdir($folder, 0777, true)) {
            addMessage('Folder ' . $folder . ' berhasil dibuat.', 'success');
        } else {
            addMessage('Gagal membuat folder ' . $folder . '.', 'danger');
            $success = false;
        }
    } else {
        addMessage('Folder ' . $folder . ' sudah ada.', 'info');
    }
}

// Cek file requirements.txt
if (!file_exists('../python-api/requirements.txt')) {
    addMessage('File requirements.txt tidak ditemukan di folder python-api.', 'warning');
}

// Cek app.py
if (!file_exists('../python-api/app.py')) {
    addMessage('File app.py tidak ditemukan di folder python-api.', 'warning');
}
?>

<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Instalasi Sistem Klasifikasi Judul Skripsi</title>
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
        .card {
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: none;
            border-radius: 10px;
            overflow: hidden;
        }
        .log-messages {
            max-height: 400px;
            overflow-y: auto;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #ddd;
            padding: 10px;
        }
        .alert {
            margin-bottom: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h2 class="text-center mb-0">Instalasi Sistem Klasifikasi Judul Skripsi</h2>
            </div>
            <div class="card-body">
                <div class="alert <?= $success ? 'alert-success' : 'alert-danger' ?>">
                    <h4 class="alert-heading">
                        <?php if ($success): ?>
                            <i class="bi bi-check-circle-fill me-2"></i>Instalasi Berhasil!
                        <?php else: ?>
                            <i class="bi bi-exclamation-triangle-fill me-2"></i>Instalasi Gagal!
                        <?php endif; ?>
                    </h4>
                    <p>
                        <?php if ($success): ?>
                            Sistem Klasifikasi Judul Skripsi telah berhasil diinstall. Silakan ikuti langkah-langkah berikut untuk menyelesaikan setup:
                        <?php else: ?>
                            Terdapat beberapa masalah dalam proses instalasi. Silakan perbaiki masalah berikut dan coba lagi:
                        <?php endif; ?>
                    </p>
                </div>
                
                <h5 class="mt-4 mb-3"><i class="bi bi-card-list me-2"></i>Log Instalasi</h5>
                <div class="log-messages mb-4">
                    <?php foreach ($messages as $msg): ?>
                        <div class="alert alert-<?= $msg['type'] ?> py-2 px-3">
                            <?= $msg['message'] ?>
                        </div>
                    <?php endforeach; ?>
                </div>
                
                <?php if ($success): ?>
                    <div class="card border mb-4">
                        <div class="card-header bg-light">
                            <h5 class="mb-0"><i class="bi bi-list-check me-2"></i>Langkah Selanjutnya</h5>
                        </div>
                        <div class="card-body">
                            <ol>
                                <li>Pastikan Anda telah menginstall semua dependensi Python dengan menjalankan <code>pip install -r requirements.txt</code> di folder <code>python-api</code>.</li>
                                <li>Pastikan Anda telah menginstall dependensi PHP dengan menjalankan <code>composer install</code> di folder <code>web</code>.</li>
                                <li>Jalankan server Python dengan menjalankan <code>python app.py</code> di folder <code>python-api</code>.</li>
                                <li>Akses aplikasi melalui web server PHP.</li>
                            </ol>
                        </div>
                    </div>
                <?php else: ?>
                    <div class="card border mb-4 border-danger">
                        <div class="card-header bg-danger text-white">
                            <h5 class="mb-0"><i class="bi bi-exclamation-circle me-2"></i>Tindakan yang Diperlukan</h5>
                        </div>
                        <div class="card-body">
                            <ol>
                                <li>Perbaiki masalah yang tertera pada log instalasi di atas.</li>
                                <li>Pastikan MySQL server berjalan dan kredensial di <code>config.php</code> sudah benar.</li>
                                <li>Pastikan PHP memiliki izin untuk membuat folder dan file.</li>
                                <li>Jalankan script ini kembali setelah memperbaiki masalah.</li>
                            </ol>
                        </div>
                    </div>
                <?php endif; ?>
                
                <div class="d-grid gap-2 d-md-flex justify-content-md-center mt-4">
                    <?php if ($success): ?>
                        <a href="index.php" class="btn btn-primary">
                            <i class="bi bi-house-door me-1"></i> Halaman Utama
                        </a>
                    <?php else: ?>
                        <a href="install.php" class="btn btn-primary">
                            <i class="bi bi-arrow-clockwise me-1"></i> Coba Lagi
                        </a>
                        <a href="install.php?reset=true" class="btn btn-danger">
                            <i class="bi bi-trash me-1"></i> Reset & Install Ulang
                        </a>
                    <?php endif; ?>
                </div>
            </div>
            <div class="card-footer text-center text-muted">
                <small>Sistem Klasifikasi Judul Skripsi v<?php echo APP_VERSION; ?></small>
            </div>
        </div>
    </div>
</body>
</html>