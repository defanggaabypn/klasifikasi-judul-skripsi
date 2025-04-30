<?php
// Konfigurasi dasar
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Zona waktu
date_default_timezone_set('Asia/Jakarta');

// Session
session_start();

// URL API Python
define('API_URL', 'http://localhost:5000');

// Ukuran file maksimum (dalam MB)
define('MAX_UPLOAD_SIZE', 10);

// Direktori aplikasi
define('APP_DIR', dirname(__FILE__));

// Versi aplikasi
define('APP_VERSION', '1.0.0');

// Kategori skripsi
define('CATEGORIES', ['RPL', 'Jaringan', 'Multimedia']);

// Konfigurasi database
define('DB_HOST', 'localhost');
define('DB_NAME', 'skripsi_classification');
define('DB_USER', 'root');
define('DB_PASS', '');

// Fungsi untuk menampilkan pesan error/sukses
function showAlert($message, $type = 'danger') {
    echo '<div class="alert alert-' . $type . ' alert-dismissible fade show" role="alert">';
    echo $message;
    echo '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>';
    echo '</div>';
}

// Fungsi untuk validasi file Excel
function validateExcelFile($file) {
    $errors = [];
    
    // Cek apakah file ada
    if (!isset($file) || $file['error'] != UPLOAD_ERR_OK) {
        $errors[] = 'File upload gagal atau tidak ada. Error code: ' . $file['error'];
        return $errors;
    }
    
    // Cek ekstensi file
    $fileExt = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
    if ($fileExt != 'xlsx') {
        $errors[] = 'Hanya file Excel (.xlsx) yang diperbolehkan';
    }
    
    // Cek ukuran file (max 10MB)
    $maxFileSize = MAX_UPLOAD_SIZE * 1024 * 1024; // MB to bytes
    if ($file['size'] > $maxFileSize) {
        $errors[] = 'Ukuran file terlalu besar (maksimal ' . MAX_UPLOAD_SIZE . 'MB)';
    }
    
    return $errors;
}

// Fungsi untuk format angka jadi persen
function formatPercent($decimal, $decimals = 2) {
    return number_format($decimal * 100, $decimals) . '%';
}

// Fungsi untuk debugging
function debug($var, $die = false) {
    echo '<pre>';
    print_r($var);
    echo '</pre>';
    if ($die) die();
}

// Fungsi untuk decode base64 dan download file
function downloadBase64File($base64Data, $filename) {
    $data = base64_decode($base64Data);
    header('Content-Description: File Transfer');
    header('Content-Type: application/octet-stream');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Expires: 0');
    header('Cache-Control: must-revalidate');
    header('Pragma: public');
    header('Content-Length: ' . strlen($data));
    echo $data;
    exit;
}

// Fungsi untuk koneksi ke database menggunakan PDO
function getPDO() {
    static $pdo;
    
    if (!$pdo) {
        try {
            $pdo = new PDO(
                'mysql:host=' . DB_HOST . ';dbname=' . DB_NAME . ';charset=utf8mb4',
                DB_USER,
                DB_PASS,
                [
                    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                    PDO::ATTR_EMULATE_PREPARES => false
                ]
            );
        } catch (PDOException $e) {
            die('Koneksi database gagal: ' . $e->getMessage());
        }
    }
    
    return $pdo;
}