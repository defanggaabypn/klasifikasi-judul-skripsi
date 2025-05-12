<?php
// File: web/save_upload.php
// Simpan informasi upload file ke database

require_once 'config.php';
require_once 'database_config.php';

// Terima data JSON
$json = file_get_contents('php://input');
$data = json_decode($json, true);

// Validasi data
if (!$data || !isset($data['filename'])) {
    echo json_encode(['success' => false, 'message' => 'Invalid data']);
    exit;
}

try {
    // Buat koneksi database
    $database = new Database();
    $conn = $database->getConnection();
    
    // Cek jika file sudah diupload (oleh Python API)
    $stmt = $database->fetch("SELECT id FROM uploaded_files WHERE original_filename = ?", [$data['filename']]);
    
    $uploadId = null;
    
    // Jika file belum tercatat, tambahkan ke database
    if (!$stmt) {
        // Simpan info upload
        $uploadId = $database->insert(
            "INSERT INTO uploaded_files (original_filename, filename, file_size, processed) VALUES (?, ?, ?, ?)",
            [$data['filename'], $data['filename'], $data['size'], 1]
        );
        
        // Log aksi upload
        $database->insert(
            "INSERT INTO activity_log (action, details, ip_address) VALUES (?, ?, ?)",
            ['upload', 'File: ' . $data['filename'], $_SERVER['REMOTE_ADDR']]
        );
    } else {
        // Jika file sudah ada, gunakan ID yang ada
        $uploadId = $stmt['id'];
    }
    
    // Berhasil - tambahkan upload_id ke response
    echo json_encode(['success' => true, 'upload_id' => $uploadId]);
} catch (Exception $e) {
    // Gagal
    error_log("Database error in save_upload.php: " . $e->getMessage());
    echo json_encode(['success' => false, 'message' => $e->getMessage()]);
}