<?php
// File: web/export.php
require_once 'config.php';
require_once 'database_config.php';
require_once 'classes/Exporter.php';

// Inisialisasi database
$database = new Database();
$connection = $database->getConnection();

// Inisialisasi exporter
$exporter = new Exporter($database);

// Cek tipe export
$type = isset($_GET['type']) ? $_GET['type'] : 'excel';
$id = isset($_GET['id']) ? intval($_GET['id']) : 0;

// Ambil data yang akan diekspor
if ($id > 0) {
    // Export berdasarkan ID file yang diupload
    $sql = "
        SELECT p.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred
        FROM predictions p
        LEFT JOIN uploaded_files u ON p.upload_id = u.id
        LEFT JOIN categories c1 ON p.actual_category_id = c1.id
        LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
        LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
        WHERE u.id = ?
        ORDER BY p.id DESC
    ";
    $data = $database->fetchAll($sql, [$id]);
} else {
    // Export semua data terakhir (100 terakhir)
    $sql = "
        SELECT p.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred
        FROM predictions p
        LEFT JOIN categories c1 ON p.actual_category_id = c1.id
        LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
        LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
        ORDER BY p.prediction_date DESC
        LIMIT 100
    ";
    $data = $database->fetchAll($sql);
}

// Format data untuk export
$exportData = [];
foreach ($data as $index => $row) {
    $exportData[] = [
        'title' => $row['title'],
        'full_title' => $row['title'],
        'actual' => $row['actual'],
        'knn_pred' => $row['knn_pred'],
        'dt_pred' => $row['dt_pred']
    ];
}

// Export berdasarkan tipe
if ($type == 'pdf') {
    $filename = 'hasil_klasifikasi_' . date('YmdHis') . '.pdf';
    $file = $exporter->exportToPDF($exportData, $filename);
    
    // Kirim file ke browser
    header('Content-Type: application/pdf');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: max-age=0');
    
    readfile($file);
    unlink($file); // Hapus file sementara
} else {
    $filename = 'hasil_klasifikasi_' . date('YmdHis') . '.xlsx';
    $file = $exporter->exportToExcel($exportData, $filename);
    
    // Kirim file ke browser
    header('Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: max-age=0');
    
    readfile($file);
    unlink($file); // Hapus file sementara
}

exit;