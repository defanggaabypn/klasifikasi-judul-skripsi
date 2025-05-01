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
        SELECT p.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
               p.confidence, p.prediction_date
        FROM predictions p
        LEFT JOIN uploaded_files u ON p.upload_file_id = u.id
        LEFT JOIN categories c1 ON p.actual_category_id = c1.id
        LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
        LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
        WHERE u.id = ?
        ORDER BY p.id DESC
    ";
    $data = $database->fetchAll($sql, [$id]);
    
    // Ambil info file untuk nama file ekspor
    $fileInfo = $database->fetch("SELECT original_filename FROM uploaded_files WHERE id = ?", [$id]);
    $baseFilename = $fileInfo ? pathinfo($fileInfo['original_filename'], PATHINFO_FILENAME) : 'hasil_klasifikasi';
} else {
    // Export semua data terakhir (100 terakhir)
    $sql = "
        SELECT p.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred,
               p.confidence, p.prediction_date
        FROM predictions p
        LEFT JOIN categories c1 ON p.actual_category_id = c1.id
        LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
        LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
        ORDER BY p.prediction_date DESC
        LIMIT 100
    ";
    $data = $database->fetchAll($sql);
    $baseFilename = 'hasil_klasifikasi';
}

// Jika tidak ada data, tampilkan error
if (empty($data)) {
    echo '<div class="alert alert-warning">Tidak ada data untuk diekspor.</div>';
    echo '<p><a href="javascript:history.back()" class="btn btn-primary">Kembali</a></p>';
    exit;
}

// Format data untuk export
$exportData = [];
foreach ($data as $index => $row) {
    $exportData[] = [
        'title' => $row['title'],
        'actual' => $row['actual'] ?: 'N/A',
        'knn_pred' => $row['knn_pred'] ?: 'N/A',
        'dt_pred' => $row['dt_pred'] ?: 'N/A',
        'confidence' => isset($row['confidence']) ? number_format($row['confidence'] * 100, 2) . '%' : 'N/A',
        'date' => isset($row['prediction_date']) ? date('Y-m-d H:i', strtotime($row['prediction_date'])) : 'N/A'
    ];
}

// Tambahkan timestamp ke nama file
$timestamp = date('YmdHis');
$filename = "{$baseFilename}_{$timestamp}";

// Export berdasarkan tipe
if ($type == 'pdf') {
    $pdfFilename = $filename . '.pdf';
    $file = $exporter->exportToPDF($exportData, $pdfFilename);
    
    // Kirim file ke browser
    header('Content-Type: application/pdf');
    header('Content-Disposition: attachment; filename="' . $pdfFilename . '"');
    header('Cache-Control: max-age=0');
    
    readfile($file);
    unlink($file); // Hapus file sementara
} else {
    $excelFilename = $filename . '.xlsx';
    $file = $exporter->exportToExcel($exportData, $excelFilename);
    
    // Kirim file ke browser
    header('Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
    header('Content-Disposition: attachment; filename="' . $excelFilename . '"');
    header('Cache-Control: max-age=0');
    
    readfile($file);
    unlink($file); // Hapus file sementara
}

exit;