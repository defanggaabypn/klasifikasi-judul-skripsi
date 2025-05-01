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

// Aktifkan error reporting untuk debugging
// error_reporting(E_ALL);
// ini_set('display_errors', 1);

// Ambil data yang akan diekspor
if ($id > 0) {
    // Export berdasarkan ID file yang diupload
    $sql = "
        SELECT p.title, c1.name as actual, c2.name as knn_pred, c3.name as dt_pred, 
               p.confidence, p.prediction_date
        FROM predictions p
        LEFT JOIN categories c1 ON p.actual_category_id = c1.id
        LEFT JOIN categories c2 ON p.knn_prediction_id = c2.id
        LEFT JOIN categories c3 ON p.dt_prediction_id = c3.id
        WHERE p.upload_file_id = ?
        ORDER BY p.id DESC
    ";
    $data = $database->fetchAll($sql, [$id]);
    
    // Ambil info file untuk nama file ekspor
    $fileInfo = $database->fetch("SELECT original_filename FROM uploaded_files WHERE id = ?", [$id]);
    $baseFilename = $fileInfo ? pathinfo($fileInfo['original_filename'], PATHINFO_FILENAME) : 'hasil_klasifikasi';
    
    // Hitung akurasi model untuk file spesifik ini
    $totalCount = count($data);
    $correctKNN = 0;
    $correctDT = 0;
    
    if ($totalCount > 0) {
        foreach ($data as $row) {
            if ($row['actual'] == $row['knn_pred']) $correctKNN++;
            if ($row['actual'] == $row['dt_pred']) $correctDT++;
        }
        
        $knnAccuracy = $totalCount > 0 ? ($correctKNN / $totalCount) : 0;
        $dtAccuracy = $totalCount > 0 ? ($correctDT / $totalCount) : 0;
        
        // Simpan akurasi untuk export ini
        // Buat model_performances entries spesifik untuk file ini jika belum ada
        $existingPerformance = $database->fetchAll(
            "SELECT id FROM model_performances WHERE upload_file_id = ? AND model_name = ?", 
            [$id, 'KNN']
        );
        
        if (empty($existingPerformance)) {
            // Insert data performa KNN
            $database->query(
                "INSERT INTO model_performances (model_name, accuracy, upload_file_id) VALUES (?, ?, ?)",
                ['KNN', $knnAccuracy, $id]
            );
            
            // Insert data performa Decision Tree
            $database->query(
                "INSERT INTO model_performances (model_name, accuracy, upload_file_id) VALUES (?, ?, ?)",
                ['Decision Tree', $dtAccuracy, $id]
            );
        } else {
            // Update data performa yang sudah ada
            $database->query(
                "UPDATE model_performances SET accuracy = ? WHERE model_name = ? AND upload_file_id = ?",
                [$knnAccuracy, 'KNN', $id]
            );
            
            $database->query(
                "UPDATE model_performances SET accuracy = ? WHERE model_name = ? AND upload_file_id = ?",
                [$dtAccuracy, 'Decision Tree', $id]
            );
        }
    }
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
    
    // Untuk ekspor semua data, kita bisa menggunakan akurasi rata-rata dari semua model
    // Ini sudah ditangani oleh class Exporter dengan query-nya sendiri
}

// Jika tidak ada data, tampilkan error
if (empty($data)) {
    header('Content-Type: text/html; charset=utf-8');
    echo '<!DOCTYPE html>
    <html>
    <head>
        <title>Error - Tidak Ada Data</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container mt-5">
            <div class="alert alert-warning">
                <i class="bi bi-exclamation-triangle"></i> Tidak ada data untuk diekspor.
            </div>
            <a href="javascript:history.back()" class="btn btn-primary">Kembali</a>
        </div>
    </body>
    </html>';
    exit;
}

// Format data sesuai dengan struktur yang diharapkan oleh Exporter class
$predictionData = [];
foreach ($data as $index => $row) {
    $predictionData[] = [
        'title' => $row['title'],
        'full_title' => $row['title'], // Gunakan title sebagai full_title juga
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

try {
    // Modifikasi class Exporter untuk menerima upload_file_id juga
    if ($type == 'pdf') {
        $pdfFilename = $filename . '.pdf';
        
        // Kirim upload_file_id ke class Exporter (kalau perlu modifikasi class Exporter juga)
        if ($id > 0) {
            // Modifikasi class Exporter atau implementasikan metode alternatif
            // Untuk sekarang, kita tambahkan variable yang memberi tahu Exporter untuk mengambil data dari model_performances tertentu
            $file = $exporter->exportToPDF($predictionData, $pdfFilename, $id);
        } else {
            $file = $exporter->exportToPDF($predictionData, $pdfFilename);
        }
        
        if (!file_exists($file)) {
            throw new Exception("File PDF tidak berhasil dibuat");
        }
        
        // Kirim file ke browser
        header('Content-Type: application/pdf');
        header('Content-Disposition: attachment; filename="' . $pdfFilename . '"');
        header('Cache-Control: max-age=0');
        
        readfile($file);
        @unlink($file); // Hapus file sementara
    } else {
        $excelFilename = $filename . '.xlsx';
        
        // Kirim upload_file_id ke class Exporter (kalau perlu modifikasi class Exporter juga)
        if ($id > 0) {
            // Modifikasi class Exporter atau implementasikan metode alternatif
            $file = $exporter->exportToExcel($predictionData, $excelFilename, $id);
        } else {
            $file = $exporter->exportToExcel($predictionData, $excelFilename);
        }
        
        if (!file_exists($file)) {
            throw new Exception("File Excel tidak berhasil dibuat");
        }
        
        // Kirim file ke browser
        header('Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');
        header('Content-Disposition: attachment; filename="' . $excelFilename . '"');
        header('Cache-Control: max-age=0');
        
        readfile($file);
        @unlink($file); // Hapus file sementara
    }
} catch (Exception $e) {
    // Log error
    error_log("Export error: " . $e->getMessage());
    
    header('Content-Type: text/html; charset=utf-8');
    echo '<!DOCTYPE html>
    <html>
    <head>
        <title>Error Eksport</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container mt-5">
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Gagal mengekspor data: ' . $e->getMessage() . '
            </div>
            <a href="javascript:history.back()" class="btn btn-primary">Kembali</a>
        </div>
    </body>
    </html>';
    exit;
}

exit;