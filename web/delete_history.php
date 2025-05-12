<?php
// File: web/delete_history.php
require_once 'config.php';
require_once 'database_config.php';

// Inisialisasi database
$database = new Database();
$connection = $database->getConnection();

// Cek apakah ID ada
if (isset($_GET['id']) && !empty($_GET['id'])) {
    $id = intval($_GET['id']);
    $success = false;
    $message = '';
    
    try {
        // Cek keberadaan file
        $fileInfo = $database->fetch("SELECT id, filename FROM uploaded_files WHERE id = ?", [$id]);
        
        if ($fileInfo) {
            // Mulai transaksi untuk memastikan semua operasi berhasil atau gagal bersama-sama
            $connection->beginTransaction();
            
            // 1. Dapatkan thesis_titles yang terkait dengan file ini
            $thesisTitles = $database->fetchAll("SELECT id FROM thesis_titles WHERE upload_file_id = ?", [$id]);
            $thesisTitleIds = array_column($thesisTitles, 'id');
            
            // 2. Hapus classification_results yang terkait dengan thesis_titles dari file ini
            if (!empty($thesisTitleIds)) {
                $placeholders = implode(',', array_fill(0, count($thesisTitleIds), '?'));
                $database->query(
                    "DELETE FROM classification_results WHERE title_id IN ($placeholders)",
                    $thesisTitleIds
                );
                
                // 3. Hapus thesis_titles yang terkait dengan file ini
                $database->query(
                    "DELETE FROM thesis_titles WHERE upload_file_id = ?",
                    [$id]
                );
            }
            
            // 4. Hapus model_performances yang terkait dengan file ini
            $database->query("DELETE FROM model_performances WHERE upload_file_id = ?", [$id]);
            
            // 5. Hapus keyword_analysis yang terkait dengan file ini
            $database->query("DELETE FROM keyword_analysis WHERE upload_file_id = ?", [$id]);
            
            // 6. Hapus predictions yang terkait dengan file ini
            $database->query("DELETE FROM predictions WHERE upload_file_id = ?", [$id]);
            
            // 7. TAMBAHKAN: Hapus model_visualizations yang terkait dengan file ini
            $database->query("DELETE FROM model_visualizations WHERE upload_file_id = ?", [$id]);
            
            // 8. Jika ada file fisik yang perlu dihapus
            if (isset($fileInfo['filename']) && file_exists($fileInfo['filename'])) {
                unlink($fileInfo['filename']);
            }
            
            // 9. Hapus file upload dari database
            $success = $database->query("DELETE FROM uploaded_files WHERE id = ?", [$id]);
            
            // Commit transaksi jika semua berhasil
            $connection->commit();
            
            if ($success) {
                $message = 'File dan semua data terkait berhasil dihapus.';
                
                // Log aktivitas
                $database->query(
                    "INSERT INTO activity_log (action, details, ip_address) VALUES (?, ?, ?)",
                    ['delete_file', "File ID: $id dihapus", $_SERVER['REMOTE_ADDR'] ?? null]
                );
            } else {
                $message = 'Gagal menghapus file.';
            }
        } else {
            $message = 'File tidak ditemukan.';
        }
    } catch (Exception $e) {
        // Rollback transaksi jika ada error
        if ($connection && $connection->inTransaction()) {
            $connection->rollBack();
        }
        $message = 'Terjadi kesalahan: ' . $e->getMessage();
    }
    
    // Redirect dengan pesan
    header('Location: history.php?success=' . ($success ? '1' : '0') . '&message=' . urlencode($message));
    exit;
} else {
    // Redirect jika tidak ada ID
    header('Location: history.php?success=0&message=' . urlencode('ID tidak valid.'));
    exit;
}
?>