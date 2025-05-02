<?php
// File: web/classes/Exporter.php
// Class untuk menghandle export data

require_once 'vendor/autoload.php'; // Composer autoload

use PhpOffice\PhpSpreadsheet\Spreadsheet;
use PhpOffice\PhpSpreadsheet\Writer\Xlsx;
use Mpdf\Mpdf;

class Exporter {
    private $database;
    
    public function __construct($database) {
        $this->database = $database;
    }
    
    // Export hasil ke Excel
    public function exportToExcel($predictionData, $filename = 'hasil_klasifikasi.xlsx', $upload_file_id = null, $isOverview = false) {
        // Buat spreadsheet baru
        $spreadsheet = new Spreadsheet();
        $sheet = $spreadsheet->getActiveSheet();
        
        // Set header
        $sheet->setCellValue('A1', 'No');
        $sheet->setCellValue('B1', 'Judul Skripsi');
        $sheet->setCellValue('C1', 'Kategori Sebenarnya');
        $sheet->setCellValue('D1', 'Prediksi KNN');
        $sheet->setCellValue('E1', 'Prediksi Decision Tree');
        $sheet->setCellValue('F1', 'Status');
        
        // Styling header
        $headerStyle = [
            'font' => [
                'bold' => true,
                'color' => ['rgb' => 'FFFFFF'],
            ],
            'fill' => [
                'fillType' => \PhpOffice\PhpSpreadsheet\Style\Fill::FILL_SOLID,
                'startColor' => ['rgb' => '0D6EFD'],
            ],
            'alignment' => [
                'horizontal' => \PhpOffice\PhpSpreadsheet\Style\Alignment::HORIZONTAL_CENTER,
            ],
        ];
        
        $sheet->getStyle('A1:F1')->applyFromArray($headerStyle);
        
        // Set data
        $row = 2;
        foreach ($predictionData as $index => $data) {
            $correctKNN = $data['actual'] === $data['knn_pred'];
            $correctDT = $data['actual'] === $data['dt_pred'];
            
            $status = 'Tidak Tepat';
            if ($correctKNN && $correctDT) {
                $status = 'Tepat (Kedua Model)';
            } else if ($correctKNN) {
                $status = 'Tepat (KNN)';
            } else if ($correctDT) {
                $status = 'Tepat (Decision Tree)';
            }
            
            $sheet->setCellValue('A' . $row, $index + 1);
            $sheet->setCellValue('B' . $row, $data['full_title'] ?? $data['title']);
            $sheet->setCellValue('C' . $row, $data['actual']);
            $sheet->setCellValue('D' . $row, $data['knn_pred']);
            $sheet->setCellValue('E' . $row, $data['dt_pred']);
            $sheet->setCellValue('F' . $row, $status);
            
            // Warnai baris berdasarkan status
            if ($correctKNN && $correctDT) {
                $sheet->getStyle('A' . $row . ':F' . $row)->getFill()
                      ->setFillType(\PhpOffice\PhpSpreadsheet\Style\Fill::FILL_SOLID)
                      ->getStartColor()->setRGB('D1E7DD');
            } else if (!$correctKNN && !$correctDT) {
                $sheet->getStyle('A' . $row . ':F' . $row)->getFill()
                      ->setFillType(\PhpOffice\PhpSpreadsheet\Style\Fill::FILL_SOLID)
                      ->getStartColor()->setRGB('F8D7DA');
            }
            
            $row++;
        }
        
        // Auto size kolom
        foreach(range('A','F') as $col) {
            $sheet->getColumnDimension($col)->setAutoSize(true);
        }
        
        // Tambahkan sheet untuk statistik
        $statSheet = $spreadsheet->createSheet();
        $statSheet->setTitle('Statistik');
        
        // Header statistik
        if ($upload_file_id || $isOverview) {
            $statSheet->setCellValue('A1', $isOverview ? 'Statistik Hasil Klasifikasi (Ringkasan Keseluruhan)' : 'Statistik Hasil Klasifikasi');
        } else {
            $statSheet->setCellValue('A1', 'Statistik Hasil Klasifikasi (Terbaru)');
        }
        $statSheet->mergeCells('A1:C1');
        $statSheet->getStyle('A1')->getFont()->setBold(true);
        $statSheet->getStyle('A1')->getFont()->setSize(14);
        
        // Tambahkan informasi sumber data
        if ($upload_file_id) {
            $statSheet->setCellValue('A2', 'Sumber: File spesifik');
        } else if ($isOverview) {
            $statSheet->setCellValue('A2', 'Sumber: Semua data dalam sistem');
        } else {
            $statSheet->setCellValue('A2', 'Sumber: 100 data terbaru');
        }
        $statSheet->mergeCells('A2:C2');
        $statSheet->getStyle('A2')->getFont()->setItalic(true);
        
        // Data statistik
        $statSheet->setCellValue('A4', 'Model');
        $statSheet->setCellValue('B4', 'Akurasi');
        $statSheet->getStyle('A4:B4')->applyFromArray($headerStyle);
        
        // Ambil statistik akurasi dari database
        if ($upload_file_id) {
            $modelStats = $this->database->fetchAll(
                "SELECT model_name, accuracy FROM model_performances 
                 WHERE upload_file_id = ? 
                 ORDER BY training_date DESC", 
                [$upload_file_id]
            );
        } else if ($isOverview) {
            // Jika ini adalah overview, coba ambil statistik global terlebih dahulu
            $globalStats = $this->database->fetchAll(
                "SELECT model_name, accuracy FROM model_performances 
                 WHERE upload_file_id IS NULL
                 ORDER BY training_date DESC"
            );
            
            // Jika tidak ada statistik global, hitung akurasi dari semua data
            if (empty($globalStats)) {
                // Hitung akurasi global dari semua prediksi
                $totalPredictions = $this->database->fetch(
                    "SELECT COUNT(*) as total FROM predictions"
                );
                
                $correctKNN = $this->database->fetch(
                    "SELECT COUNT(*) as correct FROM predictions p
                     JOIN categories c1 ON p.actual_category_id = c1.id
                     JOIN categories c2 ON p.knn_prediction_id = c2.id
                     WHERE c1.name = c2.name"
                );
                
                $correctDT = $this->database->fetch(
                    "SELECT COUNT(*) as correct FROM predictions p
                     JOIN categories c1 ON p.actual_category_id = c1.id
                     JOIN categories c3 ON p.dt_prediction_id = c3.id
                     WHERE c1.name = c3.name"
                );
                
                $totalCount = $totalPredictions['total'];
                $knnAcc = $totalCount > 0 ? $correctKNN['correct'] / $totalCount : 0;
                $dtAcc = $totalCount > 0 ? $correctDT['correct'] / $totalCount : 0;
                
                $modelStats = [
                    ['model_name' => 'KNN', 'accuracy' => $knnAcc],
                    ['model_name' => 'Decision Tree', 'accuracy' => $dtAcc]
                ];
            } else {
                $modelStats = $globalStats;
            }
        } else {
            // Gunakan statistik global jika tidak ada upload_file_id spesifik
            $modelStats = $this->database->fetchAll(
                "SELECT model_name, accuracy FROM model_performances 
                 ORDER BY training_date DESC 
                 LIMIT 2"
            );
        }
        
        $row = 5;
        foreach ($modelStats as $stat) {
            $statSheet->setCellValue('A' . $row, $stat['model_name']);
            $statSheet->setCellValue('B' . $row, number_format($stat['accuracy'] * 100, 2) . '%');
            $row++;
        }
        
        // Tambahkan informasi jumlah data
        $statSheet->setCellValue('A' . ($row + 1), 'Jumlah Data');
        $statSheet->setCellValue('B' . ($row + 1), count($predictionData));
        $statSheet->getStyle('A' . ($row + 1))->getFont()->setBold(true);
        
        // Tambahkan tanggal ekspor
        $statSheet->setCellValue('A' . ($row + 3), 'Tanggal Ekspor');
        $statSheet->setCellValue('B' . ($row + 3), date('d/m/Y H:i:s'));
        $statSheet->getStyle('A' . ($row + 3))->getFont()->setBold(true);
        
        // Set sheet aktif kembali ke sheet pertama
        $spreadsheet->setActiveSheetIndex(0);
        
        // Buat file Excel
        $writer = new Xlsx($spreadsheet);
        
        // Simpan ke file temporer
        $tempFile = tempnam(sys_get_temp_dir(), 'excel_');
        $writer->save($tempFile);
        
        return $tempFile;
    }
    
    // Export hasil ke PDF
    public function exportToPDF($predictionData, $filename = 'hasil_klasifikasi.pdf', $upload_file_id = null, $isOverview = false) {
        // Inisialisasi mPDF
        $mpdf = new Mpdf([
            'mode' => 'utf-8',
            'format' => 'A4-L',
            'margin_left' => 15,
            'margin_right' => 15,
            'margin_top' => 16,
            'margin_bottom' => 16,
            'margin_header' => 9,
            'margin_footer' => 9
        ]);
        
        // Tambahkan CSS
        $stylesheet = file_get_contents('assets/css/pdf-style.css');
        $mpdf->WriteHTML($stylesheet, \Mpdf\HTMLParserMode::HEADER_CSS);
        
        // Ambil statistik akurasi dari database
        if ($upload_file_id) {
            $modelStats = $this->database->fetchAll(
                "SELECT model_name, accuracy FROM model_performances 
                 WHERE upload_file_id = ? 
                 ORDER BY training_date DESC", 
                [$upload_file_id]
            );
        } else if ($isOverview) {
            // Jika ini adalah overview, coba ambil statistik global terlebih dahulu
            $globalStats = $this->database->fetchAll(
                "SELECT model_name, accuracy FROM model_performances 
                 WHERE upload_file_id IS NULL
                 ORDER BY training_date DESC"
            );
            
            // Jika tidak ada statistik global, hitung akurasi dari semua data
            if (empty($globalStats)) {
                // Hitung akurasi global dari semua prediksi
                $totalPredictions = $this->database->fetch(
                    "SELECT COUNT(*) as total FROM predictions"
                );
                
                $correctKNN = $this->database->fetch(
                    "SELECT COUNT(*) as correct FROM predictions p
                     JOIN categories c1 ON p.actual_category_id = c1.id
                     JOIN categories c2 ON p.knn_prediction_id = c2.id
                     WHERE c1.name = c2.name"
                );
                
                $correctDT = $this->database->fetch(
                    "SELECT COUNT(*) as correct FROM predictions p
                     JOIN categories c1 ON p.actual_category_id = c1.id
                     JOIN categories c3 ON p.dt_prediction_id = c3.id
                     WHERE c1.name = c3.name"
                );
                
                $totalCount = $totalPredictions['total'];
                $knnAcc = $totalCount > 0 ? $correctKNN['correct'] / $totalCount : 0;
                $dtAcc = $totalCount > 0 ? $correctDT['correct'] / $totalCount : 0;
                
                $modelStats = [
                    ['model_name' => 'KNN', 'accuracy' => $knnAcc],
                    ['model_name' => 'Decision Tree', 'accuracy' => $dtAcc]
                ];
            } else {
                $modelStats = $globalStats;
            }
        } else {
            // Gunakan statistik global jika tidak ada upload_file_id spesifik
            $modelStats = $this->database->fetchAll(
                "SELECT model_name, accuracy FROM model_performances 
                 ORDER BY training_date DESC 
                 LIMIT 2"
            );
        }
        
        $knn_acc = 0;
        $dt_acc = 0;
        
        foreach ($modelStats as $stat) {
            if ($stat['model_name'] == 'KNN') {
                $knn_acc = $stat['accuracy'];
            } else if ($stat['model_name'] == 'Decision Tree') {
                $dt_acc = $stat['accuracy'];
            }
        }
        
        // Tentukan judul berdasarkan jenis ekspor
        $title = $upload_file_id ? 'Hasil Klasifikasi Judul Skripsi' : 
                 ($isOverview ? 'Ringkasan Klasifikasi Judul Skripsi (Keseluruhan)' : 'Hasil Klasifikasi Judul Skripsi (Terbaru)');
        
        // Header PDF
        $html = '
        <div class="header">
            <h1>' . $title . '</h1>
            <p>Diekspor pada: ' . date('d/m/Y H:i:s') . '</p>
        </div>
        
        <div class="data-source-info">
            <p><strong>Sumber Data:</strong> ' . 
            ($upload_file_id ? 'File spesifik' : 
             ($isOverview ? 'Semua data dalam sistem' : '100 data terbaru')) . '</p>
        </div>
        
        <div class="stat-box">
            <div class="stat-item">
                <h3>Akurasi KNN</h3>
                <div class="stat-value">' . number_format($knn_acc * 100, 2) . '%</div>
            </div>
            <div class="stat-item">
                <h3>Akurasi Decision Tree</h3>
                <div class="stat-value">' . number_format($dt_acc * 100, 2) . '%</div>
            </div>
        </div>
        
        <h2>Hasil Prediksi</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>No</th>
                    <th width="40%">Judul Skripsi</th>
                    <th>Kategori Sebenarnya</th>
                    <th>Prediksi KNN</th>
                    <th>Prediksi Decision Tree</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>';
        
        // Isi tabel
        foreach ($predictionData as $index => $data) {
            $correctKNN = $data['actual'] === $data['knn_pred'];
            $correctDT = $data['actual'] === $data['dt_pred'];
            
            $status = 'Tidak Tepat';
            $statusClass = 'incorrect';
            
            if ($correctKNN && $correctDT) {
                $status = 'Tepat (Kedua Model)';
                $statusClass = 'correct-both';
            } else if ($correctKNN) {
                $status = 'Tepat (KNN)';
                $statusClass = 'correct-knn';
            } else if ($correctDT) {
                $status = 'Tepat (Decision Tree)';
                $statusClass = 'correct-dt';
            }
            
            $html .= '<tr class="' . $statusClass . '">
                <td>' . ($index + 1) . '</td>
                <td>' . htmlspecialchars($data['full_title'] ?? $data['title']) . '</td>
                <td>' . htmlspecialchars($data['actual']) . '</td>
                <td>' . htmlspecialchars($data['knn_pred']) . '</td>
                <td>' . htmlspecialchars($data['dt_pred']) . '</td>
                <td>' . $status . '</td>
            </tr>';
        }
        
        $html .= '</tbody></table>';
        
        // Tambahkan footer statistik
        $html .= '
        <div class="footer-stats">
            <p><strong>Total Data:</strong> ' . count($predictionData) . '</p>
            <p><strong>Jumlah Tepat (KNN):</strong> ' . floor($knn_acc * count($predictionData)) . ' (' . number_format($knn_acc * 100, 2) . '%)</p>
            <p><strong>Jumlah Tepat (Decision Tree):</strong> ' . floor($dt_acc * count($predictionData)) . ' (' . number_format($dt_acc * 100, 2) . '%)</p>
        </div>';
        
        // Tambahkan HTML ke PDF
        $mpdf->WriteHTML($html);
        
        // Tambahkan header dan footer
        $mpdf->SetHeader('Sistem Klasifikasi Judul Skripsi|' . date('d/m/Y') . '|Halaman {PAGENO}');
        $mpdf->SetFooter('ANALISIS PERBANDINGAN ALGORITMA K-NEAREST NEIGHBORS (KNN) DAN DECISION TREE BERDASARKAN HASIL SEMATIC SIMILARITY JUDUL SKRIPSI DAN BIDANG KONSENSTRASI (STUDI KASUS : JURUSAN PENDIDIKAN TEKNOLOGI INFORMASI DAN KOMUNIKASI)');
        
        // Simpan ke file temporer
        $tempFile = tempnam(sys_get_temp_dir(), 'pdf_');
        $mpdf->Output($tempFile, 'F');
        
        return $tempFile;
    }
}