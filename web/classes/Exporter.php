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
    public function exportToExcel($predictionData, $filename = 'hasil_klasifikasi.xlsx', $upload_file_id = null) {
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
        $statSheet->setCellValue('A1', 'Statistik Hasil Klasifikasi');
        $statSheet->mergeCells('A1:C1');
        $statSheet->getStyle('A1')->getFont()->setBold(true);
        $statSheet->getStyle('A1')->getFont()->setSize(14);
        
        // Data statistik
        $statSheet->setCellValue('A3', 'Model');
        $statSheet->setCellValue('B3', 'Akurasi');
        $statSheet->getStyle('A3:B3')->applyFromArray($headerStyle);
        
        // Ambil statistik akurasi dari database berdasarkan upload_file_id jika ada
        if ($upload_file_id) {
            $modelStats = $this->database->fetchAll(
                "SELECT model_name, accuracy FROM model_performances 
                 WHERE upload_file_id = ? 
                 ORDER BY training_date DESC", 
                [$upload_file_id]
            );
        } else {
            // Gunakan statistik global jika tidak ada upload_file_id spesifik
            $modelStats = $this->database->fetchAll(
                "SELECT model_name, accuracy FROM model_performances 
                 ORDER BY training_date DESC 
                 LIMIT 2"
            );
        }
        
        $row = 4;
        foreach ($modelStats as $stat) {
            $statSheet->setCellValue('A' . $row, $stat['model_name']);
            $statSheet->setCellValue('B' . $row, number_format($stat['accuracy'] * 100, 2) . '%');
            $row++;
        }
        
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
    public function exportToPDF($predictionData, $filename = 'hasil_klasifikasi.pdf', $upload_file_id = null) {
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
        
        // Ambil statistik akurasi dari database berdasarkan upload_file_id jika ada
        if ($upload_file_id) {
            $modelStats = $this->database->fetchAll(
                "SELECT model_name, accuracy FROM model_performances 
                 WHERE upload_file_id = ? 
                 ORDER BY training_date DESC", 
                [$upload_file_id]
            );
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
        
        // Header PDF
        $html = '
        <div class="header">
            <h1>Hasil Klasifikasi Judul Skripsi</h1>
            <p>Diekspor pada: ' . date('d/m/Y H:i:s') . '</p>
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
        </div>';
        
        // Tambahkan HTML ke PDF
        $mpdf->WriteHTML($html);
        
        // Tambahkan header dan footer
        $mpdf->SetHeader('Sistem Klasifikasi Judul Skripsi|' . date('d/m/Y') . '|Halaman {PAGENO}');
        $mpdf->SetFooter('Diekspor via Aplikasi Klasifikasi Judul Skripsi');
        
        // Simpan ke file temporer
        $tempFile = tempnam(sys_get_temp_dir(), 'pdf_');
        $mpdf->Output($tempFile, 'F');
        
        return $tempFile;
    }
}