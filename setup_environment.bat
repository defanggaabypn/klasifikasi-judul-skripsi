@echo off
ECHO =====================================================
ECHO        SETUP ENVIRONMENT UNTUK KLASIFIKASI JUDUL
ECHO =====================================================
ECHO.
ECHO Membuat Virtual Environment...

cd python-api

IF NOT EXIST venv (
    python -m venv venv
    ECHO Virtual environment berhasil dibuat.
) ELSE (
    ECHO Virtual environment sudah ada.
)

ECHO.
ECHO Mengaktifkan Virtual Environment...
call venv\Scripts\activate

ECHO.
ECHO Menginstal dependensi (ini mungkin memakan waktu beberapa menit)...
pip install -r requirements.txt

ECHO.
ECHO Setup selesai! Silakan jalankan start-server.bat untuk memulai aplikasi.
ECHO =====================================================

pause