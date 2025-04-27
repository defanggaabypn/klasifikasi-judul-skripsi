@echo off
ECHO =====================================================
ECHO        SISTEM KLASIFIKASI JUDUL SKRIPSI
ECHO               v1.0.0
ECHO =====================================================
ECHO.
ECHO Memulai API Python (IndoBERT + Machine Learning)...
ECHO PERINGATAN: JANGAN TUTUP WINDOW INI!
ECHO.
ECHO Memeriksa folder struktur...

SET BASE_DIR=%~dp0
SET API_DIR=%BASE_DIR%python-api

IF NOT EXIST "%API_DIR%" (
    mkdir "%API_DIR%"
)

IF NOT EXIST "%API_DIR%\uploads" (
    mkdir "%API_DIR%\uploads"
)

IF NOT EXIST "%API_DIR%\models" (
    mkdir "%API_DIR%\models"
)

IF NOT EXIST "%API_DIR%\cache" (
    mkdir "%API_DIR%\cache"
)

ECHO Struktur folder sudah siap.
ECHO.

cd "%API_DIR%"

ECHO Memeriksa keberadaan app.py...
IF NOT EXIST app.py (
    ECHO File app.py tidak ditemukan di %API_DIR%
    ECHO Pastikan file app.py telah dibuat dengan benar.
    GOTO ERROR
)

ECHO Memulai server API...
IF EXIST venv\Scripts\activate.bat (
    CALL venv\Scripts\activate.bat
    python app.py
) ELSE (
    ECHO Virtual environment tidak ditemukan, menggunakan Python default
    python app.py
)

GOTO END

:ERROR
ECHO =====================================================
ECHO   Terjadi kesalahan dalam menjalankan server API
ECHO =====================================================
PAUSE

:END