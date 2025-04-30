<?php
// File: web/database_config.php
// Konfigurasi koneksi database

class Database {
    private $host = 'localhost';
    private $db_name = 'skripsi_classification';
    private $username = 'root';
    private $password = '';
    private $conn;

    // Koneksi ke database
    public function getConnection() {
        $this->conn = null;

        try {
            $this->conn = new PDO(
                "mysql:host=" . $this->host . ";dbname=" . $this->db_name,
                $this->username,
                $this->password
            );
            $this->conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
            $this->conn->exec("set names utf8");
        } catch(PDOException $e) {
            echo "Kesalahan Koneksi Database: " . $e->getMessage();
        }

        return $this->conn;
    }

    // Fungsi untuk eksekusi query
    public function query($sql, $params = []) {
        try {
            $stmt = $this->conn->prepare($sql);
            $stmt->execute($params);
            return $stmt;
        } catch(PDOException $e) {
            echo "Kesalahan Query: " . $e->getMessage();
            return false;
        }
    }

    // Fungsi untuk mengambil semua data
    public function fetchAll($sql, $params = []) {
        $stmt = $this->query($sql, $params);
        return $stmt ? $stmt->fetchAll(PDO::FETCH_ASSOC) : [];
    }

    // Fungsi untuk mengambil satu baris data
    public function fetch($sql, $params = []) {
        $stmt = $this->query($sql, $params);
        return $stmt ? $stmt->fetch(PDO::FETCH_ASSOC) : null;
    }

    // Fungsi untuk insert dan return id
    public function insert($sql, $params = []) {
        $stmt = $this->query($sql, $params);
        return $stmt ? $this->conn->lastInsertId() : false;
    }
}