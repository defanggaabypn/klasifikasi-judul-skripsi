-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 01 Bulan Mei 2025 pada 00.50
-- Versi server: 10.4.32-MariaDB
-- Versi PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `skripsi_classification`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `activity_log`
--

CREATE TABLE `activity_log` (
  `id` int(11) NOT NULL,
  `action` varchar(50) NOT NULL,
  `details` text DEFAULT NULL,
  `ip_address` varchar(45) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `categories`
--

CREATE TABLE `categories` (
  `id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL,
  `description` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `categories`
--

INSERT INTO `categories` (`id`, `name`, `description`) VALUES
(1, 'RPL', 'Rekayasa Perangkat Lunak'),
(2, 'Jaringan', 'Jaringan Komputer'),
(3, 'Multimedia', 'Desain dan Multimedia');

-- --------------------------------------------------------

--
-- Struktur dari tabel `keyword_analysis`
--

CREATE TABLE `keyword_analysis` (
  `id` int(11) NOT NULL,
  `category_id` int(11) DEFAULT NULL,
  `keyword` varchar(100) NOT NULL,
  `frequency` int(11) DEFAULT 1
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `model_performances`
--

CREATE TABLE `model_performances` (
  `id` int(11) NOT NULL,
  `model_name` varchar(50) NOT NULL,
  `accuracy` float NOT NULL,
  `parameters` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`parameters`)),
  `training_date` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `predictions`
--

CREATE TABLE `predictions` (
  `id` int(11) NOT NULL,
  `title` text NOT NULL,
  `actual_category_id` int(11) DEFAULT NULL,
  `knn_prediction_id` int(11) DEFAULT NULL,
  `dt_prediction_id` int(11) DEFAULT NULL,
  `confidence` float DEFAULT 0,
  `prediction_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `upload_file_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `thesis_titles`
--

CREATE TABLE `thesis_titles` (
  `id` int(11) NOT NULL,
  `title` text NOT NULL,
  `category_id` int(11) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Struktur dari tabel `uploaded_files`
--

CREATE TABLE `uploaded_files` (
  `id` int(11) NOT NULL,
  `filename` varchar(255) NOT NULL,
  `original_filename` varchar(255) NOT NULL,
  `file_size` int(11) NOT NULL,
  `upload_date` timestamp NOT NULL DEFAULT current_timestamp(),
  `processed` tinyint(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `activity_log`
--
ALTER TABLE `activity_log`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `categories`
--
ALTER TABLE `categories`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `keyword_analysis`
--
ALTER TABLE `keyword_analysis`
  ADD PRIMARY KEY (`id`),
  ADD KEY `category_id` (`category_id`);

--
-- Indeks untuk tabel `model_performances`
--
ALTER TABLE `model_performances`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `predictions`
--
ALTER TABLE `predictions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `actual_category_id` (`actual_category_id`),
  ADD KEY `knn_prediction_id` (`knn_prediction_id`),
  ADD KEY `dt_prediction_id` (`dt_prediction_id`),
  ADD KEY `fk_predictions_upload_file` (`upload_file_id`);

--
-- Indeks untuk tabel `thesis_titles`
--
ALTER TABLE `thesis_titles`
  ADD PRIMARY KEY (`id`),
  ADD KEY `category_id` (`category_id`);

--
-- Indeks untuk tabel `uploaded_files`
--
ALTER TABLE `uploaded_files`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `activity_log`
--
ALTER TABLE `activity_log`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT untuk tabel `categories`
--
ALTER TABLE `categories`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT untuk tabel `keyword_analysis`
--
ALTER TABLE `keyword_analysis`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=481;

--
-- AUTO_INCREMENT untuk tabel `model_performances`
--
ALTER TABLE `model_performances`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=17;

--
-- AUTO_INCREMENT untuk tabel `predictions`
--
ALTER TABLE `predictions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1522;

--
-- AUTO_INCREMENT untuk tabel `thesis_titles`
--
ALTER TABLE `thesis_titles`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=945;

--
-- AUTO_INCREMENT untuk tabel `uploaded_files`
--
ALTER TABLE `uploaded_files`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- Ketidakleluasaan untuk tabel pelimpahan (Dumped Tables)
--

--
-- Ketidakleluasaan untuk tabel `keyword_analysis`
--
ALTER TABLE `keyword_analysis`
  ADD CONSTRAINT `keyword_analysis_ibfk_1` FOREIGN KEY (`category_id`) REFERENCES `categories` (`id`);

--
-- Ketidakleluasaan untuk tabel `predictions`
--
ALTER TABLE `predictions`
  ADD CONSTRAINT `fk_predictions_upload_file` FOREIGN KEY (`upload_file_id`) REFERENCES `uploaded_files` (`id`) ON DELETE SET NULL,
  ADD CONSTRAINT `predictions_ibfk_1` FOREIGN KEY (`actual_category_id`) REFERENCES `categories` (`id`),
  ADD CONSTRAINT `predictions_ibfk_2` FOREIGN KEY (`knn_prediction_id`) REFERENCES `categories` (`id`),
  ADD CONSTRAINT `predictions_ibfk_3` FOREIGN KEY (`dt_prediction_id`) REFERENCES `categories` (`id`);

--
-- Ketidakleluasaan untuk tabel `thesis_titles`
--
ALTER TABLE `thesis_titles`
  ADD CONSTRAINT `thesis_titles_ibfk_1` FOREIGN KEY (`category_id`) REFERENCES `categories` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
