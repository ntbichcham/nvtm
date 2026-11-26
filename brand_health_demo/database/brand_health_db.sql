-- =====================================================================
-- brand_health_db.sql
-- Database cho website demo:
-- "Phân tích cảm xúc theo khía cạnh phục vụ theo dõi sức khỏe thương hiệu"
--
-- DBMS   : MySQL 8+ / MariaDB 10.4+ (XAMPP)
-- Charset: utf8mb4
-- Engine : InnoDB
--
-- Quy ước nhãn của mô hình PhoBERT:
--   Aspect model_index : 0..9  (đúng thứ tự ASPECT_LIST)
--   Sentiment model_index:
--       0 = None
--       1 = Negative
--       2 = Neutral
--       3 = Positive
--
-- Lưu ý:
--   - analysis_results CHỈ lưu các khía cạnh được mô hình dự đoán khác None.
--   - Không lưu đường dẫn checkpoint trong database.
--   - Không có trường thương hiệu vì UIT-ViSFD không cung cấp định danh thương hiệu.
-- =====================================================================

SET NAMES utf8mb4;
SET time_zone = '+07:00';

CREATE DATABASE IF NOT EXISTS brand_health_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE brand_health_db;

-- ---------------------------------------------------------------------
-- TÙY CHỌN: xóa toàn bộ bảng để tạo lại từ đầu.
-- Chỉ bỏ comment 5 dòng dưới nếu bạn thực sự muốn xóa dữ liệu cũ.
-- ---------------------------------------------------------------------
-- DROP TABLE IF EXISTS analysis_results;
-- DROP TABLE IF EXISTS reviews;
-- DROP TABLE IF EXISTS aspects;
-- DROP TABLE IF EXISTS sentiment_labels;
-- DROP TABLE IF EXISTS users;

-- ---------------------------------------------------------------------
-- 1. users
-- Tài khoản đăng nhập quản trị.
-- Mật khẩu lưu bằng password_hash() của PHP (bcrypt), không lưu plaintext.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    username        VARCHAR(50)     NOT NULL,
    password_hash   VARCHAR(255)    NOT NULL,
    full_name       VARCHAR(100)    NOT NULL,
    role            ENUM('admin', 'analyst') NOT NULL DEFAULT 'admin',
    is_active       TINYINT(1)      NOT NULL DEFAULT 1,
    last_login_at   DATETIME        NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
                                    ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_users_username (username),
    KEY idx_users_active (is_active)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = 'Tài khoản quản trị website';

-- ---------------------------------------------------------------------
-- 2. aspects
-- Danh mục 10 khía cạnh của UIT-ViSFD.
-- model_index giữ đúng thứ tự đầu ra của mô hình (0..9).
-- weight là trọng số w(a) dùng cho BHI; ban đầu bằng nhau = 0.1.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aspects (
    id              TINYINT UNSIGNED NOT NULL AUTO_INCREMENT,
    code            VARCHAR(20)      NOT NULL,
    name_vi         VARCHAR(100)     NOT NULL,
    model_index     TINYINT UNSIGNED NOT NULL,
    weight          DECIMAL(6,5)     NOT NULL DEFAULT 0.10000,
    display_order   TINYINT UNSIGNED NOT NULL,
    is_active       TINYINT(1)       NOT NULL DEFAULT 1,
    PRIMARY KEY (id),
    UNIQUE KEY uq_aspects_code (code),
    UNIQUE KEY uq_aspects_model_index (model_index),
    KEY idx_aspects_display_order (display_order),
    CONSTRAINT chk_aspects_model_index
        CHECK (model_index BETWEEN 0 AND 9),
    CONSTRAINT chk_aspects_weight
        CHECK (weight >= 0),
    CONSTRAINT chk_aspects_display_order
        CHECK (display_order BETWEEN 1 AND 10)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '10 khía cạnh UIT-ViSFD và trọng số BHI';

-- ---------------------------------------------------------------------
-- 3. sentiment_labels
-- Danh mục 4 nhãn cảm xúc của mô hình.
-- is_mention = 0 cho nhãn None (không đề cập khía cạnh).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sentiment_labels (
    id              TINYINT UNSIGNED NOT NULL AUTO_INCREMENT,
    code            VARCHAR(20)      NOT NULL,
    name_vi         VARCHAR(50)      NOT NULL,
    model_index     TINYINT UNSIGNED NOT NULL,
    is_mention      TINYINT(1)       NOT NULL DEFAULT 1,
    display_order   TINYINT UNSIGNED NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_sentiment_labels_code (code),
    UNIQUE KEY uq_sentiment_labels_model_index (model_index),
    KEY idx_sentiment_labels_mention (is_mention),
    CONSTRAINT chk_sentiment_labels_model_index
        CHECK (model_index BETWEEN 0 AND 3),
    CONSTRAINT chk_sentiment_labels_display_order
        CHECK (display_order BETWEEN 0 AND 3)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '4 nhãn cảm xúc: None, Negative, Neutral, Positive';

-- ---------------------------------------------------------------------
-- 4. reviews
-- Nội dung đánh giá gốc và trạng thái phân tích.
-- review_date dùng để lọc thời gian / vẽ biểu đồ theo thời gian.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviews (
    id                  BIGINT UNSIGNED  NOT NULL AUTO_INCREMENT,
    external_id         VARCHAR(100)     NULL
                            COMMENT 'ID gốc khi import CSV/UIT-ViSFD; NULL nếu nhập tay',
    source              VARCHAR(30)      NOT NULL DEFAULT 'manual'
                            COMMENT 'manual, csv, uit_visfd, api...',
    content             TEXT             NOT NULL,
    rating              TINYINT UNSIGNED NULL
                            COMMENT 'Số sao 1..5; NULL nếu không có',
    review_date         DATE             NOT NULL
                            COMMENT 'Ngày đánh giá thực tế; dùng cho biểu đồ thời gian',
    analysis_status     ENUM('pending', 'processing', 'completed', 'failed')
                            NOT NULL DEFAULT 'pending',
    analysis_error      VARCHAR(500)     NULL
                            COMMENT 'Thông báo lỗi an toàn cho người dùng',
    analyzed_at         DATETIME         NULL,
    model_name          VARCHAR(100)     NULL
                            COMMENT 'Tên mô hình, ví dụ PhoBERT-ABSA',
    model_version       VARCHAR(100)     NULL
                            COMMENT 'Phiên bản/logic, không lưu đường dẫn checkpoint',
    inference_ms        INT UNSIGNED     NULL
                            COMMENT 'Thời gian suy luận (ms)',
    created_by          INT UNSIGNED     NULL,
    created_at          TIMESTAMP        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP        NOT NULL DEFAULT CURRENT_TIMESTAMP
                                        ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_reviews_source_external (source, external_id),
    KEY idx_reviews_review_date (review_date),
    KEY idx_reviews_status_date (analysis_status, review_date),
    KEY idx_reviews_created_at (created_at),
    KEY idx_reviews_created_by (created_by),
    CONSTRAINT fk_reviews_created_by
        FOREIGN KEY (created_by) REFERENCES users (id)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    CONSTRAINT chk_reviews_content
        CHECK (CHAR_LENGTH(TRIM(content)) > 0),
    CONSTRAINT chk_reviews_rating
        CHECK (rating IS NULL OR rating BETWEEN 1 AND 5)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = 'Đánh giá sản phẩm và trạng thái phân tích';

-- ---------------------------------------------------------------------
-- 5. analysis_results
-- Mỗi dòng = một khía cạnh được mô hình nhận diện (khác None)
-- trong một đánh giá, kèm cảm xúc và độ tin cậy.
--
-- Ràng buộc UNIQUE (review_id, aspect_id) bảo đảm mỗi đánh giá
-- chỉ có tối đa một kết quả cho mỗi khía cạnh.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analysis_results (
    id                  BIGINT UNSIGNED  NOT NULL AUTO_INCREMENT,
    review_id           BIGINT UNSIGNED  NOT NULL,
    aspect_id           TINYINT UNSIGNED NOT NULL,
    sentiment_label_id  TINYINT UNSIGNED NOT NULL,
    confidence          DECIMAL(6,5)     NULL
                            COMMENT 'Xác suất softmax của nhãn được chọn; NULL nếu model không trả về',
    created_at          TIMESTAMP        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_analysis_review_aspect (review_id, aspect_id),
    KEY idx_analysis_aspect_sentiment (aspect_id, sentiment_label_id),
    KEY idx_analysis_sentiment (sentiment_label_id),
    CONSTRAINT fk_analysis_review
        FOREIGN KEY (review_id) REFERENCES reviews (id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_analysis_aspect
        FOREIGN KEY (aspect_id) REFERENCES aspects (id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,
    CONSTRAINT fk_analysis_sentiment
        FOREIGN KEY (sentiment_label_id) REFERENCES sentiment_labels (id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,
    CONSTRAINT chk_analysis_confidence
        CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = 'Kết quả ABSA: chỉ lưu khía cạnh được nhận diện, không lưu None';

-- =====================================================================
-- DỮ LIỆU DANH MỤC
-- =====================================================================

-- 10 khía cạnh, giữ đúng thứ tự ASPECT_LIST trong dataset_loader.py.
-- Trọng số ban đầu bằng nhau: 0.1 cho mỗi khía cạnh.
INSERT INTO aspects
    (id, code, name_vi, model_index, weight, display_order, is_active)
VALUES
    (1,  'BATTERY',     'Pin',                  0, 0.10000, 1,  1),
    (2,  'CAMERA',      'Camera',               1, 0.10000, 2,  1),
    (3,  'DESIGN',      'Thiết kế',             2, 0.10000, 3,  1),
    (4,  'FEATURES',    'Tính năng',            3, 0.10000, 4,  1),
    (5,  'GENERAL',     'Tổng quát',            4, 0.10000, 5,  1),
    (6,  'PERFORMANCE', 'Hiệu năng',            5, 0.10000, 6,  1),
    (7,  'PRICE',       'Giá',                  6, 0.10000, 7,  1),
    (8,  'SCREEN',      'Màn hình',             7, 0.10000, 8,  1),
    (9,  'SER&ACC',     'Dịch vụ & Phụ kiện',   8, 0.10000, 9,  1),
    (10, 'STORAGE',     'Lưu trữ',              9, 0.10000, 10, 1)
ON DUPLICATE KEY UPDATE
    code          = VALUES(code),
    name_vi       = VALUES(name_vi),
    model_index   = VALUES(model_index),
    weight        = VALUES(weight),
    display_order = VALUES(display_order),
    is_active     = VALUES(is_active);

-- 4 nhãn cảm xúc, giữ đúng thứ tự đầu ra của mô hình.
-- is_mention = 0 cho None.
INSERT INTO sentiment_labels
    (id, code, name_vi, model_index, is_mention, display_order)
VALUES
    (1, 'None',     'Không đề cập', 0, 0, 0),
    (2, 'Negative', 'Tiêu cực',     1, 1, 1),
    (3, 'Neutral',  'Trung tính',   2, 1, 2),
    (4, 'Positive', 'Tích cực',     3, 1, 3)
ON DUPLICATE KEY UPDATE
    code          = VALUES(code),
    name_vi       = VALUES(name_vi),
    model_index   = VALUES(model_index),
    is_mention    = VALUES(is_mention),
    display_order = VALUES(display_order);

-- Tài khoản quản trị mặc định.
-- Username: admin
-- Password: Admin@123
--
-- Hash được tạo bằng bcrypt cost 12. Hãy đổi mật khẩu ngay sau lần đăng nhập đầu tiên.
-- Nếu đổi mật khẩu bằng PHP, dùng password_hash('mật_khẩu_mới', PASSWORD_DEFAULT)
-- rồi cập nhật cột password_hash.
INSERT INTO users
    (username, password_hash, full_name, role, is_active)
VALUES
    ('admin',
     '$2y$12$AKoNG0ZHuv3v0RGggiiXL.pKFOlX46EQo7jR6LoVJJDqMmzipgmbC',
     'Quản trị viên',
     'admin',
     1)
ON DUPLICATE KEY UPDATE
    username = VALUES(username);

COMMIT;

-- =====================================================================
-- KIỂM TRA NHANH SAU KHI IMPORT
-- =====================================================================
-- SELECT COUNT(*) AS so_khia_canh FROM aspects;              -- Kỳ vọng: 10
-- SELECT COUNT(*) AS so_nhan_cam_xuc FROM sentiment_labels;   -- Kỳ vọng: 4
-- SELECT id, code, model_index, weight FROM aspects ORDER BY model_index;
-- SELECT id, code, model_index, is_mention FROM sentiment_labels ORDER BY model_index;
-- SELECT id, username, role, is_active FROM users;
-- =====================================================================