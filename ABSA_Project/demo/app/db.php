<?php

declare(strict_types=1);

/**
 * Kết nối PDO dùng chung.
 */
function db(): PDO
{
    static $pdo = null;

    if ($pdo instanceof PDO) {
        return $pdo;
    }

    $c = config('db');
    $dsn = sprintf(
        'mysql:host=%s;port=%d;dbname=%s;charset=%s',
        $c['host'],
        (int) $c['port'],
        $c['name'],
        $c['charset']
    );

    try {
        $pdo = new PDO($dsn, (string) $c['user'], (string) $c['pass'], [
            PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES   => false,
            PDO::ATTR_STRINGIFY_FETCHES  => false,
        ]);
        $pdo->exec("SET time_zone = '+07:00'");
    } catch (PDOException $e) {
        error_log('DB connection failed: ' . $e->getMessage());
        http_response_code(500);
        header('Content-Type: text/html; charset=utf-8');
        echo '<!doctype html><html lang="vi"><meta charset="utf-8">'
            . '<title>Lỗi cơ sở dữ liệu</title>'
            . '<body style="font-family:system-ui;padding:2rem">'
            . '<h1>Không kết nối được cơ sở dữ liệu</h1>'
            . '<p>Hãy chắc chắn MySQL của XAMPP đang chạy và database '
            . '<code>brand_health_db</code> đã được import.</p>'
            . '<p>Chi tiết kỹ thuật đã được ghi vào log.</p></body></html>';
        exit;
    }

    return $pdo;
}