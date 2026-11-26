<?php

declare(strict_types=1);

/**
 * Điểm khởi động chung cho mọi trang public/.
 * Nạp cấu hình, session, PDO và các helper.
 */

if (!defined('BRAND_HEALTH_APP')) {
    define('BRAND_HEALTH_APP', true);
}

/**
 * Đọc cấu hình theo khóa dạng "db.host".
 */
function config(?string $path = null, $default = null)
{
    static $config = null;

    if ($config === null) {
        $config = require __DIR__ . '/config.php';
    }

    if ($path === null || $path === '') {
        return $config;
    }

    $value = $config;
    foreach (explode('.', $path) as $key) {
        if (!is_array($value) || !array_key_exists($key, $value)) {
            return $default;
        }
        $value = $value[$key];
    }

    return $value;
}

date_default_timezone_set((string) config('app.timezone', 'Asia/Ho_Chi_Minh'));

error_reporting(E_ALL);
ini_set('display_errors', '0');
ini_set('log_errors', '1');

$logDir = dirname(__DIR__) . '/storage/logs';
if (!is_dir($logDir)) {
    @mkdir($logDir, 0775, true);
}
if (is_dir($logDir) && is_writable($logDir)) {
    ini_set('error_log', $logDir . '/php-error.log');
}

// Thư mục gốc URL, ví dụ "/brand_health_demo". Hoạt động cả khi đặt ở htdocs gốc.
if (!defined('BASE_URL')) {
    $scriptDir = str_replace('\\', '/', dirname((string) ($_SERVER['SCRIPT_NAME'] ?? '/')));
    $scriptDir = rtrim($scriptDir, '/');
    define('BASE_URL', $scriptDir === '' ? '' : $scriptDir);
}

if (session_status() !== PHP_SESSION_ACTIVE) {
    $secure = !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off';

    session_name('brandhealth_sid');
    session_set_cookie_params([
        'lifetime' => 0,
        'path'     => '/',
        'httponly' => true,
        'samesite' => 'Lax',
        'secure'   => $secure,
    ]);
    session_start();
}

if (!headers_sent()) {
    header('X-Content-Type-Options: nosniff');
    header('X-Frame-Options: SAMEORIGIN');
    header('Referrer-Policy: same-origin');
}

require_once __DIR__ . '/db.php';
require_once __DIR__ . '/helpers.php';
require_once __DIR__ . '/auth.php';
require_once __DIR__ . '/api_client.php';
require_once __DIR__ . '/analysis.php';
require_once __DIR__ . '/bhi.php';