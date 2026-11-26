<?php

declare(strict_types=1);

/**
 * Các hàm tiện ích dùng chung.
 */

function e($value): string
{
    return htmlspecialchars((string) $value, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function url(string $path = ''): string
{
    if ($path === '') {
        return BASE_URL === '' ? '/' : BASE_URL . '/';
    }

    if (preg_match('#^https?://#i', $path)) {
        return $path;
    }

    return BASE_URL . '/' . ltrim($path, '/');
}

function asset(string $path): string
{
    return url('assets/' . ltrim($path, '/'));
}

function redirect(string $path): void
{
    header('Location: ' . url($path));
    exit;
}

function flash(string $type, string $message): void
{
    $_SESSION['_flash'][] = ['type' => $type, 'message' => $message];
}

function take_flashes(): array
{
    $flashes = $_SESSION['_flash'] ?? [];
    unset($_SESSION['_flash']);

    return is_array($flashes) ? $flashes : [];
}

function csrf_token(): string
{
    if (empty($_SESSION['_csrf']) || !is_string($_SESSION['_csrf'])) {
        $_SESSION['_csrf'] = bin2hex(random_bytes(32));
    }

    return $_SESSION['_csrf'];
}

function csrf_field(): string
{
    return '<input type="hidden" name="_csrf" value="' . e(csrf_token()) . '">';
}

function verify_csrf(): void
{
    $token = $_POST['_csrf'] ?? '';

    if (!is_string($token) || $token === '' || !hash_equals(csrf_token(), $token)) {
        http_response_code(403);
        header('Content-Type: text/html; charset=utf-8');
        echo '<!doctype html><html lang="vi"><meta charset="utf-8">'
            . '<title>Phiên đã hết hạn</title>'
            . '<body style="font-family:system-ui;padding:2rem">'
            . '<h1>Phiên làm việc đã hết hạn</h1>'
            . '<p>Vui lòng quay lại và thử lại.</p>'
            . '<p><a href="' . e(url()) . '">Về trang tổng quan</a></p></body></html>';
        exit;
    }
}

function format_date(?string $date, string $fallback = '—'): string
{
    if ($date === null || $date === '') {
        return $fallback;
    }

    $ts = strtotime($date);
    return $ts === false ? $fallback : date('d/m/Y', $ts);
}

function format_datetime(?string $datetime, string $fallback = '—'): string
{
    if ($datetime === null || $datetime === '') {
        return $fallback;
    }

    $ts = strtotime($datetime);
    return $ts === false ? $fallback : date('d/m/Y H:i', $ts);
}

function truncate(string $text, int $limit = 160): string
{
    $text = trim(preg_replace('/\s+/u', ' ', $text) ?? '');
    if (mb_strlen($text, 'UTF-8') <= $limit) {
        return $text;
    }

    return mb_substr($text, 0, $limit - 1, 'UTF-8') . '…';
}

function status_label(string $status): string
{
    return [
        'pending'    => 'Chờ phân tích',
        'processing' => 'Đang phân tích',
        'completed'  => 'Đã phân tích',
        'failed'     => 'Lỗi',
    ][$status] ?? $status;
}

function status_badge(string $status): string
{
    $class = [
        'pending'    => 'text-bg-secondary',
        'processing' => 'text-bg-info',
        'completed'  => 'text-bg-success',
        'failed'     => 'text-bg-danger',
    ][$status] ?? 'text-bg-secondary';

    return '<span class="badge ' . $class . '">' . e(status_label($status)) . '</span>';
}

function sentiment_badge(string $code, ?string $nameVi = null): string
{
    $class = [
        'Positive' => 'badge-sentiment positive',
        'Neutral'  => 'badge-sentiment neutral',
        'Negative' => 'badge-sentiment negative',
        'None'     => 'badge-sentiment none',
    ][$code] ?? 'badge-sentiment none';

    $label = $nameVi ?? [
        'Positive' => 'Tích cực',
        'Neutral'  => 'Trung tính',
        'Negative' => 'Tiêu cực',
        'None'     => 'Không đề cập',
    ][$code] ?? $code;

    return '<span class="' . $class . '">' . e($label) . '</span>';
}

function percent(float $value, int $decimals = 1): string
{
    return number_format($value, $decimals, ',', '.') . '%';
}

/**
 * Chuẩn hóa chuỗi tìm kiếm cho LIKE, tránh lỗi ký tự đại diện.
 */
function like_pattern(string $term): string
{
    return '%' . str_replace(['\\', '%', '_'], ['\\\\', '\\%', '\\_'], $term) . '%';
}

function query_int(string $key, int $default = 0): int
{
    $value = $_GET[$key] ?? null;

    return is_string($value) && $value !== '' ? (int) $value : $default;
}

function is_valid_date(string $value): bool
{
    $date = DateTime::createFromFormat('!Y-m-d', $value);

    return $date !== false && $date->format('Y-m-d') === $value;
}

function format_score(?float $score, int $decimals = 1): string
{
    return $score === null ? 'N/A' : number_format($score, $decimals, ',', '.');
}