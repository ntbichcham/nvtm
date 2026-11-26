<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';
require_login();

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    redirect('reviews.php');
}

verify_csrf();

// Chỉ xử lý theo lô nhỏ để tránh treo request quá lâu.
set_time_limit(180);
$limit = 20;

$stmt = db()->prepare(
    "SELECT id FROM reviews
     WHERE analysis_status IN ('pending', 'failed')
     ORDER BY review_date ASC, id ASC
     LIMIT " . $limit
);
$stmt->execute();
$ids = array_map('intval', $stmt->fetchAll(PDO::FETCH_COLUMN));

if ($ids === []) {
    flash('info', 'Không có đánh giá nào đang chờ phân tích.');
    redirect('reviews.php');
}

$success = 0;
$failed = 0;
$aspectTotal = 0;
$lastError = '';

foreach ($ids as $id) {
    try {
        $summary = analyze_review($id);
        $success++;
        $aspectTotal += $summary['aspect_count'];
    } catch (ApiException $e) {
        $failed++;
        $lastError = $e->userMessage();
        // API không sẵn sàng thì dừng sớm, tránh lặp lỗi 20 lần.
        if ($e->serviceUnavailable()) {
            break;
        }
    } catch (Throwable $e) {
        $failed++;
        $lastError = 'Lỗi khi lưu kết quả phân tích.';
        error_log('Batch analyze failed for review ' . $id . ': ' . $e->getMessage());
    }
}

$message = 'Đã xử lý ' . ($success + $failed) . ' đánh giá: '
    . $success . ' thành công, ' . $failed . ' lỗi, '
    . $aspectTotal . ' lượt khía cạnh.';
flash($failed === 0 ? 'success' : 'warning', $message);

if ($lastError !== '') {
    flash('danger', $lastError);
}

redirect('reviews.php');