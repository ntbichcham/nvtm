<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';
require_login();

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    redirect('reviews.php');
}

verify_csrf();

$id = isset($_POST['id']) ? (int) $_POST['id'] : 0;
if ($id <= 0) {
    flash('danger', 'Mã đánh giá không hợp lệ.');
    redirect('reviews.php');
}

try {
    $summary = analyze_review($id);

    $message = 'Đã phân tích đánh giá #' . $id . ': '
        . $summary['aspect_count'] . ' khía cạnh được ghi nhận';
    if ($summary['model_version'] !== '') {
        $message .= ' · ' . $summary['model_version'];
    }
    $message .= '.';

    flash('success', $message);

    foreach (array_slice($summary['warnings'], 0, 2) as $warning) {
        flash('warning', $warning);
    }
} catch (ApiException $e) {
    flash('danger', $e->userMessage());
} catch (Throwable $e) {
    error_log('analyze_review failed for review ' . $id . ': ' . $e->getMessage());
    flash('danger', 'Không thể phân tích đánh giá #' . $id . '. Vui lòng thử lại.');
}

redirect('review_view.php?id=' . $id);