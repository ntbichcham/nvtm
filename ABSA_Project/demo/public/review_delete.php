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

$stmt = db()->prepare('DELETE FROM reviews WHERE id = ?');
$stmt->execute([$id]);

if ($stmt->rowCount() > 0) {
    flash('success', 'Đã xóa đánh giá #' . $id . '.');
} else {
    flash('warning', 'Không tìm thấy đánh giá #' . $id . '.');
}

redirect('reviews.php');