<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';
require_login();

$pdo = db();
$errors = [];

$form = [
    'content'     => '',
    'rating'      => '',
    'review_date' => date('Y-m-d'),
    'source'      => 'manual',
];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    verify_csrf();

    $form['content'] = trim((string) ($_POST['content'] ?? ''));
    $form['rating'] = trim((string) ($_POST['rating'] ?? ''));
    $form['review_date'] = trim((string) ($_POST['review_date'] ?? ''));
    $form['source'] = 'manual';

    $length = mb_strlen($form['content'], 'UTF-8');
    if ($length === 0) {
        $errors['content'] = 'Nội dung đánh giá không được để trống.';
    } elseif ($length > 5000) {
        $errors['content'] = 'Nội dung tối đa 5.000 ký tự.';
    }

    if ($form['rating'] !== '' && !in_array($form['rating'], ['1', '2', '3', '4', '5'], true)) {
        $errors['rating'] = 'Số sao phải từ 1 đến 5.';
    }

    // Dấu "!" đặt giờ về 00:00:00 để so sánh ngày chính xác.
    $date = DateTime::createFromFormat('!Y-m-d', $form['review_date']);
    $today = new DateTime('today');
    if (!$date || $date->format('Y-m-d') !== $form['review_date']) {
        $errors['review_date'] = 'Ngày đánh giá không hợp lệ.';
    } elseif ($date > $today) {
        $errors['review_date'] = 'Ngày đánh giá không được ở tương lai.';
    }

    if ($errors === []) {
        $stmt = $pdo->prepare(
            'INSERT INTO reviews (source, content, rating, review_date, analysis_status, created_by)
             VALUES (?, ?, ?, ?, ?, ?)'
        );
        $stmt->execute([
            'manual',
            $form['content'],
            $form['rating'] === '' ? null : (int) $form['rating'],
            $form['review_date'],
            'pending',
            (int) current_user()['id'],
        ]);

        $newId = (int) $pdo->lastInsertId();
        flash('success', 'Đã thêm đánh giá #' . $newId . '. Đánh giá đang chờ phân tích.');
        redirect('review_view.php?id=' . $newId);
    }
}

$pageTitle = 'Thêm đánh giá';
$activeNav = 'create';
require __DIR__ . '/../app/views/layout_header.php';
?>
<div class="row g-3">
    <div class="col-lg-8">
        <div class="card">
            <div class="card-header">
                <div>
                    <h2>Nhập đánh giá mới</h2>
                    <p>Nội dung tiếng Việt, tối đa 5.000 ký tự</p>
                </div>
            </div>

            <div class="card-body">
                <?php if ($errors !== []): ?>
                    <div class="alert alert-danger py-2 small">
                        Vui lòng kiểm tra lại các trường được đánh dấu.
                    </div>
                <?php endif; ?>

                <form method="post" action="<?= e(url('review_create.php')) ?>" novalidate>
                    <?= csrf_field() ?>

                    <div class="mb-3">
                        <label class="form-label" for="content">Nội dung đánh giá</label>
                        <textarea class="form-control<?= isset($errors['content']) ? ' is-invalid' : '' ?>"
                                  id="content" name="content" rows="7" maxlength="5000" required
                                  data-counter="#contentCounter"
                                  placeholder="Ví dụ: Pin dùng rất lâu nhưng camera chụp ảnh không được đẹp, thiết kế khá sang trọng."><?= e($form['content']) ?></textarea>
                        <?php if (isset($errors['content'])): ?>
                            <div class="invalid-feedback"><?= e($errors['content']) ?></div>
                        <?php endif; ?>
                        <div class="form-text" id="contentCounter"></div>
                    </div>

                    <div class="row g-3">
                        <div class="col-sm-4">
                            <label class="form-label" for="rating">Số sao (tùy chọn)</label>
                            <select class="form-select<?= isset($errors['rating']) ? ' is-invalid' : '' ?>"
                                    id="rating" name="rating">
                                <option value="">Không có</option>
                                <?php for ($star = 1; $star <= 5; $star++): ?>
                                    <option value="<?= $star ?>"<?= $form['rating'] === (string) $star ? ' selected' : '' ?>>
                                        <?= $star ?> sao
                                    </option>
                                <?php endfor; ?>
                            </select>
                            <?php if (isset($errors['rating'])): ?>
                                <div class="invalid-feedback"><?= e($errors['rating']) ?></div>
                            <?php endif; ?>
                        </div>

                        <div class="col-sm-5">
                            <label class="form-label" for="review_date">Ngày đánh giá</label>
                            <input class="form-control<?= isset($errors['review_date']) ? ' is-invalid' : '' ?>"
                                   type="date" id="review_date" name="review_date"
                                   value="<?= e($form['review_date']) ?>" max="<?= e(date('Y-m-d')) ?>" required>
                            <?php if (isset($errors['review_date'])): ?>
                                <div class="invalid-feedback"><?= e($errors['review_date']) ?></div>
                            <?php endif; ?>
                        </div>

                        <div class="col-sm-3">
                            <label class="form-label">Nguồn</label>
                            <input class="form-control" type="text" value="Nhập tay" disabled>
                        </div>
                    </div>

                    <div class="d-flex gap-2 mt-4">
                        <button type="submit" class="btn btn-primary">Lưu đánh giá</button>
                        <a class="btn btn-outline-secondary" href="<?= e(url('reviews.php')) ?>">Hủy</a>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <div class="col-lg-4">
        <div class="card">
            <div class="card-header"><h2>Lưu ý</h2></div>
            <div class="card-body small text-muted">
                <p class="mb-2">Đánh giá được lưu với trạng thái <strong>Chờ phân tích</strong>.</p>
                <p class="mb-2">Số sao và ngày chỉ dùng để tra cứu, không được đưa vào mô hình PhoBERT.</p>
                <p class="mb-0">Mô hình chỉ đọc nội dung văn bản và trả về tối đa 10 khía cạnh UIT-ViSFD.</p>
            </div>
        </div>
    </div>
</div>

<?php require __DIR__ . '/../app/views/layout_footer.php'; ?>