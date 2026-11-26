<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';
require_login();

$pdo = db();
$id = query_int('id', 0);

$stmt = $pdo->prepare(
    'SELECT r.id, r.external_id, r.source, r.content, r.rating, r.review_date,
            r.analysis_status, r.analysis_error, r.analyzed_at, r.model_name,
            r.model_version, r.inference_ms, r.created_at,
            u.full_name AS created_by_name
     FROM reviews r
     LEFT JOIN users u ON u.id = r.created_by
     WHERE r.id = ?
     LIMIT 1'
);
$stmt->execute([$id]);
$review = $stmt->fetch();

if (!$review) {
    http_response_code(404);
    $pageTitle = 'Không tìm thấy';
    $activeNav = 'reviews';
    require __DIR__ . '/../app/views/layout_header.php';
    ?>
    <div class="card">
        <div class="card-body empty-state">
            <strong>Không tìm thấy đánh giá #<?= (int) $id ?></strong>
            Bản ghi có thể đã bị xóa.
            <div class="mt-3">
                <a class="btn btn-sm btn-primary" href="<?= e(url('reviews.php')) ?>">Về danh sách</a>
            </div>
        </div>
    </div>
    <?php
    require __DIR__ . '/../app/views/layout_footer.php';
    exit;
}

$resultStmt = $pdo->prepare(
    'SELECT ar.confidence, a.code AS aspect_code, a.name_vi AS aspect_name,
            sl.code AS sentiment_code, sl.name_vi AS sentiment_name
     FROM analysis_results ar
     JOIN aspects a ON a.id = ar.aspect_id
     JOIN sentiment_labels sl ON sl.id = ar.sentiment_label_id
     WHERE ar.review_id = ?
     ORDER BY a.display_order'
);
$resultStmt->execute([$id]);
$results = $resultStmt->fetchAll();

$pageTitle = 'Đánh giá #' . (int) $review['id'];
$activeNav = 'reviews';
require __DIR__ . '/../app/views/layout_header.php';
?>
<div class="page-head">
    <div>
        <p>Nguồn: <?= e((string) $review['source']) ?>
            <?php if ($review['external_id'] !== null): ?>
                · ID ngoài: <?= e((string) $review['external_id']) ?>
            <?php endif; ?>
        </p>
    </div>
    <div class="d-flex gap-2">
        <a class="btn btn-sm btn-outline-secondary" href="<?= e(url('reviews.php')) ?>">← Danh sách</a>
        <form method="post" action="<?= e(url('review_delete.php')) ?>" class="m-0"
              data-confirm="Xóa đánh giá #<?= (int) $review['id'] ?> và toàn bộ kết quả phân tích?">
            <?= csrf_field() ?>
            <input type="hidden" name="id" value="<?= (int) $review['id'] ?>">
            <button type="submit" class="btn btn-sm btn-outline-danger">Xóa</button>
        </form>
    </div>
</div>

<div class="detail-grid">
    <div class="card">
        <div class="card-header">
            <div>
                <h2>Nội dung đánh giá</h2>
                <p>Ngày <?= e(format_date((string) $review['review_date'])) ?>
                    <?php if ($review['rating'] !== null): ?>
                        · <?= (int) $review['rating'] ?> sao
                    <?php endif; ?>
                </p>
            </div>
            <?= status_badge((string) $review['analysis_status']) ?>
        </div>
        <div class="card-body">
            <div class="review-body"><?= e((string) $review['content']) ?></div>

            <div class="d-flex gap-2 mt-3 align-items-center flex-wrap">
                <form method="post" action="<?= e(url('review_analyze.php')) ?>" class="m-0"
                      <?= $review['analysis_status'] === 'completed' ? 'data-confirm="Phân tích lại sẽ thay thế kết quả hiện có. Tiếp tục?"' : '' ?>>
                    <?= csrf_field() ?>
                    <input type="hidden" name="id" value="<?= (int) $review['id'] ?>">
                    <button type="submit" class="btn btn-primary">
                        <?= $review['analysis_status'] === 'completed' ? 'Phân tích lại' : 'Phân tích bằng PhoBERT' ?>
                    </button>
                </form>
                <span class="small text-muted">
                    Gọi Python API tại <?= e(api_base_url()) ?>. Có thể mất vài giây.
                </span>
            </div>
        </div>
    </div>

    <div class="card">
        <div class="card-header"><h2>Thông tin</h2></div>
        <div class="card-body">
            <dl class="meta-list mb-0">
                <dt>Mã</dt>
                <dd>#<?= (int) $review['id'] ?></dd>
                <dt>Ngày đánh giá</dt>
                <dd><?= e(format_date((string) $review['review_date'])) ?></dd>
                <dt>Số sao</dt>
                <dd><?= $review['rating'] !== null ? (int) $review['rating'] . ' sao' : '—' ?></dd>
                <dt>Người nhập</dt>
                <dd><?= e((string) ($review['created_by_name'] ?? '—')) ?></dd>
                <dt>Ngày tạo</dt>
                <dd><?= e(format_datetime((string) $review['created_at'])) ?></dd>
                <dt>Phân tích lúc</dt>
                <dd><?= e(format_datetime($review['analyzed_at'] !== null ? (string) $review['analyzed_at'] : null)) ?></dd>
                <dt>Mô hình</dt>
                <dd><?= e((string) ($review['model_name'] ?? '—')) ?></dd>
                <dt>Phiên bản</dt>
                <dd><?= e((string) ($review['model_version'] ?? '—')) ?></dd>
                <dt>Thời gian</dt>
                <dd><?= $review['inference_ms'] !== null ? (int) $review['inference_ms'] . ' ms' : '—' ?></dd>
            </dl>

            <?php if ($review['analysis_error'] !== null): ?>
                <div class="alert alert-warning small mt-3 mb-0">
                    <?= e((string) $review['analysis_error']) ?>
                </div>
            <?php endif; ?>
        </div>
    </div>
</div>

<div class="card mt-3">
    <div class="card-header">
        <div>
            <h2>Kết quả phân tích theo khía cạnh</h2>
            <p>Chỉ hiển thị khía cạnh được mô hình dự đoán khác None</p>
        </div>
        <span class="badge text-bg-light"><?= count($results) ?> khía cạnh</span>
    </div>
    <div class="card-body p-0">
        <?php if ($results === []): ?>
            <div class="empty-state">
                <strong>Chưa có kết quả ABSA</strong>
                <?= $review['analysis_status'] === 'pending'
                    ? 'Đánh giá đang chờ được phân tích.'
                    : 'Chưa có khía cạnh nào được ghi nhận cho đánh giá này.' ?>
            </div>
        <?php else: ?>
            <div class="table-responsive">
                <table class="table align-middle">
                    <thead>
                    <tr>
                        <th style="width:60px">STT</th>
                        <th>Khía cạnh</th>
                        <th style="width:160px">Cảm xúc</th>
                        <th style="width:160px">Độ tin cậy</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($results as $index => $row): ?>
                        <?php $confidence = $row['confidence'] !== null ? (float) $row['confidence'] : null; ?>
                        <tr>
                            <td class="text-muted small"><?= $index + 1 ?></td>
                            <td>
                                <strong><?= e((string) $row['aspect_name']) ?></strong>
                                <small class="text-muted d-block"><?= e((string) $row['aspect_code']) ?></small>
                            </td>
                            <td><?= sentiment_badge((string) $row['sentiment_code'], (string) $row['sentiment_name']) ?></td>
                            <td>
                                <?php if ($confidence !== null): ?>
                                    <div class="d-flex align-items-center gap-2">
                                        <div class="weight-bar flex-grow-1">
                                            <span style="width: <?= e(number_format($confidence * 100, 1, '.', '')) ?>%"></span>
                                        </div>
                                        <small class="text-muted text-nowrap"><?= e(percent($confidence * 100, 1)) ?></small>
                                    </div>
                                <?php else: ?>
                                    <span class="text-muted">—</span>
                                <?php endif; ?>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        <?php endif; ?>
    </div>
</div>

<?php require __DIR__ . '/../app/views/layout_footer.php'; ?>