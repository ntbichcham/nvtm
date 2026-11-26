<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';
require_login();

$pdo = db();

$q = trim((string) ($_GET['q'] ?? ''));
$status = (string) ($_GET['status'] ?? '');
$allowedStatuses = ['pending', 'processing', 'completed', 'failed'];

if ($status !== '' && !in_array($status, $allowedStatuses, true)) {
    $status = '';
}
if (mb_strlen($q, 'UTF-8') > 100) {
    $q = mb_substr($q, 0, 100, 'UTF-8');
}

$page = max(1, query_int('page', 1));
$perPage = 10;

$where = [];
$params = [];

if ($q !== '') {
    $where[] = 'content LIKE ?';
    $params[] = like_pattern($q);
}
if ($status !== '') {
    $where[] = 'analysis_status = ?';
    $params[] = $status;
}

$whereSql = $where === [] ? '' : ' WHERE ' . implode(' AND ', $where);

$countStmt = $pdo->prepare('SELECT COUNT(*) FROM reviews' . $whereSql);
$countStmt->execute($params);
$total = (int) $countStmt->fetchColumn();

$totalPages = max(1, (int) ceil($total / $perPage));
if ($page > $totalPages) {
    $page = $totalPages;
}
$offset = ($page - 1) * $perPage;

$listStmt = $pdo->prepare(
    'SELECT id, content, rating, review_date, analysis_status, analyzed_at
     FROM reviews' . $whereSql . '
     ORDER BY review_date DESC, id DESC
     LIMIT ' . $perPage . ' OFFSET ' . $offset
);
$listStmt->execute($params);
$reviews = $listStmt->fetchAll();

$pageTitle = 'Đánh giá';
$activeNav = 'reviews';
$pendingAnalyzable = (int) $pdo->query(
    "SELECT COUNT(*) FROM reviews WHERE analysis_status IN ('pending', 'failed')"
)->fetchColumn();
require __DIR__ . '/../app/views/layout_header.php';

$queryBase = ['q' => $q, 'status' => $status];
?>
<div class="card">
    <div class="card-header flex-wrap">
        <div>
            <h2>Danh sách đánh giá</h2>
            <p><?= number_format($total, 0, ',', '.') ?> bản ghi<?= $q !== '' ? ' khớp từ khóa' : '' ?></p>
        </div>
        <div class="d-flex gap-2 align-items-center flex-wrap">
            <?php if ($pendingAnalyzable > 0): ?>
                <form method="post" action="<?= e(url('review_analyze_pending.php')) ?>" class="m-0"
                      data-confirm="Phân tích tối đa 20 đánh giá đang chờ bằng PhoBERT?">
                    <?= csrf_field() ?>
                    <button type="submit" class="btn btn-sm btn-outline-primary">
                        Phân tích đang chờ (<?= number_format($pendingAnalyzable, 0, ',', '.') ?>)
                    </button>
                </form>
            <?php endif; ?>
            <a class="btn btn-sm btn-primary" href="<?= e(url('review_create.php')) ?>">+ Thêm đánh giá</a>
        </div>
    </div>

    <div class="card-body border-bottom">
        <form method="get" action="<?= e(url('reviews.php')) ?>" class="filter-bar">
            <div>
                <label class="form-label mb-1" for="q">Tìm nội dung</label>
                <input class="form-control" type="search" id="q" name="q" value="<?= e($q) ?>"
                       placeholder="Nhập từ khóa..." maxlength="100">
            </div>
            <div>
                <label class="form-label mb-1" for="status">Trạng thái</label>
                <select class="form-select" id="status" name="status">
                    <option value="">Tất cả</option>
                    <?php foreach ($allowedStatuses as $option): ?>
                        <option value="<?= e($option) ?>"<?= $status === $option ? ' selected' : '' ?>>
                            <?= e(status_label($option)) ?>
                        </option>
                    <?php endforeach; ?>
                </select>
            </div>
            <div class="d-flex gap-2">
                <button class="btn btn-primary" type="submit">Lọc</button>
                <?php if ($q !== '' || $status !== ''): ?>
                    <a class="btn btn-outline-secondary" href="<?= e(url('reviews.php')) ?>">Xóa lọc</a>
                <?php endif; ?>
            </div>
        </form>
    </div>

    <div class="card-body p-0">
        <?php if ($reviews === []): ?>
            <div class="empty-state">
                <strong>Không tìm thấy đánh giá phù hợp</strong>
                Thử đổi từ khóa hoặc bỏ bộ lọc trạng thái.
            </div>
        <?php else: ?>
            <div class="table-responsive">
                <table class="table align-middle">
                    <thead>
                    <tr>
                        <th style="width:56px">#</th>
                        <th style="width:100px">Ngày</th>
                        <th>Nội dung</th>
                        <th style="width:80px">Sao</th>
                        <th style="width:140px">Trạng thái</th>
                        <th style="width:90px"></th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($reviews as $review): ?>
                        <tr>
                            <td class="text-muted small"><?= (int) $review['id'] ?></td>
                            <td class="text-nowrap text-muted small"><?= e(format_date((string) $review['review_date'])) ?></td>
                            <td class="review-text"><?= e(truncate((string) $review['content'], 150)) ?></td>
                            <td class="text-muted small"><?= $review['rating'] !== null ? (int) $review['rating'] : '—' ?></td>
                            <td><?= status_badge((string) $review['analysis_status']) ?></td>
                            <td class="text-end">
                                <a class="btn btn-sm btn-outline-primary"
                                   href="<?= e(url('review_view.php?id=' . (int) $review['id'])) ?>">Xem</a>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        <?php endif; ?>
    </div>

    <?php if ($totalPages > 1): ?>
        <div class="card-body pt-0">
            <nav aria-label="Phân trang">
                <ul class="pagination pagination-sm justify-content-center">
                    <?php for ($i = 1; $i <= $totalPages; $i++): ?>
                        <?php $link = 'reviews.php?' . http_build_query($queryBase + ['page' => $i]); ?>
                        <li class="page-item<?= $i === $page ? ' active' : '' ?>">
                            <a class="page-link" href="<?= e(url($link)) ?>"><?= $i ?></a>
                        </li>
                    <?php endfor; ?>
                </ul>
            </nav>
        </div>
    <?php endif; ?>
</div>

<?php require __DIR__ . '/../app/views/layout_footer.php'; ?>