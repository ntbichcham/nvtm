<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';
require_login();

$pdo = db();

// ---------------------------------------------------------------------
// Bộ lọc thời gian
// ---------------------------------------------------------------------
$fromRaw = trim((string) ($_GET['from'] ?? ''));
$toRaw = trim((string) ($_GET['to'] ?? ''));
$filterError = '';

if ($fromRaw !== '' && !is_valid_date($fromRaw)) {
    $filterError = 'Ngày bắt đầu không hợp lệ, đã bỏ qua bộ lọc.';
}
if ($toRaw !== '' && !is_valid_date($toRaw)) {
    $filterError = 'Ngày kết thúc không hợp lệ, đã bỏ qua bộ lọc.';
}

$from = is_valid_date($fromRaw) ? $fromRaw : '';
$to = is_valid_date($toRaw) ? $toRaw : '';

if ($from !== '' && $to !== '' && $from > $to) {
    $filterError = 'Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc.';
    $from = '';
    $to = '';
}

$where = [];
$params = [];
if ($from !== '') {
    $where[] = 'r.review_date >= ?';
    $params[] = $from;
}
if ($to !== '') {
    $where[] = 'r.review_date <= ?';
    $params[] = $to;
}
$whereSql = $where === [] ? '' : ' WHERE ' . implode(' AND ', $where);

$data = bhi_aggregate($from !== '' ? $from : null, $to !== '' ? $to : null);

$pendingStmt = $pdo->prepare(
    "SELECT COUNT(*) FROM reviews r" . $whereSql
    . ($whereSql === '' ? ' WHERE' : ' AND')
    . " r.analysis_status IN ('pending', 'processing', 'failed')"
);
$pendingStmt->execute($params);
$pendingCount = (int) $pendingStmt->fetchColumn();

$recentStmt = $pdo->prepare(
    'SELECT id, content, rating, review_date, analysis_status, analyzed_at
     FROM reviews r' . $whereSql . '
     ORDER BY review_date DESC, id DESC
     LIMIT 8'
);
$recentStmt->execute($params);
$recent = $recentStmt->fetchAll();

$rating = bhi_rating($data['bhi']);
$coveredAspects = array_values(array_filter(
    $data['aspects'],
    static fn (array $aspect): bool => (bool) $aspect['covered']
));

if ($from !== '' && $to !== '') {
    $rangeLabel = 'Từ ' . format_date($from) . ' đến ' . format_date($to);
} elseif ($from !== '') {
    $rangeLabel = 'Từ ' . format_date($from) . ' đến nay';
} elseif ($to !== '') {
    $rangeLabel = 'Đến ' . format_date($to);
} else {
    $rangeLabel = 'Tất cả thời gian';
}

$presets = [
    ['label' => '7 ngày',  'from' => date('Y-m-d', strtotime('-6 days'))],
    ['label' => '30 ngày', 'from' => date('Y-m-d', strtotime('-29 days'))],
    ['label' => '90 ngày', 'from' => date('Y-m-d', strtotime('-89 days'))],
    ['label' => 'Tất cả',  'from' => ''],
];

$chartData = [
    'aspects' => [
        'labels' => array_map(static fn (array $a): string => (string) $a['name_vi'], $coveredAspects),
        'scores' => array_map(static fn (array $a): float => (float) $a['score'], $coveredAspects),
    ],
    'sentiment' => [
        'labels' => ['Tích cực', 'Trung tính', 'Tiêu cực'],
        'values' => [
            $data['sentiment_counts']['Positive'],
            $data['sentiment_counts']['Neutral'],
            $data['sentiment_counts']['Negative'],
        ],
        'colors' => ['#10b981', '#94a3b8', '#ef4444'],
    ],
    'trend' => [
        'labels' => array_map(static fn (array $m): string => (string) $m['label'], $data['monthly']),
        'values' => array_map(static fn (array $m) => $m['bhi'], $data['monthly']),
    ],
];

$withCharts = true;
$pageTitle = 'Tổng quan';
$activeNav = 'dashboard';
require __DIR__ . '/../app/views/layout_header.php';
?>

<?php if ($filterError !== ''): ?>
    <div class="alert alert-warning py-2 small"><?= e($filterError) ?></div>
<?php endif; ?>

<div class="card mb-3">
    <div class="card-body filter-card">
        <form method="get" action="<?= e(url('index.php')) ?>" class="filter-bar">
            <div>
                <label class="form-label mb-1" for="from">Từ ngày</label>
                <input class="form-control" type="date" id="from" name="from" value="<?= e($from) ?>">
            </div>
            <div>
                <label class="form-label mb-1" for="to">Đến ngày</label>
                <input class="form-control" type="date" id="to" name="to" value="<?= e($to) ?>">
            </div>
            <div class="d-flex gap-2">
                <button class="btn btn-primary" type="submit">Áp dụng</button>
                <a class="btn btn-outline-secondary" href="<?= e(url('index.php')) ?>">Xóa lọc</a>
            </div>
            <div class="filter-presets">
                <?php foreach ($presets as $preset): ?>
                    <?php
                    $isActive = $preset['from'] === ''
                        ? ($from === '' && $to === '')
                        : ($from === $preset['from'] && $to === '');
                    $href = $preset['from'] === ''
                        ? url('index.php')
                        : url('index.php?from=' . urlencode($preset['from']));
                    ?>
                    <a class="preset-chip<?= $isActive ? ' active' : '' ?>" href="<?= e($href) ?>">
                        <?= e($preset['label']) ?>
                    </a>
                <?php endforeach; ?>
            </div>
        </form>
        <div class="small text-muted mt-2"><?= e($rangeLabel) ?></div>
    </div>
</div>

<div class="stat-grid">
    <div class="stat-card accent bhi-card">
        <div class="stat-label">Brand Health Index</div>
        <div class="stat-value">
            <?= $data['bhi'] === null ? 'N/A' : e(format_score($data['bhi'])) ?><span class="stat-unit">/100</span>
        </div>
        <div class="stat-hint">
            <span class="rating-badge <?= e($rating['class']) ?>"><?= e($rating['label']) ?></span>
            · <?= (int) $data['coverage_count'] ?>/<?= (int) $data['aspects_total'] ?> khía cạnh có dữ liệu
        </div>
    </div>

    <div class="stat-card">
        <div class="stat-label">Độ phủ dữ liệu</div>
        <div class="stat-value"><?= (int) $data['coverage_count'] ?><span class="stat-unit">/<?= (int) $data['aspects_total'] ?></span></div>
        <div class="stat-hint"><?= e(format_score($data['coverage_percent'])) ?>% khía cạnh · trọng số dùng <?= e(number_format((float) $data['weight_used'], 2, ',', '.')) ?></div>
    </div>

    <div class="stat-card">
        <div class="stat-label">Đã phân tích</div>
        <div class="stat-value"><?= number_format((int) $data['analyzed_reviews'], 0, ',', '.') ?></div>
        <div class="stat-hint">trên <?= number_format((int) $data['total_reviews'], 0, ',', '.') ?> đánh giá
            <?= $pendingCount > 0 ? ' · ' . number_format($pendingCount, 0, ',', '.') . ' chờ' : '' ?></div>
    </div>

    <div class="stat-card">
        <div class="stat-label">Lượt khía cạnh</div>
        <div class="stat-value"><?= number_format((int) $data['total_mentions'], 0, ',', '.') ?></div>
        <div class="stat-hint">
            <?= (int) $data['sentiment_counts']['Positive'] ?> tích cực ·
            <?= (int) $data['sentiment_counts']['Neutral'] ?> trung tính ·
            <?= (int) $data['sentiment_counts']['Negative'] ?> tiêu cực
        </div>
    </div>
</div>

<div class="note-strip">
    <span>i</span>
    <div>
        <strong>Cách tính:</strong> Tích cực = 1,0; Trung tính = 0,5; Tiêu cực = 0.
        Điểm khía cạnh là trung bình các lượt nhắc × 100.
        BHI là trung bình có trọng số trên các khía cạnh <em>có dữ liệu</em>, trọng số được chuẩn hóa lại;
        khía cạnh không có dữ liệu hiển thị <strong>N/A</strong> và không tham gia vào BHI.
    </div>
</div>

<div class="row g-3">
    <div class="col-lg-7">
        <div class="card h-100">
            <div class="card-header">
                <div>
                    <h2>Điểm theo khía cạnh</h2>
                    <p>Thang 0–100 · chỉ hiển thị khía cạnh có dữ liệu</p>
                </div>
                <span class="badge text-bg-light"><?= (int) $data['coverage_count'] ?>/<?= (int) $data['aspects_total'] ?></span>
            </div>
            <div class="card-body">
                <?php if ($coveredAspects === []): ?>
                    <div class="empty-state py-4">
                        <strong>Chưa có dữ liệu khía cạnh</strong>
                        Hãy phân tích ít nhất một đánh giá bằng PhoBERT.
                    </div>
                <?php else: ?>
                    <div class="chart-box"><canvas id="aspectChart" aria-label="Điểm theo khía cạnh"></canvas></div>
                <?php endif; ?>
            </div>
        </div>
    </div>

    <div class="col-lg-5">
        <div class="card h-100">
            <div class="card-header">
                <div>
                    <h2>Phân bố cảm xúc</h2>
                    <p><?= number_format((int) $data['total_mentions'], 0, ',', '.') ?> lượt khía cạnh</p>
                </div>
            </div>
            <div class="card-body">
                <?php if ($data['total_mentions'] === 0): ?>
                    <div class="empty-state py-4">
                        <strong>Chưa có dữ liệu</strong>
                        Phân bố cảm xúc sẽ hiển thị sau khi có kết quả ABSA.
                    </div>
                <?php else: ?>
                    <div class="chart-box"><canvas id="sentimentChart" aria-label="Phân bố cảm xúc"></canvas></div>
                <?php endif; ?>
            </div>
        </div>
    </div>
</div>

<div class="card mt-3">
    <div class="card-header">
        <div>
            <h2>Xu hướng BHI theo tháng</h2>
            <p>Dựa trên ngày đánh giá · <?= count($data['monthly']) ?> tháng có dữ liệu</p>
        </div>
    </div>
    <div class="card-body">
        <?php if ($data['monthly'] === []): ?>
            <div class="empty-state py-4">
                <strong>Chưa có dữ liệu thời gian</strong>
                Phân tích đánh giá để bắt đầu theo dõi xu hướng.
            </div>
        <?php else: ?>
            <?php if (count($data['monthly']) < 2): ?>
                <div class="small text-muted mb-2">
                    Cần ít nhất 2 tháng dữ liệu để thấy xu hướng rõ ràng; hiện có <?= count($data['monthly']) ?> tháng.
                </div>
            <?php endif; ?>
            <div class="chart-box chart-box-sm"><canvas id="trendChart" aria-label="Xu hướng BHI theo tháng"></canvas></div>
        <?php endif; ?>
    </div>
</div>

<div class="card mt-3">
    <div class="card-header">
        <div>
            <h2>Chi tiết 10 khía cạnh</h2>
            <p>Khía cạnh thiếu dữ liệu hiển thị N/A và bị loại khỏi BHI</p>
        </div>
        <a class="btn btn-sm btn-outline-primary" href="<?= e(url('aspects.php')) ?>">Xem danh mục</a>
    </div>
    <div class="card-body p-0">
        <div class="table-responsive">
            <table class="table align-middle">
                <thead>
                <tr>
                    <th>Khía cạnh</th>
                    <th style="width:90px">Trọng số</th>
                    <th style="width:95px">Lượt nhắc</th>
                    <th style="width:220px">Phân bố cảm xúc</th>
                    <th style="width:180px">Điểm</th>
                </tr>
                </thead>
                <tbody>
                <?php foreach ($data['aspects'] as $aspect): ?>
                    <tr>
                        <td>
                            <strong><?= e((string) $aspect['name_vi']) ?></strong>
                            <small class="text-muted d-block"><?= e((string) $aspect['code']) ?></small>
                        </td>
                        <td class="text-muted small"><?= e(number_format((float) $aspect['weight'], 2, ',', '.')) ?></td>
                        <td><span class="fw-semibold"><?= number_format((int) $aspect['mentions'], 0, ',', '.') ?></span></td>
                        <td>
                            <?php if ((int) $aspect['mentions'] === 0): ?>
                                <span class="text-muted small">Chưa có dữ liệu</span>
                            <?php else: ?>
                                <div class="d-flex gap-1 flex-wrap">
                                    <span class="badge-sentiment positive"><?= (int) $aspect['positive'] ?></span>
                                    <span class="badge-sentiment neutral"><?= (int) $aspect['neutral'] ?></span>
                                    <span class="badge-sentiment negative"><?= (int) $aspect['negative'] ?></span>
                                </div>
                            <?php endif; ?>
                        </td>
                        <td>
                            <?php if ($aspect['score'] === null): ?>
                                <span class="score-na">N/A</span>
                            <?php else: ?>
                                <?php $score = (float) $aspect['score']; ?>
                                <div class="d-flex align-items-center gap-2">
                                    <div class="weight-bar flex-grow-1">
                                        <span class="score-fill <?= $score >= 70 ? 'good' : ($score >= 50 ? 'medium' : 'poor') ?>"
                                              style="width: <?= e(number_format($score, 1, '.', '')) ?>%"></span>
                                    </div>
                                    <small class="text-nowrap fw-semibold"><?= e(format_score($score)) ?></small>
                                </div>
                            <?php endif; ?>
                        </td>
                    </tr>
                <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>
</div>

<div class="card mt-3">
    <div class="card-header">
        <div>
            <h2>Đánh giá gần đây</h2>
            <p>8 bản ghi mới nhất trong khoảng đang chọn</p>
        </div>
        <a class="btn btn-sm btn-outline-primary" href="<?= e(url('reviews.php')) ?>">Xem tất cả</a>
    </div>
    <div class="card-body p-0">
        <?php if ($recent === []): ?>
            <div class="empty-state">
                <strong>Chưa có đánh giá nào</strong>
                Thêm đánh giá đầu tiên để bắt đầu theo dõi.
                <div class="mt-3">
                    <a class="btn btn-sm btn-primary" href="<?= e(url('review_create.php')) ?>">Thêm đánh giá</a>
                </div>
            </div>
        <?php else: ?>
            <div class="table-responsive">
                <table class="table align-middle">
                    <thead>
                    <tr>
                        <th style="width:90px">Ngày</th>
                        <th>Nội dung</th>
                        <th style="width:130px">Trạng thái</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($recent as $review): ?>
                        <tr>
                            <td class="text-nowrap text-muted small"><?= e(format_date((string) $review['review_date'])) ?></td>
                            <td class="review-text">
                                <a href="<?= e(url('review_view.php?id=' . (int) $review['id'])) ?>">
                                    <?= e(truncate((string) $review['content'], 110)) ?>
                                </a>
                                <?php if ($review['rating'] !== null): ?>
                                    <small class="d-block"><?= (int) $review['rating'] ?> sao</small>
                                <?php endif; ?>
                            </td>
                            <td><?= status_badge((string) $review['analysis_status']) ?></td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        <?php endif; ?>
    </div>
</div>

<?php if ($data['total_mentions'] > 0 || $data['monthly'] !== []): ?>
    <script type="application/json" id="bhi-chart-data"><?= json_encode($chartData, JSON_UNESCAPED_UNICODE | JSON_HEX_TAG | JSON_HEX_AMP | JSON_HEX_APOS | JSON_HEX_QUOT) ?></script>
<?php endif; ?>

<?php require __DIR__ . '/../app/views/layout_footer.php'; ?>