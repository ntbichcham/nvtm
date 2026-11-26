<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';
require_login();

$data = bhi_aggregate();
$aspects = $data['aspects'];
$rating = bhi_rating($data['bhi']);

$pageTitle = 'Khía cạnh';
$activeNav = 'aspects';
require __DIR__ . '/../app/views/layout_header.php';
?>
<div class="stat-grid">
    <div class="stat-card accent">
        <div class="stat-label">BHI toàn thời gian</div>
        <div class="stat-value">
            <?= $data['bhi'] === null ? 'N/A' : e(format_score($data['bhi'])) ?><span class="stat-unit">/100</span>
        </div>
        <div class="stat-hint">
            <span class="rating-badge <?= e($rating['class']) ?>"><?= e($rating['label']) ?></span>
        </div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Độ phủ</div>
        <div class="stat-value"><?= (int) $data['coverage_count'] ?><span class="stat-unit">/<?= (int) $data['aspects_total'] ?></span></div>
        <div class="stat-hint"><?= e(format_score($data['coverage_percent'])) ?>% khía cạnh có dữ liệu</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Lượt khía cạnh</div>
        <div class="stat-value"><?= number_format((int) $data['total_mentions'], 0, ',', '.') ?></div>
        <div class="stat-hint">Từ <?= number_format((int) $data['analyzed_reviews'], 0, ',', '.') ?> đánh giá đã phân tích</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Trọng số đang dùng</div>
        <div class="stat-value"><?= e(number_format((float) $data['weight_used'], 2, ',', '.')) ?></div>
        <div class="stat-hint">Trên tổng trọng số 1,00 của 10 khía cạnh</div>
    </div>
</div>

<div class="card">
    <div class="card-header">
        <div>
            <h2>10 khía cạnh UIT-ViSFD</h2>
            <p>Giữ đúng thứ tự đầu ra của mô hình · Trọng số BHI cố định 0,10</p>
        </div>
    </div>

    <div class="card-body p-0">
        <div class="table-responsive">
            <table class="table align-middle">
                <thead>
                <tr>
                    <th style="width:60px">Model</th>
                    <th>Khía cạnh</th>
                    <th style="width:120px">Trọng số</th>
                    <th style="width:100px">Lượt nhắc</th>
                    <th style="width:170px">Điểm</th>
                    <th style="width:240px">Phân bố cảm xúc</th>
                </tr>
                </thead>
                <tbody>
                <?php foreach ($aspects as $aspect): ?>
                    <tr>
                        <td class="text-muted small">#<?= (int) $aspect['model_index'] ?></td>
                        <td>
                            <strong><?= e((string) $aspect['name_vi']) ?></strong>
                            <small class="text-muted d-block"><?= e((string) $aspect['code']) ?></small>
                        </td>
                        <td>
                            <div class="d-flex align-items-center gap-2">
                                <div class="weight-bar flex-grow-1">
                                    <span style="width: <?= e(number_format((float) $aspect['weight'] * 100, 1, '.', '')) ?>%"></span>
                                </div>
                                <small class="text-muted"><?= e(number_format((float) $aspect['weight'], 2, ',', '.')) ?></small>
                            </div>
                        </td>
                        <td><span class="fw-semibold"><?= number_format((int) $aspect['mentions'], 0, ',', '.') ?></span></td>
                        <td>
                            <?php if ($aspect['score'] === null): ?>
                                <span class="score-na">N/A</span>
                                <small class="text-muted d-block">Chưa có dữ liệu</small>
                            <?php else: ?>
                                <?php $score = (float) $aspect['score']; ?>
                                <div class="d-flex align-items-center gap-2">
                                    <div class="weight-bar flex-grow-1">
                                        <span class="score-fill <?= $score >= 70 ? 'good' : ($score >= 50 ? 'medium' : 'poor') ?>"
                                              style="width: <?= e(number_format($score, 1, '.', '')) ?>%"></span>
                                    </div>
                                    <small class="fw-semibold text-nowrap"><?= e(format_score($score)) ?></small>
                                </div>
                            <?php endif; ?>
                        </td>
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
                    </tr>
                <?php endforeach; ?>
                </tbody>
            </table>
        </div>
    </div>

    <div class="card-body border-top">
        <p class="small text-muted mb-0">
            Trọng số BHI được cố định 0,10 cho mỗi khía cạnh, không thay đổi khi thêm đánh giá.
            Cột <strong>Lượt nhắc</strong> và <strong>Điểm</strong> chỉ có dữ liệu sau khi đánh giá được phân tích bằng PhoBERT.
            Khía cạnh không có dữ liệu hiển thị N/A và bị loại khỏi công thức BHI; trọng số của các khía cạnh
            còn lại được chuẩn hóa lại.
        </p>
    </div>
</div>

<?php require __DIR__ . '/../app/views/layout_footer.php'; ?>