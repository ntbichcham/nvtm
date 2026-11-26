<?php

declare(strict_types=1);

/**
 * Tính Brand Health Index (BHI) từ analysis_results.
 *
 * Quy ước:
 *   Positive = 1,0
 *   Neutral  = 0,5
 *   Negative = 0,0
 *
 * Điểm khía cạnh = trung bình điểm cảm xúc của các lượt nhắc × 100.
 * BHI = trung bình có trọng số trên các khía cạnh CÓ dữ liệu,
 *       trọng số được chuẩn hóa lại theo tổng trọng số của các khía cạnh đó.
 * Khía cạnh không có dữ liệu: điểm N/A, không tham gia vào BHI.
 */

function bhi_sentiment_value(string $sentiment): ?float
{
    return match ($sentiment) {
        'Positive' => 1.0,
        'Neutral'  => 0.5,
        'Negative' => 0.0,
        default    => null,
    };
}

/**
 * Tổng hợp dữ liệu BHI, có thể lọc theo khoảng ngày đánh giá.
 *
 * @return array{
 *   bhi: float|null,
 *   coverage_count: int,
 *   coverage_percent: float,
 *   aspects_total: int,
 *   aspects: array<int, array<string, mixed>>,
 *   missing: array<int, string>,
 *   sentiment_counts: array{Positive: int, Neutral: int, Negative: int},
 *   total_mentions: int,
 *   total_reviews: int,
 *   analyzed_reviews: int,
 *   weight_used: float,
 *   monthly: array<int, array<string, mixed>>
 * }
 */
function bhi_aggregate(?string $from = null, ?string $to = null): array
{
    $pdo = db();

    $aspectRows = $pdo->query(
        'SELECT id, code, name_vi, model_index, weight, display_order
         FROM aspects
         WHERE is_active = 1
         ORDER BY display_order'
    )->fetchAll();

    $aspects = [];
    foreach ($aspectRows as $row) {
        $aspects[(int) $row['id']] = [
            'id'            => (int) $row['id'],
            'code'          => (string) $row['code'],
            'name_vi'       => (string) $row['name_vi'],
            'model_index'   => (int) $row['model_index'],
            'weight'        => (float) $row['weight'],
            'display_order' => (int) $row['display_order'],
            'mentions'      => 0,
            'positive'      => 0,
            'neutral'       => 0,
            'negative'      => 0,
            'score'         => null,
            'covered'       => false,
        ];
    }

    $where = [];
    $params = [];

    if ($from !== null && $from !== '') {
        $where[] = 'r.review_date >= ?';
        $params[] = $from;
    }
    if ($to !== null && $to !== '') {
        $where[] = 'r.review_date <= ?';
        $params[] = $to;
    }

    $whereSql = $where === [] ? '' : ' WHERE ' . implode(' AND ', $where);

    $sql = 'SELECT ar.aspect_id, sl.code AS sentiment, r.review_date
            FROM analysis_results ar
            JOIN reviews r ON r.id = ar.review_id
            JOIN sentiment_labels sl ON sl.id = ar.sentiment_label_id'
            . $whereSql . '
            ORDER BY r.review_date ASC, ar.id ASC';

    $stmt = $pdo->prepare($sql);
    $stmt->execute($params);

    $monthlyMatrix = [];
    foreach ($stmt as $row) {
        $aspectId = (int) $row['aspect_id'];
        if (!isset($aspects[$aspectId])) {
            continue;
        }

        $key = strtolower((string) $row['sentiment']);
        if (!in_array($key, ['positive', 'neutral', 'negative'], true)) {
            continue;
        }

        $aspects[$aspectId]['mentions']++;
        $aspects[$aspectId][$key]++;

        $month = substr((string) $row['review_date'], 0, 7);
        if ($month !== '') {
            $monthlyMatrix[$month][$aspectId][$key] = ($monthlyMatrix[$month][$aspectId][$key] ?? 0) + 1;
        }
    }

    $sentimentCounts = ['Positive' => 0, 'Neutral' => 0, 'Negative' => 0];
    $missing = [];
    $covered = 0;
    $weightedSum = 0.0;
    $weightUsed = 0.0;

    foreach ($aspects as &$aspect) {
        $mentions = (int) $aspect['mentions'];

        if ($mentions > 0) {
            $score = (
                (int) $aspect['positive'] * 1.0
                + (int) $aspect['neutral'] * 0.5
                + (int) $aspect['negative'] * 0.0
            ) / $mentions * 100.0;

            $aspect['score'] = round($score, 2);
            $aspect['covered'] = true;
            $covered++;
            $weightedSum += (float) $aspect['weight'] * $score;
            $weightUsed += (float) $aspect['weight'];
        } else {
            $missing[] = (string) $aspect['name_vi'];
        }

        $sentimentCounts['Positive'] += (int) $aspect['positive'];
        $sentimentCounts['Neutral'] += (int) $aspect['neutral'];
        $sentimentCounts['Negative'] += (int) $aspect['negative'];
    }
    unset($aspect);

    $bhi = $weightUsed > 0 ? round($weightedSum / $weightUsed, 2) : null;
    $totalMentions = $sentimentCounts['Positive'] + $sentimentCounts['Neutral'] + $sentimentCounts['Negative'];

    $countStmt = $pdo->prepare('SELECT COUNT(*) FROM reviews r' . $whereSql);
    $countStmt->execute($params);
    $totalReviews = (int) $countStmt->fetchColumn();

    $analyzedStmt = $pdo->prepare(
        'SELECT COUNT(DISTINCT r.id)
         FROM reviews r
         JOIN analysis_results ar ON ar.review_id = r.id'
        . $whereSql
    );
    $analyzedStmt->execute($params);
    $analyzedReviews = (int) $analyzedStmt->fetchColumn();

    ksort($monthlyMatrix);
    $monthly = [];
    foreach ($monthlyMatrix as $month => $aspectCounts) {
        $monthWeighted = 0.0;
        $monthWeight = 0.0;
        $monthMentions = 0;
        $monthCovered = 0;

        foreach ($aspectCounts as $aspectId => $counts) {
            $mentions = (int) array_sum($counts);
            if ($mentions <= 0 || !isset($aspects[$aspectId])) {
                continue;
            }

            $score = (
                (int) ($counts['positive'] ?? 0) * 1.0
                + (int) ($counts['neutral'] ?? 0) * 0.5
            ) / $mentions * 100.0;

            $weight = (float) $aspects[$aspectId]['weight'];
            $monthWeighted += $weight * $score;
            $monthWeight += $weight;
            $monthMentions += $mentions;
            $monthCovered++;
        }

        $monthly[] = [
            'month'          => (string) $month,
            'label'          => substr((string) $month, 5, 2) . '/' . substr((string) $month, 0, 4),
            'bhi'            => $monthWeight > 0 ? round($monthWeighted / $monthWeight, 2) : null,
            'mentions'       => $monthMentions,
            'coverage_count' => $monthCovered,
        ];
    }

    $aspectsTotal = count($aspects);

    return [
        'bhi'               => $bhi,
        'coverage_count'    => $covered,
        'coverage_percent'  => $aspectsTotal > 0 ? round($covered / $aspectsTotal * 100, 1) : 0.0,
        'aspects_total'     => $aspectsTotal,
        'aspects'           => array_values($aspects),
        'missing'           => $missing,
        'sentiment_counts'  => $sentimentCounts,
        'total_mentions'    => $totalMentions,
        'total_reviews'     => $totalReviews,
        'analyzed_reviews'  => $analyzedReviews,
        'weight_used'       => round($weightUsed, 5),
        'monthly'           => $monthly,
    ];
}

/**
 * Xếp loại mức BHI để hiển thị.
 *
 * @return array{label: string, class: string}
 */
function bhi_rating(?float $score): array
{
    if ($score === null) {
        return ['label' => 'Chưa đủ dữ liệu', 'class' => 'na'];
    }
    if ($score >= 70) {
        return ['label' => 'Tốt', 'class' => 'good'];
    }
    if ($score >= 50) {
        return ['label' => 'Trung bình', 'class' => 'medium'];
    }
    return ['label' => 'Yếu', 'class' => 'poor'];
}