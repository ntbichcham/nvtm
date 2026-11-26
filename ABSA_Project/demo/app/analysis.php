<?php

declare(strict_types=1);

/**
 * Điều phối một lần phân tích:
 *   1. Đánh dấu processing.
 *   2. Gọi Python API.
 *   3. Ghi analysis_results trong transaction.
 *   4. Cập nhật trạng thái review.
 */

/**
 * @return array{review_id: int, aspect_count: int, model_name: string, model_version: string, warnings: array<int, string>}
 *
 * @throws ApiException khi dịch vụ phân tích lỗi.
 * @throws RuntimeException khi dữ liệu nội bộ không hợp lệ.
 */
function analyze_review(int $reviewId): array
{
    $pdo = db();

    $stmt = $pdo->prepare('SELECT id, content FROM reviews WHERE id = ? LIMIT 1');
    $stmt->execute([$reviewId]);
    $review = $stmt->fetch();

    if (!$review) {
        throw new RuntimeException('Không tìm thấy đánh giá #' . $reviewId . '.');
    }

    $aspectMap = [];
    foreach ($pdo->query('SELECT code, id FROM aspects') as $row) {
        $aspectMap[(string) $row['code']] = (int) $row['id'];
    }

    $sentimentMap = [];
    foreach ($pdo->query('SELECT code, id FROM sentiment_labels') as $row) {
        $sentimentMap[(string) $row['code']] = (int) $row['id'];
    }

    if (count($aspectMap) !== 10 || count($sentimentMap) !== 4) {
        throw new RuntimeException('Danh mục aspects/sentiment_labels không đúng. Hãy import lại database.');
    }

    $pdo->prepare("UPDATE reviews SET analysis_status = 'processing', analysis_error = NULL WHERE id = ?")
        ->execute([$reviewId]);

    try {
        $result = api_predict((string) $review['content'], $reviewId);
    } catch (ApiException $e) {
        mark_analysis_failed($reviewId, $e->userMessage());
        throw $e;
    }

    // Ánh xạ code -> id; bỏ qua khía cạnh lạ và ghi log để không làm hỏng giao dịch.
    $rows = [];
    foreach ($result['aspects'] as $aspect) {
        if (!isset($aspectMap[$aspect['code']])) {
            error_log('Unknown aspect code from API: ' . $aspect['code']);
            continue;
        }
        if (!isset($sentimentMap[$aspect['sentiment']])) {
            error_log('Unknown sentiment code from API: ' . $aspect['sentiment']);
            continue;
        }

        $aspectId = $aspectMap[$aspect['code']];
        $rows[$aspectId] = [
            'review_id'  => $reviewId,
            'aspect_id'  => $aspectId,
            'sentiment_id' => $sentimentMap[$aspect['sentiment']],
            'confidence' => $aspect['confidence'],
        ];
    }

    $pdo->beginTransaction();
    try {
        $pdo->prepare('DELETE FROM analysis_results WHERE review_id = ?')->execute([$reviewId]);

        $insert = $pdo->prepare(
            'INSERT INTO analysis_results (review_id, aspect_id, sentiment_label_id, confidence)
             VALUES (?, ?, ?, ?)'
        );
        foreach ($rows as $row) {
            $insert->execute([
                $row['review_id'],
                $row['aspect_id'],
                $row['sentiment_id'],
                $row['confidence'],
            ]);
        }

        $update = $pdo->prepare(
            "UPDATE reviews
             SET analysis_status = 'completed',
                 analysis_error = NULL,
                 analyzed_at = NOW(),
                 model_name = ?,
                 model_version = ?,
                 inference_ms = ?
             WHERE id = ?"
        );
        $update->execute([
            $result['model_name'],
            $result['model_version'],
            $result['inference_ms'],
            $reviewId,
        ]);

        $pdo->commit();
    } catch (Throwable $e) {
        if ($pdo->inTransaction()) {
            $pdo->rollBack();
        }
        error_log('Failed to persist analysis for review ' . $reviewId . ': ' . $e->getMessage());
        mark_analysis_failed($reviewId, 'Không lưu được kết quả phân tích vào database.');
        throw new RuntimeException('Không lưu được kết quả phân tích.', 0, $e);
    }

    return [
        'review_id'     => $reviewId,
        'aspect_count'  => count($rows),
        'model_name'    => $result['model_name'],
        'model_version' => $result['model_version'],
        'warnings'      => $result['warnings'],
    ];
}

/**
 * Đánh dấu thất bại nhưng không xóa kết quả cũ đã có.
 */
function mark_analysis_failed(int $reviewId, string $safeMessage): void
{
    $pdo = db();
    $safeMessage = mb_substr($safeMessage, 0, 500, 'UTF-8');

    $check = $pdo->prepare('SELECT COUNT(*) FROM analysis_results WHERE review_id = ?');
    $check->execute([$reviewId]);
    $hasResults = (int) $check->fetchColumn() > 0;

    $status = $hasResults ? 'completed' : 'failed';

    $stmt = $pdo->prepare(
        'UPDATE reviews SET analysis_status = ?, analysis_error = ? WHERE id = ?'
    );
    $stmt->execute([$status, $safeMessage, $reviewId]);
}