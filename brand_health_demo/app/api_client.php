<?php

declare(strict_types=1);

/**
 * Client gọi Python API PhoBERT ABSA.
 *
 * Không bao giờ trả nguyên văn lỗi kỹ thuật của API cho người dùng.
 * Chi tiết được ghi vào storage/logs/php-error.log.
 */

class ApiException extends RuntimeException
{
    private string $userMessage;
    private bool $serviceUnavailable;

    public function __construct(
        string $userMessage,
        string $logMessage = '',
        ?Throwable $previous = null,
        bool $serviceUnavailable = false
    ) {
        parent::__construct($logMessage !== '' ? $logMessage : $userMessage, 0, $previous);
        $this->userMessage = $userMessage;
        $this->serviceUnavailable = $serviceUnavailable;
    }

    public function userMessage(): string
    {
        return $this->userMessage;
    }

    public function serviceUnavailable(): bool
    {
        return $this->serviceUnavailable;
    }
}

function api_base_url(): string
{
    return rtrim((string) config('api.base_url', 'http://127.0.0.1:8000'), '/');
}

/**
 * Gửi request tới Python API và trả về mảng JSON đã giải mã.
 *
 * @throws ApiException
 */
function api_request(string $path, ?array $payload = null): array
{
    $url = api_base_url() . $path;
    $timeout = max(5, (int) config('api.timeout', 30));

    $ch = curl_init($url);
    if ($ch === false) {
        throw new ApiException('Không khởi tạo được kết nối tới dịch vụ phân tích.');
    }

    $headers = ['Accept: application/json'];

    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_CONNECTTIMEOUT => 5,
        CURLOPT_TIMEOUT        => $timeout + 5,
        CURLOPT_FOLLOWLOCATION => false,
        CURLOPT_NOPROXY        => '127.0.0.1,localhost',
        CURLOPT_HTTPHEADER     => $headers,
    ]);

    if ($payload !== null) {
        try {
            $json = json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR);
        } catch (JsonException $e) {
            throw new ApiException('Không chuẩn bị được dữ liệu gửi tới dịch vụ phân tích.', $e->getMessage(), $e);
        }

        $headers[] = 'Content-Type: application/json; charset=utf-8';
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $json);
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    }

    $apiKey = trim((string) config('api.key', ''));
    if ($apiKey !== '') {
        $headers[] = 'X-API-Key: ' . $apiKey;
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    }

    $raw = curl_exec($ch);
    $errno = curl_errno($ch);
    $error = curl_error($ch);
    $status = (int) curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
    curl_close($ch);

    if ($raw === false || $errno !== 0) {
        error_log(sprintf('API transport error | url=%s | errno=%d | %s', $url, $errno, $error));
        throw new ApiException(
            'Không kết nối được dịch vụ phân tích PhoBERT. '
            . 'Hãy chắc chắn Python API đang chạy tại ' . api_base_url() . '.',
            '',
            null,
            true
        );
    }

    $data = json_decode($raw, true);

    if ($status >= 400) {
        $detail = is_array($data) && isset($data['detail']) && is_string($data['detail'])
            ? $data['detail']
            : '';
        error_log(sprintf('API HTTP error | status=%d | detail=%s', $status, $detail));

        $message = match ($status) {
            400, 422 => 'Dịch vụ phân tích báo dữ liệu đánh giá không hợp lệ.',
            401       => 'Dịch vụ phân tích từ chối API key. Kiểm tra cấu hình.',
            503       => 'Mô hình PhoBERT chưa sẵn sàng. Kiểm tra cửa sổ Python API.',
            504       => 'Dịch vụ phân tích phản hồi quá lâu. Vui lòng thử lại.',
            default   => 'Dịch vụ phân tích trả về lỗi (HTTP ' . $status . ').',
        };

        throw new ApiException($message, '', null, $status === 503 || $status === 504);
    }

    if (!is_array($data)) {
        error_log('API returned invalid JSON: ' . mb_substr($raw, 0, 500, 'UTF-8'));
        throw new ApiException('Phản hồi của dịch vụ phân tích không hợp lệ.');
    }

    return $data;
}

/**
 * Gọi /predict và kiểm tra cấu trúc kết quả.
 *
 * @return array{aspects: array<int, array{code: string, sentiment: string, confidence: float|null}>, model_name: string, model_version: string, inference_ms: int|null, warnings: array<int, string>}
 *
 * @throws ApiException
 */
function api_predict(string $text, ?int $reviewId = null): array
{
    $data = api_request('/predict', [
        'text'                 => $text,
        'review_id'            => $reviewId,
        'include_all_aspects'  => false,
    ]);

    if (($data['success'] ?? false) !== true) {
        throw new ApiException('Dịch vụ phân tích không trả về kết quả thành công.');
    }

    $aspects = $data['aspects'] ?? null;
    if (!is_array($aspects)) {
        throw new ApiException('Phản hồi của dịch vụ phân tích thiếu danh sách khía cạnh.');
    }

    $clean = [];
    foreach ($aspects as $aspect) {
        if (!is_array($aspect)) {
            throw new ApiException('Phản hồi của dịch vụ phân tích có khía cạnh không hợp lệ.');
        }

        $code = trim((string) ($aspect['code'] ?? ''));
        $sentiment = trim((string) ($aspect['sentiment'] ?? ''));
        $confidence = $aspect['confidence'] ?? null;

        if ($code === '' || !in_array($sentiment, ['Negative', 'Neutral', 'Positive'], true)) {
            error_log('API invalid aspect entry: ' . json_encode($aspect, JSON_UNESCAPED_UNICODE));
            throw new ApiException('Phản hồi của dịch vụ phân tích có khía cạnh không hợp lệ.');
        }

        if ($confidence !== null && !is_numeric($confidence)) {
            $confidence = null;
        }

        $clean[] = [
            'code'       => $code,
            'sentiment'  => $sentiment,
            'confidence' => $confidence === null ? null : max(0.0, min(1.0, (float) $confidence)),
        ];
    }

    $model = is_array($data['model'] ?? null) ? $data['model'] : [];
    $warnings = is_array($data['warnings'] ?? null) ? $data['warnings'] : [];
    $inferenceMs = $data['inference_ms'] ?? null;

    return [
        'aspects'       => $clean,
        'model_name'    => mb_substr((string) ($model['name'] ?? 'PhoBERT-ABSA'), 0, 100, 'UTF-8'),
        'model_version' => mb_substr((string) ($model['version'] ?? 'unknown'), 0, 100, 'UTF-8'),
        'inference_ms'  => is_numeric($inferenceMs) ? (int) round((float) $inferenceMs) : null,
        'warnings'      => array_values(array_filter($warnings, 'is_string')),
    ];
}