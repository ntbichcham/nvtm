<?php

declare(strict_types=1);

/**
 * Cấu hình website.
 *
 * Không đặt mật khẩu thật hoặc API key vào file được commit công khai.
 * Với demo local, tài khoản MySQL mặc định của XAMPP là root / không mật khẩu.
 */

return [
    'app' => [
        'name'    => 'Brand Health Demo',
        'tagline' => 'Phân tích cảm xúc theo khía cạnh',
        'timezone' => 'Asia/Ho_Chi_Minh',
        'debug'   => false,
    ],

    'db' => [
        'host'    => '127.0.0.1',
        'port'    => 3306,
        'name'    => 'brand_health_db',
        'user'    => 'root',
        'pass'    => '',
        'charset' => 'utf8mb4',
    ],

    // Dùng ở Giai đoạn 5. Hiện tại chỉ hiển thị trạng thái cấu hình.
    'api' => [
        'base_url' => 'http://127.0.0.1:8000',
        'key'      => '',
        'timeout'  => 30,
    ],
];