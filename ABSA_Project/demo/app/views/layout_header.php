<?php
/**
 * Phần đầu trang, sidebar và thông báo flash.
 *
 * Biến mong đợi:
 *   $pageTitle  string
 *   $activeNav  string  (dashboard|reviews|create|aspects|password)
 */

$user = current_user();
$flashes = take_flashes();
$pageTitle = $pageTitle ?? 'Tổng quan';
$activeNav = $activeNav ?? '';

$navItems = [
    ['key' => 'dashboard', 'href' => 'index.php',        'label' => 'Tổng quan'],
    ['key' => 'reviews',   'href' => 'reviews.php',      'label' => 'Đánh giá'],
    ['key' => 'create',    'href' => 'review_create.php', 'label' => 'Thêm đánh giá'],
    ['key' => 'aspects',   'href' => 'aspects.php',      'label' => 'Khía cạnh'],
];
?>
<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title><?= e($pageTitle) ?> · <?= e((string) config('app.name')) ?></title>
    <link rel="stylesheet" href="<?= e(asset('vendor/bootstrap/bootstrap.min.css')) ?>">
    <link rel="stylesheet" href="<?= e(asset('css/app.css')) ?>">
</head>
<body>
<div class="app-shell">
    <aside class="sidebar">
        <a class="brand" href="<?= e(url('index.php')) ?>">
            <span class="brand-mark">BH</span>
            <span class="brand-text">
                <strong>Brand Health</strong>
                <small><?= e((string) config('app.tagline')) ?></small>
            </span>
        </a>

        <nav class="sidebar-nav">
            <?php foreach ($navItems as $item): ?>
                <a class="nav-link<?= $activeNav === $item['key'] ? ' active' : '' ?>"
                   href="<?= e(url($item['href'])) ?>">
                    <span class="nav-dot"></span><?= e($item['label']) ?>
                </a>
            <?php endforeach; ?>
        </nav>

        <div class="sidebar-foot">
            <div class="model-chip">
                <span class="model-dot"></span>
                <div>
                    <strong>PhoBERT-ABSA</strong>
                    <small>UIT-ViSFD · 10 khía cạnh</small>
                </div>
            </div>
            <a class="nav-link small-link<?= $activeNav === 'password' ? ' active' : '' ?>"
               href="<?= e(url('change_password.php')) ?>">Đổi mật khẩu</a>
        </div>
    </aside>

    <div class="app-main">
        <header class="topbar">
            <div class="topbar-title">
                <h1><?= e($pageTitle) ?></h1>
                <p>Phân tích cảm xúc theo khía cạnh phục vụ theo dõi sức khỏe thương hiệu</p>
            </div>

            <div class="topbar-user">
                <div class="user-meta">
                    <strong><?= e($user['full_name'] ?? '') ?></strong>
                    <small><?= e($user['role'] ?? '') ?> · <?= e($user['username'] ?? '') ?></small>
                </div>
                <form method="post" action="<?= e(url('logout.php')) ?>" class="m-0">
                    <?= csrf_field() ?>
                    <button type="submit" class="btn btn-sm btn-outline-secondary">Đăng xuất</button>
                </form>
            </div>
        </header>

        <main class="content">
            <?php foreach ($flashes as $flash): ?>
                <div class="alert alert-<?= e($flash['type'] ?? 'info') ?> alert-dismissible fade show" role="alert">
                    <?= e($flash['message'] ?? '') ?>
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Đóng"></button>
                </div>
            <?php endforeach; ?>