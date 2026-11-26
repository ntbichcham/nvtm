<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';

if (current_user() !== null) {
    redirect('index.php');
}

$errors = [];
$username = '';
$now = time();

$attempts = $_SESSION['_login_attempts'] ?? [];
$attempts = is_array($attempts) ? array_values(array_filter($attempts, static function ($ts) use ($now) {
    return is_int($ts) && $ts > $now - 300;
})) : [];
$_SESSION['_login_attempts'] = $attempts;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    verify_csrf();

    $username = trim((string) ($_POST['username'] ?? ''));
    $password = (string) ($_POST['password'] ?? '');

    if (count($attempts) >= 5) {
        $errors[] = 'Bạn đã thử đăng nhập quá nhiều lần. Vui lòng chờ khoảng 5 phút.';
    } elseif ($username === '' || $password === '') {
        $errors[] = 'Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.';
    } else {
        $user = attempt_login($username, $password);

        if ($user === null) {
            $_SESSION['_login_attempts'][] = $now;
            $errors[] = 'Tên đăng nhập hoặc mật khẩu không đúng.';
        } else {
            unset($_SESSION['_login_attempts']);
            login_user($user);
            flash('success', 'Đăng nhập thành công. Chào mừng ' . $user['full_name'] . '.');
            redirect('index.php');
        }
    }
}

$flashes = take_flashes();
?>
<!doctype html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Đăng nhập · <?= e((string) config('app.name')) ?></title>
    <link rel="stylesheet" href="<?= e(asset('vendor/bootstrap/bootstrap.min.css')) ?>">
    <link rel="stylesheet" href="<?= e(asset('css/app.css')) ?>">
</head>
<body>
<div class="login-wrap">
    <div class="login-card">
        <div class="login-brand">
            <span class="brand-mark">BH</span>
            <div>
                <strong><?= e((string) config('app.name')) ?></strong>
                <small><?= e((string) config('app.tagline')) ?></small>
            </div>
        </div>

        <h1 class="h5 mb-1">Đăng nhập quản trị</h1>
        <p class="text-muted small mb-4">Dùng tài khoản được tạo trong bảng <code>users</code>.</p>

        <?php foreach ($flashes as $flash): ?>
            <div class="alert alert-<?= e($flash['type'] ?? 'info') ?> py-2 small" role="alert">
                <?= e($flash['message'] ?? '') ?>
            </div>
        <?php endforeach; ?>

        <?php if ($errors !== []): ?>
            <div class="alert alert-danger py-2 small" role="alert">
                <?php foreach ($errors as $error): ?>
                    <div><?= e($error) ?></div>
                <?php endforeach; ?>
            </div>
        <?php endif; ?>

        <form method="post" action="<?= e(url('login.php')) ?>" autocomplete="on" novalidate>
            <?= csrf_field() ?>

            <div class="mb-3">
                <label class="form-label" for="username">Tên đăng nhập</label>
                <input class="form-control" type="text" id="username" name="username"
                       value="<?= e($username) ?>" maxlength="50" required autofocus
                       autocomplete="username">
            </div>

            <div class="mb-4">
                <label class="form-label" for="password">Mật khẩu</label>
                <input class="form-control" type="password" id="password" name="password"
                       required autocomplete="current-password">
            </div>

            <button type="submit" class="btn btn-primary w-100 py-2">Đăng nhập</button>
        </form>

        <p class="text-muted small mt-4 mb-0">
            Tài khoản mặc định sau khi import database: <strong>admin</strong>.
            Hãy đổi mật khẩu ngay sau lần đăng nhập đầu tiên.
        </p>
    </div>
</div>
<script src="<?= e(asset('vendor/bootstrap/bootstrap.bundle.min.js')) ?>"></script>
</body>
</html>