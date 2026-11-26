<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';
require_login();

$pdo = db();
$user = current_user();
$errors = [];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    verify_csrf();

    $current = (string) ($_POST['current_password'] ?? '');
    $new = (string) ($_POST['new_password'] ?? '');
    $confirm = (string) ($_POST['confirm_password'] ?? '');

    $stmt = $pdo->prepare('SELECT password_hash FROM users WHERE id = ? LIMIT 1');
    $stmt->execute([(int) $user['id']]);
    $hash = (string) $stmt->fetchColumn();

    if ($current === '' || !password_verify($current, $hash)) {
        $errors[] = 'Mật khẩu hiện tại không đúng.';
    }

    $newLength = mb_strlen($new, 'UTF-8');
    if ($newLength < 8) {
        $errors[] = 'Mật khẩu mới phải có ít nhất 8 ký tự.';
    } elseif ($newLength > 72) {
        $errors[] = 'Mật khẩu mới tối đa 72 ký tự.';
    } elseif (password_verify($new, $hash)) {
        $errors[] = 'Mật khẩu mới phải khác mật khẩu hiện tại.';
    }

    if ($new !== $confirm) {
        $errors[] = 'Xác nhận mật khẩu không khớp.';
    }

    if ($errors === []) {
        $newHash = password_hash($new, PASSWORD_BCRYPT, ['cost' => 12]);
        $update = $pdo->prepare('UPDATE users SET password_hash = ? WHERE id = ?');
        $update->execute([$newHash, (int) $user['id']]);

        session_regenerate_id(true);
        flash('success', 'Đã đổi mật khẩu. Hãy dùng mật khẩu mới cho lần đăng nhập sau.');
        redirect('index.php');
    }
}

$pageTitle = 'Đổi mật khẩu';
$activeNav = 'password';
require __DIR__ . '/../app/views/layout_header.php';
?>
<div class="row g-3">
    <div class="col-lg-6">
        <div class="card">
            <div class="card-header">
                <div>
                    <h2>Đổi mật khẩu</h2>
                    <p>Tài khoản: <?= e((string) $user['username']) ?></p>
                </div>
            </div>
            <div class="card-body">
                <?php if ($errors !== []): ?>
                    <div class="alert alert-danger py-2 small">
                        <?php foreach ($errors as $error): ?>
                            <div><?= e($error) ?></div>
                        <?php endforeach; ?>
                    </div>
                <?php endif; ?>

                <form method="post" action="<?= e(url('change_password.php')) ?>" novalidate>
                    <?= csrf_field() ?>

                    <div class="mb-3">
                        <label class="form-label" for="current_password">Mật khẩu hiện tại</label>
                        <input class="form-control" type="password" id="current_password"
                               name="current_password" required autocomplete="current-password">
                    </div>

                    <div class="mb-3">
                        <label class="form-label" for="new_password">Mật khẩu mới</label>
                        <input class="form-control" type="password" id="new_password"
                               name="new_password" required minlength="8" maxlength="72"
                               autocomplete="new-password">
                        <div class="form-text">Từ 8 đến 72 ký tự.</div>
                    </div>

                    <div class="mb-4">
                        <label class="form-label" for="confirm_password">Nhập lại mật khẩu mới</label>
                        <input class="form-control" type="password" id="confirm_password"
                               name="confirm_password" required minlength="8" maxlength="72"
                               autocomplete="new-password">
                    </div>

                    <div class="d-flex gap-2">
                        <button type="submit" class="btn btn-primary">Cập nhật mật khẩu</button>
                        <a class="btn btn-outline-secondary" href="<?= e(url('index.php')) ?>">Hủy</a>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <div class="col-lg-6">
        <div class="card">
            <div class="card-header"><h2>Khuyến nghị bảo mật</h2></div>
            <div class="card-body small text-muted">
                <p class="mb-2">Tài khoản seed dùng mật khẩu công khai, chỉ phù hợp cho môi trường demo local.</p>
                <p class="mb-2">Mật khẩu được lưu bằng <code>password_hash()</code> với bcrypt cost 12.</p>
                <p class="mb-0">Không dùng lại mật khẩu của email hay tài khoản khác.</p>
            </div>
        </div>
    </div>
</div>

<?php require __DIR__ . '/../app/views/layout_footer.php'; ?>