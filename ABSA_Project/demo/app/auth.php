<?php

declare(strict_types=1);

/**
 * Xác thực người dùng bằng PHP session + password_verify.
 */

function current_user(): ?array
{
    static $loaded = false;
    static $user = null;

    if ($loaded) {
        return $user;
    }
    $loaded = true;

    $id = $_SESSION['user_id'] ?? null;
    if (!is_int($id) && !ctype_digit((string) $id)) {
        return $user = null;
    }

    $stmt = db()->prepare(
        'SELECT id, username, full_name, role, is_active, last_login_at
         FROM users
         WHERE id = ?
         LIMIT 1'
    );
    $stmt->execute([(int) $id]);
    $row = $stmt->fetch();

    if (!$row || !(int) $row['is_active']) {
        unset($_SESSION['user_id']);
        return $user = null;
    }

    return $user = $row;
}

function require_login(): void
{
    if (current_user() === null) {
        flash('warning', 'Vui lòng đăng nhập để tiếp tục.');
        redirect('login.php');
    }
}

function is_admin(): bool
{
    $user = current_user();
    return $user !== null && $user['role'] === 'admin';
}

/**
 * Kiểm tra thông tin đăng nhập.
 * Trả về mảng user khi thành công, null khi thất bại.
 */
function attempt_login(string $username, string $password): ?array
{
    $stmt = db()->prepare(
        'SELECT id, username, password_hash, full_name, role, is_active
         FROM users
         WHERE username = ?
         LIMIT 1'
    );
    $stmt->execute([$username]);
    $row = $stmt->fetch();

    if (!$row) {
        // Giữ thời gian phản hồi tương đối đồng đều để hạn chế dò tài khoản.
        password_verify($password, '$2y$12$invalidinvalidinvalidinvalidinvalidinvalidinvalidinvalidin');
        return null;
    }

    if (!(int) $row['is_active'] || !password_verify($password, (string) $row['password_hash'])) {
        return null;
    }

    if (password_needs_rehash((string) $row['password_hash'], PASSWORD_BCRYPT, ['cost' => 12])) {
        $newHash = password_hash($password, PASSWORD_BCRYPT, ['cost' => 12]);
        $update = db()->prepare('UPDATE users SET password_hash = ? WHERE id = ?');
        $update->execute([$newHash, (int) $row['id']]);
    }

    $mark = db()->prepare('UPDATE users SET last_login_at = NOW() WHERE id = ?');
    $mark->execute([(int) $row['id']]);

    unset($row['password_hash'], $row['is_active']);
    return $row;
}

function login_user(array $user): void
{
    session_regenerate_id(true);
    $_SESSION['user_id'] = (int) $user['id'];
}

function logout_user(): void
{
    $_SESSION = [];

    if (ini_get('session.use_cookies')) {
        $params = session_get_cookie_params();
        setcookie(
            session_name(),
            '',
            time() - 42000,
            $params['path'],
            $params['domain'],
            (bool) $params['secure'],
            (bool) $params['httponly']
        );
    }

    session_destroy();
}