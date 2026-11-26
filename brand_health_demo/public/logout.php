<?php

declare(strict_types=1);

require __DIR__ . '/../app/bootstrap.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    redirect('index.php');
}

verify_csrf();
logout_user();

session_name('brandhealth_sid');
session_start();
flash('success', 'Bạn đã đăng xuất.');
redirect('login.php');