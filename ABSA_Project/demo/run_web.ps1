param(
    [string]$Xampp = "D:\xamp"
)

$ErrorActionPreference = "Stop"

function Test-PortOpen {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $conn
}

function Wait-Port {
    param([int]$Port, [int]$TimeoutSeconds = 30)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortOpen -Port $Port) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

$mysqld = Join-Path $Xampp "mysql\bin\mysqld.exe"
$mysql = Join-Path $Xampp "mysql\bin\mysql.exe"
$httpd = Join-Path $Xampp "apache\bin\httpd.exe"
$myIni = Join-Path $Xampp "mysql\bin\my.ini"

foreach ($path in @($mysqld, $mysql, $httpd)) {
    if (-not (Test-Path $path)) {
        throw "Không tìm thấy $path. Kiểm tra lại đường dẫn XAMPP."
    }
}

if (-not (Test-PortOpen -Port 3306)) {
    Write-Host "Đang khởi động MySQL..."
    Start-Process -FilePath $mysqld -ArgumentList "--defaults-file=$myIni", "--standalone" -WindowStyle Hidden
    if (-not (Wait-Port -Port 3306)) {
        throw "MySQL không mở cổng 3306. Xem $Xampp\mysql\data\mysql_error.log"
    }
}
Write-Host "MySQL đang chạy (cổng 3306)."

if (-not (Test-PortOpen -Port 80)) {
    Write-Host "Đang khởi động Apache..."
    Start-Process -FilePath $httpd -WindowStyle Hidden
    if (-not (Wait-Port -Port 80)) {
        throw "Apache không mở cổng 80. Xem $Xampp\apache\logs\error.log"
    }
}
Write-Host "Apache đang chạy (cổng 80)."

$databaseExists = & $mysql -u root -N -B -e "SHOW DATABASES LIKE 'brand_health_db';" 2>$null
if (-not $databaseExists) {
    Write-Warning "Chưa có database brand_health_db."
    Write-Host "Import bằng lệnh:"
    Write-Host "  cmd /c `"`"$mysql`" -u root --default-character-set=utf8mb4 < `"$PSScriptRoot\database\brand_health_db.sql`"`""
} else {
    $reviewCount = & $mysql -u root -N -B -e "SELECT COUNT(*) FROM brand_health_db.reviews;" 2>$null
    Write-Host "Database brand_health_db sẵn sàng ($reviewCount đánh giá)."
}

Write-Host ""
Write-Host "Website: http://localhost/brand_health_demo/"
Write-Host "Đăng nhập: admin / Admin@123 (đổi mật khẩu sau lần đầu)."
Write-Host "Python API cần chạy riêng bằng .\run_api.ps1 cho Giai đoạn 5."