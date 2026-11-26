param(
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:PYTHONIOENCODING = "utf-8"

# Dùng JAVA_HOME nếu đã có, nếu không dùng JDK portable của dự án.
$jdk = $env:JAVA_HOME
if (-not $jdk -or -not (Test-Path (Join-Path $jdk "bin\java.exe"))) {
    $jdk = "D:\Ct255-NVTM(BI)\tools\jdk-17"
}
if (Test-Path (Join-Path $jdk "bin\java.exe")) {
    $env:JAVA_HOME = $jdk
    $env:PATH = (Join-Path $jdk "bin") + ";" + $env:PATH
    Write-Host "JAVA_HOME = $jdk"
} else {
    Write-Warning "Không tìm thấy Java tại $jdk. API sẽ báo degraded nếu dùng VnCoreNLP."
}

$python = "D:\conda_envs\vicloabsa_gpu\python.exe"
if (-not (Test-Path $python)) {
    throw "Không tìm thấy Python: $python"
}

$uvicornArgs = @(
    "-m", "uvicorn",
    "python_api.main:app",
    "--host", "127.0.0.1",
    "--port", "$Port"
)
if ($Reload) {
    $uvicornArgs += "--reload"
}

Write-Host "Chạy API tại http://127.0.0.1:$Port"
Write-Host "Docs: http://127.0.0.1:$Port/docs"
& $python @uvicornArgs