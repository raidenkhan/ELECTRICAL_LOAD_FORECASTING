# GridForecast Pro Release Zipping Script
# This script bundles only the necessary source code and configurations.
# It excludes local dev folders (node_modules, venv, build caches, SQLite files)
# to keep the zip file size as small as possible.

$staging = "gridforecast_staging"

Write-Host "Creating clean staging directory: $staging..." -ForegroundColor Cyan
if (Test-Path $staging) {
    Remove-Item -Recurse -Force $staging
}
New-Item -ItemType Directory -Force -Path $staging | Out-Null

# PRE-FLIGHT: Warn if seed data hasn't been exported yet
$seedFile = "Backend\data\ecg_seed.csv"
if (-not (Test-Path $seedFile)) {
    Write-Host ""
    Write-Host "  WARNING: $seedFile not found!" -ForegroundColor Yellow
    Write-Host "  GRIDCo engineers will see a blank dashboard until they upload data." -ForegroundColor Yellow
    Write-Host "  To fix: run  python tools/export_seed.py  from the Backend directory first." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "  Seed data found: $seedFile" -ForegroundColor Green
}

# 1. Copy root metadata/configuration files
Write-Host "Copying configuration files..." -ForegroundColor Cyan
$rootFiles = @(
    ".env.example",
    "docker-compose.yml",
    "RUNBOOK_DOCKER.md",
    "setup.ps1",
    "setup.sh"
)

foreach ($file in $rootFiles) {
    if (Test-Path $file) {
        Copy-Item $file -Destination $staging -Force
    }
}

# 2. Copy Backend using robocopy to exclude dev folders
Write-Host "Copying Backend (excluding venv, cache, local SQLite db)..." -ForegroundColor Cyan
$backendStaging = Join-Path $staging "Backend"
New-Item -ItemType Directory -Force -Path $backendStaging | Out-Null
robocopy Backend $backendStaging /E /XD venv __pycache__ .pytest_cache /XF app.db loadforecast.db api_test_log.txt error_log.txt .env .env.production temp_verification_data.csv test_sample.csv /NFL /NDL /R:0 /W:0 | Out-Null

# 3. Copy frontend using robocopy to exclude node_modules, build cache
Write-Host "Copying frontend (excluding node_modules, build cache)..." -ForegroundColor Cyan
$frontendStaging = Join-Path $staging "frontend"
New-Item -ItemType Directory -Force -Path $frontendStaging | Out-Null
robocopy frontend $frontendStaging /E /XD node_modules .next /XF tsconfig.tsbuildinfo dependencies_found.tmp /NFL /NDL /R:0 /W:0 | Out-Null

# 5. Compress the staging folder
$zipName = "gridforecast_release.zip"
Write-Host "Compressing files into $zipName..." -ForegroundColor Cyan
if (Test-Path $zipName) {
    Remove-Item $zipName -Force
}
Compress-Archive -Path "$staging\*" -DestinationPath $zipName -Force

# 6. Cleanup staging directory
Write-Host "Cleaning up staging files..." -ForegroundColor Cyan
Remove-Item -Recurse -Force $staging

Write-Host "=== Release ZIP Created Successfully! ===" -ForegroundColor Green
Write-Host "File created: $zipName" -ForegroundColor Green
