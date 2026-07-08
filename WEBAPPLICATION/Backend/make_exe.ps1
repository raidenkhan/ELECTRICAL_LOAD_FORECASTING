<#
.SYNOPSIS
    Build the GRIDCo Load Forecaster standalone Windows .exe via PyInstaller.
.DESCRIPTION
    1. Builds the Next.js frontend (static export)
    2. Installs Python dependencies (with pre-built wheels only)
    3. Installs CPU-only PyTorch from official index
    4. Installs PyInstaller
    5. Runs PyInstaller to produce dist/gridco/
    6. Reports final size
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = $PSScriptRoot
$frontend = Join-Path $root "frontend"

$py = "python"
$pip = "pip"
if (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe") {
    $py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
    $pipExe = "$env:LOCALAPPDATA\Programs\Python\Python313\Scripts\pip.exe"
    if (Test-Path $pipExe) { $pip = $pipExe }
}

$pyVer = & $py -c "import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))"
Write-Host "Using Python $pyVer" -ForegroundColor Cyan

Write-Host "=== GRIDCo Build ===" -ForegroundColor Cyan

Write-Host "[1/4] Building frontend..." -ForegroundColor Yellow
Push-Location $frontend
try {
    if (-not (Test-Path "node_modules")) {
        npm install
        if (-not $?) { throw "npm install failed" }
    }
    npm run build
    if (-not $?) { throw "npm build failed" }
    $outDir = Join-Path $frontend "out"
    if (-not (Test-Path $outDir)) { throw "no out/ dir" }
    $size = (Get-ChildItem -Recurse $outDir | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("Frontend: " + [math]::Round($size, 1) + " MB") -ForegroundColor Green
} finally { Pop-Location }

Write-Host "[2/4] Installing deps..." -ForegroundColor Yellow
Push-Location $backend
try {
    & $py -m pip install --upgrade pip
    & $pip install "--only-binary=:all:" -r requirements.txt
    if (-not $?) { throw "pip install failed" }
    & $pip install torch "--index-url=https://download.pytorch.org/whl/cpu" "--only-binary=:all:"
    if (-not $?) { throw "torch install failed" }
    & $pip install pyinstaller
    if (-not $?) { throw "pyinstaller install failed" }
} finally { Pop-Location }

Write-Host "[3/4] Running PyInstaller..." -ForegroundColor Yellow
Push-Location $backend
try {
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    & $py -m PyInstaller build_exe.spec --clean --noconfirm
    if (-not $?) { throw "PyInstaller failed" }
} finally { Pop-Location }

Write-Host "[4/5] Copying launchers..." -ForegroundColor Yellow
$distDir = Join-Path (Join-Path $backend "dist") "gridco"
if (Test-Path $distDir) {
    Copy-Item (Join-Path $backend "run-gridco.vbs") $distDir
    Copy-Item (Join-Path $backend "run-gridco.bat") $distDir
    Write-Host "Launchers copied: run-gridco.bat (console) + run-gridco.vbs (silent)" -ForegroundColor Green
}

Write-Host "[5/5] Done" -ForegroundColor Cyan
if (Test-Path $distDir) {
    $totalSize = (Get-ChildItem -Recurse $distDir | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("Output: " + [math]::Round($totalSize, 1) + " MB") -ForegroundColor Green
} else {
    Write-Host "dist/gridco/ not found" -ForegroundColor Red
}
