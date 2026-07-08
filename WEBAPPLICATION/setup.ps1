Write-Host "=== GridForecast Pro Setup ===" -ForegroundColor Cyan

# Check for Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Docker is not installed. Please install Docker Desktop and try again." -ForegroundColor Red
    exit
}

# Initialize .env
if (-not (Test-Path .env)) {
    Write-Host "Creating .env from .env.example..."
    Copy-Item .env.example .env
    # Generate random secret key
    $SecKey = [Convert]::ToHexString((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))
    (Get-Content .env) -replace 'SECRET_KEY=generate-a-secure-key-here', "SECRET_KEY=$SecKey" | Set-Content .env
    Write-Host ".env initialized with new SECRET_KEY."
}

Write-Host "Building and starting containers..."
docker compose build
docker compose up -d

Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host "Access the dashboard at: http://localhost:3000"
Write-Host "API documentation at: http://localhost:8000/docs"
