@echo off
title GridForecast Pro Controller
:menu
cls
echo ===================================================
echo               GRIDForecast Pro Controller
echo ===================================================
echo   [1] Start System (Run in background)
echo   [2] Stop System (Shut down containers safely)
echo   [3] View Live Logs (Press Ctrl+C to stop viewing)
echo   [4] Check Service Status (Healthy check)
echo   [5] Rebuild and Restart (Use after code updates)
echo   [6] Open App in Browser
echo   [7] Exit
echo ===================================================
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto start_sys
if "%choice%"=="2" goto stop_sys
if "%choice%"=="3" goto view_logs
if "%choice%"=="4" goto check_status
if "%choice%"=="5" goto rebuild_sys
if "%choice%"=="6" goto open_browser
if "%choice%"=="7" goto exit_menu
goto menu

:start_sys
echo.
echo Checking for Docker...
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH!
    echo Please install Docker Desktop first: https://www.docker.com/products/docker-desktop/
    pause
    goto menu
)
echo Creating .env file if it doesn't exist...
if not exist .env (
    echo Creating .env from .env.example...
    copy .env.example .env
    echo .env created!
)
echo Starting GridForecast Pro...
docker compose up -d
echo.
echo System started successfully in the background!
echo Access Dashboard: http://localhost:3000
echo Access API Docs:  http://localhost:8000/docs
echo.
pause
goto menu

:stop_sys
echo.
echo Stopping GridForecast Pro containers...
docker compose down
echo Containers stopped!
pause
goto menu

:view_logs
echo.
echo Streaming logs (Press Ctrl+C to stop)...
docker compose logs -f
pause
goto menu

:check_status
echo.
echo Checking running containers...
docker compose ps
echo.
echo Checking backend health...
curl -s http://localhost:8000/health || echo [WARNING] Backend is not responding yet.
echo.
pause
goto menu

:rebuild_sys
echo.
echo Rebuilding containers (this will download updates if needed)...
docker compose down
docker compose build --no-cache
docker compose up -d
echo Rebuild complete!
pause
goto menu

:open_browser
echo.
echo Opening http://localhost:3000 in your browser...
start http://localhost:3000
goto menu

:exit_menu
exit
