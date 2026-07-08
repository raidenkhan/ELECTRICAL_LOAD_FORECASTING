@echo off
title GRIDCo Load Forecaster
cd /d "%~dp0"
echo ===================================================
echo        GRIDCo Load Forecasting System
echo ===================================================
echo.
echo Starting server...
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop the server
echo.
dist\gridco\gridco_launcher.exe
pause
