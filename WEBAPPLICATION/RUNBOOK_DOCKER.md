# GridForecast Pro: Docker Deployment Guide

This guide is for GRIDCo engineers to install the system on a local machine or server with minimum technical effort.

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS)
- Or **Docker Engine** + **Docker Compose** (Linux)

## Installation

### Windows
1. Open PowerShell in the project directory.
2. Run: `./setup.ps1`

### Linux / macOS
1. Open a terminal in the project directory.
2. Run: `chmod +x setup.sh && ./setup.sh`

## Accessing the System
- **Dashboard:** [http://localhost:3000](http://localhost:3000)
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## Common Commands
- **Stop system:** `docker compose down`
- **View logs:** `docker compose logs -f`
- **Rebuild after updates:** `docker compose build --no-cache`

## Troubleshooting
If the database fails to start, ensure port `5432` is not already in use by another PostgreSQL installation on your machine.
