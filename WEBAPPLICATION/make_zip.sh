#!/bin/bash
# GridForecast Pro Release Zipping Script (Linux/macOS)
# Excludes local dev folders (node_modules, venv, build caches, SQLite files)
# to keep the zip file size as small as possible.

set -e

STAGING="gridforecast_staging"
ZIP_NAME="gridforecast_release.zip"

echo "Creating clean staging directory: $STAGING..."
rm -rf "$STAGING"
mkdir -p "$STAGING"

# 1. Copy root metadata/configuration files
echo "Copying configuration files..."
rootFiles=(
    ".env.example"
    "docker-compose.yml"
    "RUNBOOK_DOCKER.md"
    "setup.ps1"
    "setup.sh"
)

for file in "${rootFiles[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$STAGING/"
    fi
done

# 2. Copy Backend (excluding venv, cache, local databases/logs/secrets)
echo "Copying Backend..."
mkdir -p "$STAGING/Backend"
rsync -a --exclude="venv/" --exclude="__pycache__/" --exclude=".pytest_cache/" \
         --exclude="app.db" --exclude="loadforecast.db" \
         --exclude="api_test_log.txt" --exclude="error_log.txt" \
         --exclude=".env" --exclude=".env.production" \
         Backend/ "$STAGING/Backend/"

# 3. Copy frontend (excluding node_modules, build cache, tsbuildinfo)
echo "Copying frontend..."
mkdir -p "$STAGING/frontend"
rsync -a --exclude="node_modules/" --exclude=".next/" \
         --exclude="tsconfig.tsbuildinfo" --exclude="dependencies_found.tmp" \
         frontend/ "$STAGING/frontend/"

# 5. Compress the staging folder
echo "Compressing files into $ZIP_NAME..."
rm -f "$ZIP_NAME"
cd "$STAGING"
zip -r "../$ZIP_NAME" . > /dev/null
cd ..

# 6. Cleanup staging directory
echo "Cleaning up staging files..."
rm -rf "$STAGING"

echo "=== Release ZIP Created Successfully! ==="
echo "File created: $ZIP_NAME"
