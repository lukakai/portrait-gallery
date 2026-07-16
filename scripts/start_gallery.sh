#!/bin/zsh
set -e

unset XPC_SERVICE_NAME

PROJECT_DIR="/Users/lukai/portrait-gallery"
IMAGE_BUNDLE="/Volumes/copy/portrait-gallery-images.sparsebundle"
IMAGE_MOUNT="$PROJECT_DIR/data/copy_gallery_images"
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"

mkdir -p "$IMAGE_MOUNT"

if [ ! -f "$IMAGE_MOUNT/.portrait-gallery-store" ]; then
  /usr/bin/hdiutil attach "$IMAGE_BUNDLE" -mountpoint "$IMAGE_MOUNT" -nobrowse
fi

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" app/main.py
