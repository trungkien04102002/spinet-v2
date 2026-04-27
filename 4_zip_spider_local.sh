#!/bin/bash
# Local Mac: zip the SPIDER dataset for upload to Google Drive.
#
# Usage:
#   ./4_zip_spider_local.sh
#
# Output:
#   spider.zip (~3 GB)
#
# After this, upload spider.zip to Google Drive and copy the file ID.
# Then on Vast.ai run: ./5_download_spider.sh <FILE_ID>

set -e

echo "======================================================================"
echo "Local Mac: zip SPIDER dataset for upload"
echo "======================================================================"

if [ ! -d "spider" ]; then
    echo "Error: spider/ folder not found in current directory"
    echo "Run this script from /Users/kienha/spinet-v2/"
    exit 1
fi

if [ -f "spider.zip" ]; then
    echo "spider.zip already exists. Remove first? (y/N)"
    read -n 1 ans
    echo
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        rm spider.zip
    else
        echo "Aborted."
        exit 1
    fi
fi

echo
echo "[1/2] Zipping spider/ ..."
echo "  Source size: $(du -sh spider/ | awk '{print $1}')"
echo "  Note: .mha files compress only marginally; expect zip size close to source."
echo

# -0: store, no compression (much faster, .mha is already binary-dense)
# -r: recursive
# Add overview.csv and radiological_gradings.csv at top level too
zip -r0 spider.zip spider/

echo
echo "[2/2] Done."
echo "  Output: spider.zip ($(du -sh spider.zip | awk '{print $1}'))"
echo
echo "======================================================================"
echo "Next steps:"
echo "  1. Upload spider.zip to Google Drive"
echo "  2. Right-click on the file → Get link → set 'Anyone with the link'"
echo "  3. Copy the FILE_ID from URL (between /d/ and /view):"
echo "     https://drive.google.com/file/d/<FILE_ID>/view"
echo "  4. On Vast.ai instance run:"
echo "     ./5_download_spider.sh <FILE_ID>"
echo "======================================================================"
