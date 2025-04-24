#!/bin/bash

# --- Configuration ---
TARGET_DIR="/work/scratch/maurdu/data"
URLS=(
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_001.zip"
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_002.zip"
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_003.zip"
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_004.zip"
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_005.zip"
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_006.zip"
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_007.zip"
)
# Set to true to remove the zip file after successful unpacking, false to keep it
REMOVE_ZIP_AFTER_UNPACK=true
MIN_FILE_COUNT_TO_SKIP=100 # Minimum number of files in target dir to skip download/extract
# --- End Configuration ---

# --- Ensure target directory exists ---
echo "Ensuring base target directory exists: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"
if [ $? -ne 0 ]; then
    echo "Error: Could not create target directory ${TARGET_DIR}. Please check permissions."
    exit 1
fi
# --- Process each URL ---
for url in "${URLS[@]}"; do
    echo "-----------------------------------------"
    echo "Processing URL: ${url}"

    # Extract filename from URL (e.g., ai_001_001.zip)
    filename=$(basename "${url}")
    # Extract base name without extension (e.g., ai_001_001)
    base_name="${filename%.*}"
    # Full path to the downloaded zip file
    zip_file_path="${TARGET_DIR}/${filename}"
    # Define the *final expected* extraction path (no extra nesting)
    final_dir_path="${TARGET_DIR}/${base_name}"

    # --- Check if target directory exists and is populated ---
    echo "Checking for existing data in: ${final_dir_path}"
    if [ -d "${final_dir_path}" ]; then
        echo "Directory ${final_dir_path} exists. Checking file count..."
        # Count files recursively within the directory
        file_count=$(find "${final_dir_path}" -type f | wc -l)
        echo "Found ${file_count} files."
        if [ "$file_count" -ge "$MIN_FILE_COUNT_TO_SKIP" ]; then
            echo "Directory ${final_dir_path} exists and contains at least ${MIN_FILE_COUNT_TO_SKIP} files. Skipping download and extraction."
            continue # Skip to the next URL
        else
             echo "Directory ${final_dir_path} exists but has fewer than ${MIN_FILE_COUNT_TO_SKIP} files. Proceeding with download/extraction (may overwrite)."
             # Decide if you want to remove the existing directory first
             # echo "Warning: Overwriting contents in ${final_dir_path}"
             # rm -rf "${final_dir_path}" # Uncomment to force clean extraction
        fi
    else
        echo "Directory ${final_dir_path} does not exist. Proceeding with download."
    fi

    # --- Download ---
    echo "Downloading ${filename} to ${TARGET_DIR}..."
    # Use wget -P to specify the output directory
    # Add -nc flag to avoid re-downloading if zip file already exists
    wget -nc -P "${TARGET_DIR}" "${url}"
    if [ $? -ne 0 ]; then
        # Check if download failed because file already exists (-nc flag)
        if [ -f "${zip_file_path}" ]; then
             echo "Zip file ${zip_file_path} already exists. Proceeding to unpack."
        else
            echo "Error: Failed to download ${url}. Skipping this file."
            continue # Skip to the next URL
        fi
    else
        echo "Download complete: ${zip_file_path}"
    fi

    # --- Unpack ---
    # No need to create specific extract path directory beforehand,
    # unzip will create the base_name directory if it's inside the zip.
    echo "Unpacking ${filename} into ${TARGET_DIR}..."
    # Unzip directly into the main target directory.
    # Assumes the zip file contains a single top-level folder named "${base_name}"
    unzip -o "${zip_file_path}" -d "${TARGET_DIR}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to unpack ${zip_file_path}. Please check if 'unzip' is installed and the file is valid."
        # Keep the zip file for manual inspection if unpacking fails
        continue # Skip to the next URL
    fi
    echo "Unpacking complete. Data should be in ${final_dir_path}."

    # --- Optional Cleanup ---
    if [ "$REMOVE_ZIP_AFTER_UNPACK" = true ]; then
        echo "Removing downloaded zip file: ${zip_file_path}"
        rm "${zip_file_path}"
        if [ $? -ne 0 ]; then
            echo "Warning: Could not remove zip file ${zip_file_path}."
        fi
    else
         echo "Keeping downloaded zip file: ${zip_file_path}"
    fi

done

echo "-----------------------------------------"
echo "All processing tasks finished."
echo "Data should be in subdirectories under ${TARGET_DIR}"
exit 0