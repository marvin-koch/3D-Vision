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
# --- End Configuration ---

# --- Ensure target directory exists ---
echo "Ensuring target directory exists: ${TARGET_DIR}"
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
    # Define the final extraction path (nested)
    extract_path="${TARGET_DIR}/${base_name}/${base_name}"

    # --- Download ---
    echo "Downloading ${filename} to ${TARGET_DIR}..."
    # Use wget -P to specify the output directory
    wget -P "${TARGET_DIR}" "${url}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download ${url}. Skipping this file."
        continue # Skip to the next URL
    fi
    echo "Download complete: ${zip_file_path}"

    # --- Create nested directory structure ---
    echo "Creating extraction directory: ${extract_path}"
    mkdir -p "${extract_path}"
    if [ $? -ne 0 ]; then
        echo "Error: Could not create extraction directory ${extract_path}. Skipping unpacking for ${filename}."
        # Optionally remove the downloaded zip if directory creation failed
        # rm -f "${zip_file_path}"
        continue # Skip to the next URL
    fi

    # --- Unpack ---
    echo "Unpacking ${filename} into ${extract_path}..."
    # Use unzip -d to specify the destination directory
    unzip "${zip_file_path}" -d "${extract_path}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to unpack ${zip_file_path}. Please check if 'unzip' is installed and the file is valid."
        # Keep the zip file for manual inspection if unpacking fails
        continue # Skip to the next URL
    fi
    echo "Unpacking complete for ${filename}."

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
echo "All downloads and unpacking tasks finished."
echo "Data should be in subdirectories under ${TARGET_DIR}"
exit 0