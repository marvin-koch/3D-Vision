#!/bin/bash
set -euo pipefail # Exit on error, treat unset variables as errors, and ensure pipelines fail correctly.

BASE_PATH="/work/scratch/maurdu/data"
# Set to true for testing (will print what would be deleted), false to actually delete.
DRY_RUN=true # 

# --- Sanity check: Ensure BASE_PATH exists ---
if [ ! -d "$BASE_PATH" ]; then
    echo "ERROR: Base path '$BASE_PATH' does not exist."
    exit 1
fi

# --- Define delete action based on DRY_RUN ---
RM_CMD_ARGS=()
FIND_PRINT_ACTION="-print" # Default for dry run Step 2
if [ "$DRY_RUN" = true ]; then
    echo "--- DRY RUN MODE ENABLED: No files will be deleted. Actions will be printed. ---"
    # For Step 1, find's -delete will be replaced by -print
    # For Step 2, we'll use -print and not pipe to rm
else
    if [[ "$0" == *"delete.sh"* ]]; then # Check if running directly, not sourced, for prompt
        read -r -p "--- LIVE MODE: This script WILL DELETE files from '$BASE_PATH'. Are you absolutely sure? (yes/NO): " confirmation
        if [[ "$confirmation" != "yes" ]]; then
            echo "Aborted by user."
            exit 1
        fi
    fi
    echo "--- LIVE MODE ENABLED: Files will be deleted. ---"
    # Step 1 will use -delete directly if not DRY_RUN
    # Step 2 will use -print0 and pipe to xargs rm
    RM_CMD_ARGS=(rm -f) # For xargs in Step 2
    FIND_PRINT_ACTION="-print0" # For piping to xargs
fi

echo ""
echo "Starting cleanup in: $BASE_PATH"
echo "Will process direct subfolders of '$BASE_PATH' (e.g., ai_001_001, ai_001_002)."
echo "Excluding 'json_output' folder from processing."
echo "-----------------------------------------------------"

find "$BASE_PATH" -maxdepth 1 -mindepth 1 -type d -print0 | while IFS= read -r -d $'\0' current_folder_to_process; do
    folder_name=$(basename "$current_folder_to_process")

    if [ "$folder_name" == "json_output" ]; then
        echo "SKIPPING: $current_folder_to_process (json_output folder)"
        echo "-----------------------------------------------------"
        continue
    fi

    # Optional: If you want to strictly only process folders starting with "ai_", uncomment the following:
    # if [[ ! "$folder_name" =~ ^ai_ ]]; then
    #     echo "SKIPPING: $current_folder_to_process (name does not start with 'ai_')"
    #     echo "-----------------------------------------------------"
    #     continue
    # fi

    echo "Processing: $current_folder_to_process"

    # --- Step 1: Cleanup within all 'scene_cam_00_final_preview' subdirectories ---
    echo "  [Step 1] In all 'scene_cam_00_final_preview' subdirectories found under '$current_folder_to_process':"
    echo "           Deleting JPGs that DO NOT end with '.color.jpg'."
    echo "           Keeping files ending with '.color.jpg' and non-JPG files."

    scfp_found_for_step1=false
    find "$current_folder_to_process" -type d -name "scene_cam_00_final_preview" -print0 | while IFS= read -r -d $'\0' scfp_dir; do
        scfp_found_for_step1=true
        if [ "$DRY_RUN" = true ]; then
             echo "    -> Examining for cleanup in: $scfp_dir"
             find "$scfp_dir" -maxdepth 1 -type f -name "*.jpg" -not -name "*.color.jpg" -print
        else
             echo "    -> Cleaning JPGs in: $scfp_dir"
             find "$scfp_dir" -maxdepth 1 -type f -name "*.jpg" -not -name "*.color.jpg" -delete
        fi
    done

    if [ "$DRY_RUN" = true ] && [ "$scfp_found_for_step1" = false ]; then
        echo "    -> No 'scene_cam_00_final_preview' subdirectories found in '$current_folder_to_process' for Step 1."
    fi

    # --- Step 2: Delete all other files ---
    echo "  [Step 2] In '$current_folder_to_process' (excluding contents of any 'scene_cam_00_final_preview' subdirectories):"
    echo "           Deleting all files."

    # Find files to delete (respecting -prune), then pipe to xargs rm if not dry run
    # FIND_PRINT_ACTION is -print for dry run, -print0 for live run
    # RM_CMD_ARGS is empty for dry run, (rm -f) for live run
    if [ "$DRY_RUN" = true ]; then
        find "$current_folder_to_process" \
            \( -name "scene_cam_00_final_preview" -type d -prune \) -o \
            \( -type f $FIND_PRINT_ACTION \)
    else
        # Use -print0 and xargs -0 for safety with filenames
        find "$current_folder_to_process" \
            \( -name "scene_cam_00_final_preview" -type d -prune \) -o \
            \( -type f -print0 \) | xargs -0 --no-run-if-empty "${RM_CMD_ARGS[@]}"
    fi
    echo "-----------------------------------------------------"
done

echo ""
if [ "$DRY_RUN" = true ]; then
    echo "--- DRY RUN FINISHED. No files were actually deleted. Review the output above. ---"
else
    echo "--- LIVE RUN FINISHED. ---"
fi

echo "Script complete."
