import os
import subprocess
import time
import glob


def upload_images(desired_images, files_to_download):

    base_data_dir = "data"
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)
        print(f"Created base data folder: {base_data_dir}")
    else:
        print(f"Base data folder {base_data_dir} already exists.")


    
    # Download Loop
    for image_id in desired_images:
        # Create a folder for the image
        image_dir = os.path.join(base_data_dir, image_id)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            #print(f"Created folder: {image_dir}")
        else:
            #print(f"Folder {image_dir} already exists.")
            pass

        # Try to download all the required files.
        for file_name in files_to_download:
            # Check if the file already exists.
            matching_files = glob.glob(os.path.join(image_dir, "**", file_name), recursive=True)
            if matching_files:
                #print(f"File {file_name} already exists in {image_dir} (found: {matching_files[0]}). Skipping download.")
                continue

            # Build the command for download.py.
            cmd = [
                "python", "download.py",
                "--contains", image_id,
                "--contains", file_name,
                "--directory", image_dir,
                "--silent"
            ]
            #print(f"Downloading {file_name} for {image_id} ...")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, stdin=subprocess.DEVNULL)

            # Poll for file existence and stability.
            timeout = 1000  # seconds
            stable_iterations = 2  # number of consecutive checks required for stability
            stable_count = 0
            prev_size = -1
            start_time = time.time()
            while True:
                matching_files = glob.glob(os.path.join(image_dir, "**", file_name), recursive=True)
                if matching_files:
                    # Check the file size of the first match.
                    file_path = matching_files[0]
                    try:
                        current_size = os.path.getsize(file_path)
                    except OSError:
                        current_size = 0

                    # If the file size is non-zero and hasn't changed since last check:
                    if current_size > 0 and current_size == prev_size:
                        stable_count += 1
                    else:
                        stable_count = 0  # Reset if it changed or if it's still zero

                    prev_size = current_size

                    if stable_count >= stable_iterations:
                        #print(f"File {file_name} appears fully downloaded (stable size: {current_size} bytes). Terminating process.")
                        proc.terminate()  # terminate the download process if file is stable
                        break
                else:
                    # No file found yet: reset stable count.
                    stable_count = 0
                    prev_size = -1

                if time.time() - start_time > timeout:
                    #print(f"Timeout reached while waiting for {file_name} in {image_dir}. Killing process.")
                    proc.kill()
                    break
                time.sleep(1)

            # Capture any remaining output.
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()

            if proc.returncode not in [0, None]:
                #print(f"Process for {file_name} in {image_id} exited with return code {proc.returncode}.")
                pass



