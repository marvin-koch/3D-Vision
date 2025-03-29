from upload_hypersim import upload_images, delete_images, generate_image_list
import os
import torch
import sys
from deeplsd.models.deeplsd_inference import DeepLSD
from line_understanding.pipeline import process_image_pipeline, plot_pipeline_results
from line_understanding.json_saver import save_lines_to_json 
import argparse
import h5py


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Define starting image and num images")

    # Define the command-line arguments you expect
    parser.add_argument('-i', '--start_image', type=str, help="Start image", required=True)
    parser.add_argument('-n', '--n_images', type=int, help="Number of images", required=True)
    parser.add_argument('-f', '--file_type', type=str, help="File storage type", required=True)

    # Parse the arguments from the command line
    args = parser.parse_args()
    file_type = str(args.file_type)
    frame_str = "0001"
    start_image = str(args.start_image)
    num_images = int(args.n_images)
    desired_images = generate_image_list(start_image, num_images)
    files_to_download = [
        f"frame.{frame_str}.color.jpg",
        f"frame.{frame_str}.depth_meters.hdf5",
        f"frame.{frame_str}.normal_world.hdf5",
        f"frame.{frame_str}.position.hdf5"
    ]
    
    upload_images(desired_images, files_to_download)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = {'detect_lines': True, 'line_detection_params': {'merge': False, 'filtering': True, 'grad_thresh': 3}}
    ckpt = torch.load('../weights/deeplsd_md.tar', map_location='cpu', weights_only=False)
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()
    
    if file_type == "json":
        for image_id in desired_images:
            processed_data = process_image_pipeline(image_id, frame_str, net, device)
            if processed_data is not None:
                plot_pipeline_results(processed_data, frame_str)
                save_lines_to_json(image_id, processed_data["line_info"],processed_data["coplanarity_matrix"])
    elif file_type == "hdf5":       
        # Open HDF5 file for writing
        output_dir = "hdf5_output"
        os.makedirs(output_dir, exist_ok=True)
        hdf5_file = os.path.join(output_dir, f"{start_image}.hdf5")
        with h5py.File(hdf5_file, 'w') as f:
            for image_id in desired_images:
                processed_data = process_image_pipeline(image_id, frame_str, net, device)
                if processed_data is not None:
                    img_data = save_lines_to_json(image_id, processed_data["line_info"],processed_data["coplanarity_matrix"], save=False)
                    image_group = f.create_group(img_data['image_id'])
                    image_group.create_dataset('coordinates', data=[line['coordinates'] for line in img_data['lines']])
                    image_group.create_dataset('confidence_scores', data=[line['confidence_score'] for line in img_data['lines']])
                    image_group.create_dataset('coplanarity_matrix', data=img_data['coplanarity_matrix'])
    else:
        print("Invalid file type. Please choose either json or hdf5.")       
    delete_images(desired_images)