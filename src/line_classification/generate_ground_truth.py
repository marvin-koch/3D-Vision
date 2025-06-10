from upload_hypersim import upload_images
import os
import torch
from tqdm import tqdm
from deeplsd.models.deeplsd_inference import DeepLSD

from line_understanding.pipeline import process_image

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process images with DeepLSD line detection")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="data",
        help="Base directory containing the image data (default: data)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["hypersim", "scannet", "eth3d", "diode"],
        default="hypersim",
        help="Dataset type to process (default: hypersim)"
    )
    return parser.parse_args()

if __name__ == "__main__":    

    args = parse_args()
    
    base_dir = args.base_dir
    dataset = args.dataset

    all_entries = os.listdir(base_dir)

    desired_images = [
        entry
        for entry in all_entries
        if os.path.isdir(os.path.join(base_dir, entry))
    ]
    

    
    print("Generate Images")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    conf = {'detect_lines': True, 'line_detection_params': {'merge': False, 'filtering': True, 'grad_thresh': 3}}
    ckpt = torch.load('../weights/deeplsd_md.tar', map_location=device, weights_only=False)
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()
    cuda_available = torch.cuda.is_available()

    outer_loop = tqdm(desired_images, desc="Processing Dataset")
    for image_id in outer_loop:
        outer_loop.set_description("Processing Image {}".format(image_id))
        image_dir = os.path.join(base_dir, image_id)

        all_entries = os.listdir(image_dir)

        frames = [
            entry
            for entry in all_entries
            if os.path.isdir(os.path.join(image_dir, entry))
        ]
        for frame_str in tqdm(frames, desc="Processing Frames", leave=False):
            print(f"Generate for {image_id}, {frame_str}")
            
            required_file = frame_str
            
            if dataset == "hypersim":
                cam_view_color = "scene_cam_00_final_preview"

                required_file = os.path.join(image_dir, image_id, "images", cam_view_color, f"frame.{frame_str}.color.jpg")

            elif dataset=="scannet":
                required_file = os.path.join(image_dir,"color", f"{frame_str}.jpg")
            
            elif dataset=="eth3d" or dataset=="diode":
                required_file = os.path.join(image_dir,frame_str,"image.jpg")
            
            
            
            
            if not os.path.isfile(required_file):
                print(f"Skipping: {os.path.join(required_file, )} does not exist.")
                
            if dataset=="hypersim":
                process_image(
                    base_dir=base_dir, image_id=image_id, frame_str=frame_str, net=net, device=device, thickness = 1,
                    thresh_normal=8.2e13, thresh_depth=0.2, dataset=dataset, plot=True, file_path = required_file,
                )
            elif dataset=="scannet":
                 process_image(
                    base_dir=base_dir, image_id=image_id, frame_str=frame_str, net=net, device=device, thickness = 1,
                    thresh_normal=1e14, thresh_depth=5, dataset=dataset, plot=True, file_path = required_file,
                ) #1.7e14
            elif dataset=="eth3d":
                process_image(
                    base_dir=base_dir, image_id=image_id, frame_str=frame_str, net=net, device=device, thickness = 1,
                    thresh_normal=140000000000000, thresh_depth=800, dataset=dataset, plot=True, file_path = required_file,
                ) #1.7e14
            else:    
                process_image(
                    base_dir=base_dir, image_id=image_id, frame_str=frame_str, net=net, device=device, thickness = 1,
                    thresh_normal=70000000000000, thresh_depth=50, dataset=dataset, plot=True, file_path = required_file,
                ) #1.7e14
        if cuda_available:
            torch.cuda.empty_cache()
            
    print("Finished processing.")