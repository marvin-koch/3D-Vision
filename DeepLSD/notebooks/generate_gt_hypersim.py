from upload_hypersim import upload_images
import os
import torch
from tqdm import tqdm
from deeplsd.models.deeplsd_inference import DeepLSD

from line_understanding.pipeline import process_image

if __name__ == "__main__":


    # frames = [f"{i:04d}" for i in range(1, 100)]
    
    # desired_images = [
    # "ai_006_001",
    # "ai_006_002",
    # "ai_006_003",
    # "ai_006_004",
    # "ai_006_005",
    # "ai_006_006",
    # "ai_006_007",
    # "ai_006_008",
    # "ai_006_009",
    # "ai_006_010",
    # "ai_007_001",
    # "ai_007_002",
    # "ai_007_003",
    # "ai_007_004",
    # "ai_007_005",
    # "ai_007_006",
    # "ai_007_007",
    # "ai_007_008",
    # "ai_007_009",
    # "ai_007_010",
    # "ai_008_001",
    # "ai_008_002",
    # "ai_008_003",
    # "ai_008_004",
    # "ai_008_005",
    # "ai_008_006",
    # "ai_008_007",
    # "ai_008_008",
    # "ai_008_009",
    # "ai_008_010",
    # "ai_009_001",
    # "ai_009_002",
    # "ai_009_003",
    # "ai_009_004",
    # "ai_009_005",
    # "ai_009_006",
    # "ai_009_007",
    # "ai_009_008",
    # "ai_009_009",
    # "ai_009_010",
    # "ai_010_001",
    # "ai_010_002",
    # "ai_010_003",
    # "ai_010_004",
    # "ai_010_005",
    # "ai_010_006",
    # "ai_010_007",
    # "ai_010_008",
    # "ai_010_009",
    # ]
    
    base_dir = 'eth3d'

    all_entries = os.listdir(base_dir)

    # Keep only directories that start with "DSC_"
    desired_images = [
        entry
        for entry in all_entries
        if os.path.isdir(os.path.join(base_dir, entry))
    ]
    

    dataset = "midas"
    
    print("Generate Images")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        # Keep only directories that start with "DSC_"
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
            
            elif dataset=="eth3d" or dataset=="diode" or dataset=="moge" or dataset=="midas":
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
                    thresh_normal=140000000000000, thresh_depth=800, dataset=dataset, plot=False, file_path = required_file,
                ) #1.7e14
            elif dataset=="diode":    
                process_image(
                    base_dir=base_dir, image_id=image_id, frame_str=frame_str, net=net, device=device, thickness = 1,
                    thresh_normal=70000000000000, thresh_depth=50, dataset=dataset, plot=False, file_path = required_file,
                ) #1.7e14
            elif dataset=="moge":
                process_image(
                    base_dir=base_dir, image_id=image_id, frame_str=frame_str, net=net, device=device, thickness = 1,
                    thresh_normal=70000000000000, thresh_depth=1, dataset=dataset, plot=False, file_path = required_file,
                ) #1.7e14
            else:
                process_image(
                    base_dir=base_dir, image_id=image_id, frame_str=frame_str, net=net, device=device, thickness = 1,
                    thresh_normal=90000000000000, thresh_depth=5, dataset=dataset, plot=False, file_path = required_file,
                )
        if cuda_available:
            torch.cuda.empty_cache()
            
    print("Finished processing.")