from upload_hypersim import upload_images
import os
import torch
from tqdm import tqdm
from deeplsd.models.deeplsd_inference import DeepLSD

from line_understanding.pipeline import process_image

if __name__ == "__main__":


    # frames = [f"{i:04d}" for i in range(1, 100)]
    
    # desired_images = [
    #     #"ai_001_001",
    #     #"ai_001_002",
    #     #"ai_001_003",
    #     # "ai_001_004",
    #     # "ai_001_005",
    #     # "ai_001_006",
    #     # "ai_001_007",
    #     # "ai_001_008",
    #     # "ai_001_009",
    #     "ai_001_010",
    #     "ai_002_001",
    #     "ai_002_002",
    #     "ai_002_003",
    #     "ai_002_004",
    #     "ai_002_005",
    #     "ai_002_006",
    #     "ai_002_007",
    # ]
    
    
    frames = [str(i) for i in range(0, 784)]
    
    desired_images = [
        "scene0707_00",
  
    ]
    
    base_dir = '/Users/marvin/Documents/ETHZ/MA4/3DVision/repo/DeepLSD/notebooks/scannet/scans_test'
    
    dataset = "scannet"
    
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
        for frame_str in tqdm(frames, desc="Processing Frames", leave=False):
            print(f"generate for {image_id}, {frame_str}")
            
            image_dir = os.path.join(base_dir, image_id)
            required_file = frame_str
            
            if dataset == "hypersim":
                cam_view_color = "scene_cam_00_final_preview"

                required_file = os.path.join(image_dir, image_id, "images", cam_view_color, f"frame.{frame_str}.color.jpg")

            elif dataset=="scannet":
                required_file = os.path.join(image_dir,"color", f"{frame_str}.jpg")
            
            
            
            if not os.path.isfile(required_file):
                print(f"Skipping: {os.path.join(required_file, )} does not exist.")
                continue
            process_image(
                base_dir=base_dir, image_id=image_id, frame_str=frame_str, net=net, device=device, thickness = 1,
                thresh_normal=8.2e13, thresh_depth=0.2, dataset=dataset, plot=True, file_path = required_file,
            )
        if cuda_available:
            torch.cuda.empty_cache()
            
    print("Finished processing.")