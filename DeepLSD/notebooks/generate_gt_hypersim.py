from upload_hypersim import upload_images
import os
import torch
from deeplsd.models.deeplsd_inference import DeepLSD

from ground_truth.main_process import process_image

if __name__ == "__main__":


    frames = [f"{i:04d}" for i in range(1, 100)]
    
    desired_images = [
        # "ai_001_001",
        # "ai_001_002",
        # "ai_001_003",
         "ai_001_004",
         "ai_001_005",
         "ai_001_006",
         "ai_001_007",
        "ai_001_008",
        "ai_001_009",
        "ai_001_010",
        # "ai_002_001",
    ]
    
    print("Generate Images")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = {'detect_lines': True, 'line_detection_params': {'merge': False, 'filtering': True, 'grad_thresh': 3}}
    ckpt = torch.load('../weights/deeplsd_md.tar', map_location=device, weights_only=False)
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()
    image_dir = "/work/scratch/maurdu/data"
    cam_view_color = "scene_cam_00_final_preview"
    for image_id in desired_images:
        for frame_str in frames:
            print(f"generate for {image_id}, {frame_str}")
            

            required_file = os.path.join(image_dir, image_id, "images", cam_view_color, f"frame.{frame_str}.color.jpg")

            if not os.path.isfile(required_file):
                print(f"Skipping: {os.path.join(required_file, )} does not exist.")
                continue
            composite_after, pred_lines, img, normals, world_coordinates, valid_mask, line_info, scores, isstruct, original_lines = process_image(
                image_dir, image_id, frame_str, net, device,
                depth_thresh=50, normal_thresh=1.5 * 1e7,
                dataset="hypersim",
                #moge_model = moge,
                file_path = required_file,
            )
            
    print("Finished processing.")