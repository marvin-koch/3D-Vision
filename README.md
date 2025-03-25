# 3D-Vision

Generate a batch of ground truths
```
python /DeepLSD/notebooks/generate_ground_truth_batch -i <start_image> -n <num_images> -f <file_type>
```

### Arguments:

- `-i` or `--start_image`: 
  - **Description**: Id of starting image.
  - **Type**: String (required).
  
- `-n` or `--n_images`: 
  - **Description**: Number of images (starting from start_image) whos ground truth we want to generate
  - **Type**: Integer (required).
  
- `-f` or `--file_type`: 
  - **Description**: Type of file storage.
  - **Type**: String (required).
  - **Allowed Values**: `json`, `hdf5`.
