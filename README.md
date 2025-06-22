# Graph Neural Networks for Line and Coplanarity Classification


## Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
```
Our repo already includes the DeepLSD line detector, you can install simply with:
```bash
bash install.sh
```
## Generate ground truths

Process images with DeepLSD line detection to generate ground truth data for different datasets.

### Basic Usage

```bash
python src/line_classification/generate_ground_truth.py
```


#### Arguments

- `--base_dir`: Base directory containing the image data (default: `data`)
- `--dataset`: Dataset type to process (default: `hypersim`)
- Supported datasets: `hypersim`, `scannet`, `eth3d`, `diode`

### Example

```bash
# Combine both options
python src/line_classification/generate_ground_truth.py --base_dir /path/to/your/data --dataset hypersim
```

### Dataset-Specific Processing

The script automatically applies different processing parameters based on the selected dataset:

- **hypersim**: `thresh_normal=8.2e13`, `thresh_depth=0.2`
- **scannet**: `thresh_normal=1e14`, `thresh_depth=5`
- **eth3d**: `thresh_normal=140000000000000`, `thresh_depth=800`
- **diode**: `thresh_normal=70000000000000`, `thresh_depth=50`


## Train Model

We provide the config we used to train our model, you can change the hyperparameters with it. 
```bash
cd src/line_classification/lightning_tools/
python train.py --config config.yaml
```

Our trained model weight can be found in:

```
src/line_classification/lightning_tools/checkpoint
```


## Inference Notebook
An example inference notebook with visualizations can be found in:
```
src/line_classification/InferenceGNN.ipynb
```
