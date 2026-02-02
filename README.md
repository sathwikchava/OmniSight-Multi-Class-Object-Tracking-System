#OmniSight: Multi-Class Object Tracking Model in Complex Environments

This project implements a versatile object tracking system using **YOLOv8**. While originally structured for autonomous driving contexts, the model supports the standard **COCO dataset**, meaning it can detect and track **81 different classes** of objects. It is a Large-Scale Multi-Class Object Tracking in Complex Environments.

**It extends far beyond just vehicles**, capable of recognizing:
- **Living Beings**: Persons, cats, dogs, birds, horses, sheep, cows, elephants, etc.
- **Vehicles**: Bicycles, cars, motorcycles, airplanes, buses, trains, trucks, boats.
- **Household & Personal**: Backpacks, umbrellas, handbags, suitcases, ties, bottles, cups, forks, knives, spoons.
- **Food**: Bananas, apples, sandwiches, oranges, broccoli, carrots, pizza, donuts, cake.
- **Furniture & Electronics**: Chairs, couches, potted plants, beds, dining tables, TVs, laptops, mice, keyboards, cell phones.
- **Sports & Outdoors**: Frisbees, skis, snowboards, sports balls, kites, baseball bats, skateboards, tennis rackets.

This makes the tracker suitable for a wide variety of applications, from traffic monitoring to retail analytics, home security, and wildlife observation.

## 📂 Project Structure

- **`datasets/`**: Contains the dataset used for training.
    - **Dataset Used**: `coco128` (A subset illustrating the wide range of detectable objects).
    - **Configuration**: Defined in `data.yaml` (Lists all 81 detectible classes).
- **`models/`**: Stores the model weights.
    - **`yolov8n.pt`**: The pre-trained YOLOv8 Nano model. This is the lightweight, fast baseline model.
    - **`fine_tuned_model.pt`**: Further trained/fine-tuned version of YOLOv8n, optimized for this specific project/dataset.
- **`videos/`**: Contains input video files for testing the tracking system.
    - **Available Test Videos**:
        - `sample1.mp4`
        - `sample2.mp4`
        - `sample3.mp4`
        - `test3.mp4`
- **`outputs/`**: The destination folder where processed videos with tracking overlays are saved.
- **`runs/`**: Automatically generated folder by YOLOv8 containing training logs, confusion matrices, and validation metrics (found in `runs/detect/train*`).
- **`scripts/`**: Python scripts for executing various parts of the pipeline.

## 🚀 How to Run

### Prerequisities
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 1. Object Tracking (Inference)
You can run tracking on the provided test videos.

**Option A: Use the Fine-Tuned Model**
This script uses `models/fine_tuned_model.pt` to track objects in `videos/sample2.mp4`.
```bash
python scripts/fine_tuning.py
```
*Note: Although named `fine_tuning.py`, this script performs **tracking/inference**.*

**Option B: Use the Baseline Model**
This script uses the standard `models/yolov8n.pt` model to track objects in `videos/sample3.mp4`.
```bash
python scripts/track_objects.py
```

**Where is the output?**
- The processed videos with bounding boxes and class labels will be saved in the **`outputs/`** directory (e.g., `outputs/sample44_output.mp4` or `outputs/sample2_output.mp4`).

### 2. Export Model (ONNX)
To export the fine-tuned PyTorch model to ONNX format for deployment (e.g., on edge devices):
```bash
python scripts/export_to_onnx.py
```
- **Output**: Generates `models/fine_tuned_model.onnx`.

### 3. Model Training / Fine-Tuning
To retrain or fine-tune the model yourself:
- Open **`fine_tuning.ipynb`** in Jupyter Notebook or VS Code.
- Run the cells to load the `yolov8n.pt` model and train it using the configuration in `data.yaml`.
- Training results (loss graphs, best weights) will be saved to `runs/detect/train/`.

## 🧠 Model & Dataset Details

- **Model Architecture**: YOLOv8 Nano (n). Chosen for its speed and efficiency, making it suitable for real-time autonomous driving tasks.
- **Dataset**: COCO128. A diverse dataset containing common objects found in driving scenarios (cars, pedestrians, traffic lights, signs).
- **Classes**: 81 object categories (see `data.yaml` for the full list).
