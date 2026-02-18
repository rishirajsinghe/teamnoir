# Offroad Semantic Segmentation using DINOv2

## Hackathon Submission

---

## 📌 1. Project Overview

This project implements semantic segmentation for offroad driving scenes using a pretrained **DINOv2 (ViT-S/14)** backbone along with a lightweight **ConvNeXt-style segmentation head**.

The primary objective of this work is to improve segmentation quality by maximizing the **Mean Intersection over Union (IoU)** metric through controlled training optimization and hyperparameter tuning.

---

## 📊 2. Final Validation Performance

After multiple experimental training runs (1, 10 and 20 epochs), the best validation performance obtained is:


| Metric            | Value  |
|-------------------|--------|
| Mean IoU          | 0.3546 |
| Dice Score        | 0.5141 |
| Pixel Accuracy    | 0.7082 |
| Validation Loss   | 0.9556 |


Training run at 20 epochs produced:

| Metric            | Value  |
|-------------------|--------|
| Mean IoU          | 0.3603 |
| Dice Score        | 0.5207 |
| Pixel Accuracy    | 0.7123 |
| Validation Loss   | 0.9501 |

---

## 🧠 3. Model Architecture

| Component            | Details              |
|----------------------|----------------------|
| Backbone             | DINOv2 ViT-S/14      |
| Segmentation Head    | ConvNeXt-style       |
| Embedding Dimension  | 384                  |
| Loss Function        | Weighted CrossEntropy|
| Optimizer            | AdamW                |
| Learning Rate        | 3e-4                 |
| Epochs Trained       | 20                   |
| Backbone Training    | Frozen               |
| GPU Used             | RTX 4050 (CUDA)      |

---

## 🧰 4. Environment & Dependencies

### Required Setup

- Python 3.10+
- CUDA-enabled GPU
- Conda (recommended)
- PyTorch (CUDA version)

---

## ⚙️ 5. Installation & Environment Setup

### Step 1: Create Conda Environment

```bash
conda create -n EDU python=3.10
conda activate EDU
```

---

### Step 2: Install Required Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python pillow matplotlib tqdm
```

---

### Step 3: Verify GPU Availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected Output:

```
True
```

---

## 🚀 6. Training the Model

Navigate to project directory:

```bash
cd Offroad_Segmentation_Scripts
```

Start training:

```bash
python train_segmentation.py
```

### During Training

- Pretrained DINOv2 backbone is loaded
- Backbone remains frozen
- Segmentation head is trained
- Weighted CrossEntropy loss is applied
- AdamW optimizer is used
- Training runs for defined epochs

---

## 📁 7. Model Output After Training

After training completes, the following files are generated:

```
train_stats/
segmentation_head.pth
```

---

## 🧪 8. Model Evaluation

Evaluate trained model on validation dataset:

```bash
python test_segmentation.py --data_dir ../Offroad_Segmentation_Training_Dataset/val
```

---

## 📂 9. Output Files After Evaluation

After evaluation, a `predictions/` folder will be created:

```
predictions/
│── masks/
│── masks_color/
│── comparisons/
│── evaluation_metrics.txt
│── per_class_metrics.png
```

---

## 📈 10. Metric Interpretation

- **Mean IoU** → Primary segmentation performance metric  
- **Dice Score** → Region overlap accuracy  
- **Pixel Accuracy** → Overall classification accuracy  
- **Per-class IoU** → Performance across individual classes  

Strong performing classes:
- Sky
- Landscape

Challenging classes:
- Logs
- Rocks
- Small cluttered objects

---

