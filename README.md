# GCP Marker Localization and Shape Classification

A deep learning pipeline for detecting Ground Control Point (GCP) markers in large aerial survey images and predicting both marker coordinates and marker shape.

---

## Overview

This project processes large aerial imagery and predicts:

* Marker coordinates (X, Y)
* Marker shape

  * L-Shape
  * Square
  * Cross

The solution uses a multi-stage computer vision pipeline built with PyTorch and ResNet18 models.

---

## Pipeline Architecture

```text
Full Image
     │
     ▼
Tile Generator
     │
     ▼
Marker Classifier
     │
     ▼
Best Tile
 ┌────┴────┐
 ▼         ▼
Localizer  Shape Classifier
 ▼         ▼
(X,Y)     Shape
     │
     ▼
predictions.json
```

---

## Dataset

The dataset consists of:

* Large aerial survey images
* Ground Control Point (GCP) annotations
* Marker shape labels

Example annotation:

```json
{
    "mark": {
        "x": 3272.7,
        "y": 1089.3
    },
    "verified_shape": "Cross"
}
```

Each image contains a single GCP marker.

---

## Data Acquisition

The notebook automatically:

1. Downloads the annotation JSON file.
2. Downloads the image dataset from Google Drive.
3. Reconstructs the original folder structure locally.

Dataset download typically takes approximately 30 minutes depending on network speed.

---

## Exploratory Data Analysis

Initial analysis includes:

* Shape distribution
* Marker coordinate visualization
* Sample image inspection
* Annotation validation

---

## Dataset Preprocessing

The original images are several thousand pixels wide while GCP markers occupy only a small region.

To improve training efficiency:

### Positive Samples

For each annotated image:

* Generate a 512×512 crop around the marker.
* Apply random crop offsets.
* Generate multiple augmented crops.
* Resize crops to 224×224.

Stored information:

* Cropped image
* Relative marker coordinates
* Marker shape

### Negative Samples

Additional crops are generated that do not contain any marker.

These are used to train the marker classifier.

### Preprocessed Dataset

Samples are stored as compressed NPZ files:

```python
{
    "image": image,
    "target": [x, y],
    "shape": shape
}
```

Preprocessing reduces training time significantly by avoiding repeated crop generation during training.

---

## Models

### 1. Marker Localizer

#### Purpose

Predict the marker position within a crop.

#### Architecture

ResNet18 backbone with regression head:

```text
ResNet18
    ↓
Linear
    ↓
ReLU
    ↓
Dropout
    ↓
Linear(2)
```

#### Output

```python
[x, y]
```

#### Loss Function

L1 Loss (Mean Absolute Error)

#### Saved Model

```text
best_localizer.pt
```

---

### 2. Marker Classifier

#### Purpose

Determine whether a crop contains a marker.

#### Classes

```text
0 = No Marker
1 = Marker
```

#### Architecture

ResNet18

#### Loss Function

Cross Entropy Loss

#### Saved Model

```text
best_classifier.pt
```

---

### 3. Shape Classifier

#### Purpose

Predict marker shape.

#### Classes

```text
0 = L-Shape
1 = Square
2 = Cross
```

#### Architecture

ResNet18

#### Loss Function

Cross Entropy Loss

#### Saved Model

```text
best_shape_model.pt
```

---

## Inference Pipeline

### Step 1: Tile Generation

The input image is divided into overlapping 512×512 tiles.

### Step 2: Marker Detection

Each tile is passed through the marker classifier.

The tile with the highest confidence score is selected.

### Step 3: Coordinate Localization

The selected tile is passed to the localizer.

The localizer predicts:

```python
(local_x, local_y)
```

These coordinates are converted back to the original image coordinate system.

### Step 4: Shape Prediction

The selected tile is passed to the shape classifier.

Predicted classes:

* L-Shape
* Square
* Cross

---

## Output

Predictions are exported as:

```text
predictions.json
```

Example:

```json
{
    "project1/survey1/2/DJI_0431.JPG": {
        "mark": {
            "x": 1024.5,
            "y": 850.2
        },
        "verified_shape": "L-Shape"
    }
}
```

This format matches the required submission structure.

---

## Results

### Marker Classification

* Validation Accuracy ≈ 99.8%

### Coordinate Localization

* Validation Error ≈ 2 pixels

### End-to-End Pipeline

* Median Error ≈ 5 pixels

### Failure Mode

The primary failure mode occurs when the classifier selects an incorrect tile from the full image, resulting in large localization errors.

---

## Repository Structure

```text
.
├── GCP_Pipeline.ipynb
├── README.md
├── requirements.txt
├── predictions.json
└── models/
    ├── best_classifier.pt
    ├── best_localizer.pt
    └── best_shape_model.pt
```

---

## Requirements

```text
torch
torchvision
opencv-python
numpy
matplotlib
scikit-learn
tqdm
gdown
google-api-python-client
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

### Hard Negative Mining

Train the classifier using difficult false-positive examples generated during inference.

### Stronger Backbone

Evaluate:

* ResNet34
* EfficientNet
* ConvNeXt

### Heatmap-Based Localization

Replace coordinate regression with heatmap prediction for improved robustness.

### Multi-Task Learning

Train a single model to jointly predict:

* Marker presence
* Coordinates
* Shape

### End-to-End Detection

Evaluate object detection architectures:

* YOLO
* Faster R-CNN
* RetinaNet

to eliminate tile-selection errors.

---

## Author

Developed as part of a Ground Control Point (GCP) localization and classification assignment using PyTorch and Google Colab.
