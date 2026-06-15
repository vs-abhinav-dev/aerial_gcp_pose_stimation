# GCP Marker Localization and Shape Classification Pipeline

## Overview

This project detects Ground Control Point (GCP) markers in large aerial survey images and predicts:

1. Marker coordinates (X, Y)
2. Marker shape (L-Shape, Square, or Cross)

The solution is implemented in a single Google Colab notebook and uses a multi-stage deep learning pipeline built with PyTorch and ResNet18 models.

---

# Dataset Acquisition

The dataset is provided through Google Drive.

The notebook automatically:

1. Downloads the annotation file (`gcp_marks.json`)
2. Downloads the image dataset from the provided Google Drive folder
3. Reconstructs the folder structure locally in Colab

The download stage is the most time-consuming part of the workflow and typically takes approximately 30 minutes.

---

# Exploratory Data Analysis (EDA)

Basic dataset analysis is performed to understand:

* Number of annotated images
* Distribution of marker shapes
* Marker coordinate ranges
* Sample visualization of GCP locations

The annotations contain:

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

# Dataset Preprocessing

The original aerial images are very large (several thousand pixels wide), while GCP markers occupy only a small portion of the image.

Direct training on full-resolution images would be computationally inefficient.

## Cropping Strategy

For every annotated image:

1. A 512×512 crop is generated around the marker.
2. Random offsets are applied to the crop center.
3. Multiple augmented crops are generated from each image.
4. Crops are resized to 224×224.

This process increases the dataset size to approximately 5000 training samples.

---

## Positive Samples

Positive samples contain a GCP marker.

Stored information:

* Cropped image
* Marker coordinates relative to crop
* Marker shape

---

## Negative Samples

Additional crops without a GCP marker are generated.

These samples are used for training the marker classifier.

Purpose:

* Teach the classifier to distinguish marker-containing tiles from background tiles.
* Reduce false positive detections.

---

## Preprocessed Dataset Format

Each sample is stored as a compressed NPZ file containing:

```python
{
    "image": image,
    "target": [x, y],
    "shape": shape
}
```

Preprocessing significantly reduces training time because expensive crop generation is performed only once.

---

# Model Architecture

Three independent ResNet18-based models are trained.

---

# 1. Marker Localizer

## Goal

Predict the marker position within a crop.

## Input

224×224 crop

## Output

```python
[x, y]
```

Coordinates of the marker center.

## Architecture

ResNet18 backbone with a regression head:

```python
Linear -> ReLU -> Dropout -> Linear(2)
```

## Loss Function

L1 Loss (Mean Absolute Error)

## Output

Saved as:

```text
best_localizer.pt
```

---

# 2. Marker Classifier

## Goal

Determine whether a crop contains a GCP marker.

## Classes

```text
0 = No Marker
1 = Marker
```

## Architecture

ResNet18

Final layer:

```python
Linear(..., 2)
```

## Loss Function

Cross Entropy Loss

## Output

Saved as:

```text
best_classifier.pt
```

---

# 3. Shape Classifier

## Goal

Predict the marker shape.

## Classes

```text
L-Shape
Square
Cross
```

## Architecture

ResNet18

Final layer:

```python
Linear(..., 3)
```

## Loss Function

Cross Entropy Loss

## Output

Saved as:

```text
best_shape_model.pt
```

---

# Inference Pipeline

The final prediction pipeline consists of three stages.

## Step 1: Tile Generation

The full image is divided into overlapping 512×512 tiles.

---

## Step 2: Marker Detection

Each tile is passed through the marker classifier.

The tile with the highest confidence score is selected.

---

## Step 3: Coordinate Localization

The selected tile is passed to the localizer.

The localizer predicts:

```python
(local_x, local_y)
```

These coordinates are transformed back to the original image coordinate system.

---

## Step 4: Shape Prediction

The selected tile is passed to the shape classifier.

Predicted classes:

* L-Shape
* Square
* Cross

---

# Output Format

The final predictions are exported to:

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

This matches the format required by the provided annotations.

---

# Results

Observed performance:

* Marker Classification Accuracy: ~99.8%
* Crop Localization Error: ~2 pixels
* End-to-End Median Error: ~5 pixels

The primary failure mode occurs when the classifier selects an incorrect tile from the full image.

---

# Future Improvements

## 1. Hard Negative Mining

Current negative samples are randomly generated.

A stronger classifier could be trained by collecting false positives produced during inference and retraining using those difficult examples.

Expected impact:

* Reduced catastrophic localization failures.

---

## 2. ResNet34 or EfficientNet Backbone

Replacing ResNet18 with a larger backbone may improve localization and shape classification accuracy.

Expected impact:

* Lower coordinate error.
* Better shape classification.

---

## 3. Heatmap-Based Localization

Instead of directly regressing coordinates, predict a heatmap representing marker probability.

Advantages:

* More robust localization.
* Better handling of ambiguous cases.

---

## 4. Multi-Task Learning

Train a single network to jointly predict:

* Marker presence
* Marker coordinates
* Marker shape

Advantages:

* Reduced inference time.
* Shared feature learning.

---

## 5. Object Detection Models

Replace the two-stage classifier/localizer pipeline with a dedicated detector such as:

* YOLO
* Faster R-CNN
* RetinaNet

Advantages:

* End-to-end training.
* Elimination of tile-selection failures.

---

# Conclusion

The implemented system successfully detects GCP markers in large aerial images and predicts both marker coordinates and marker shape. The pipeline uses three lightweight ResNet18 models and produces predictions in the required submission format while maintaining low localization error and high classification accuracy.
