# Car Damage Prediction

Deep-learning image classification project for identifying **8 car damage classes** from vehicle images. The project compares a CNN built from scratch with multiple transfer-learning architectures and selects the best-performing model using a consistent train/validation/test pipeline.

## Project Overview

The goal is to automate car-damage classification from images and evaluate how different CNN approaches perform on a relatively small, imbalanced dataset.

The final dataset contains **1,512 unique RGB JPEG images**, all standardized to **224 × 224** pixels across 8 classes:

- bumper_dent
- bumper_scratch
- door_dent
- door_scratch
- glass_shatter
- head_lamp
- tail_lamp
- unknown

The `unknown` class represents images with no recognizable damage belonging to the defined damage categories.

## Workflow

1. Removed duplicate label rows and validated image-label mappings.
2. Audited image dimensions, format, channels, aspect ratio, class distribution and sample images.
3. Used a stratified 70/15/15 train-validation-test split.
4. Applied mild augmentation: horizontal flips, rotation, color jitter and affine transforms.
5. Used dataset-specific normalization for the scratch CNN and ImageNet normalization for pretrained models.
6. Built a CNN from scratch as a baseline.
7. Applied transfer learning using ResNet18, EfficientNet-B0 and ResNet50.
8. Fine-tuned pretrained models progressively by first training the classifier head and then unfreezing deeper layers.
9. Selected checkpoints using validation loss and evaluated final models on the untouched test set.
10. Added single-image inference to compare actual and predicted classes on unseen test samples.

## Model Performance

| Model | Test Accuracy |
|---|---:|
| Scratch CNN V1 | 30.40% |
| EfficientNet-B0 | 82.82% |
| ResNet18 | 87.22% |
| **ResNet50** | **88.55%** |

**Best model:** ResNet50 fine-tuned on the car-damage dataset.

The final ResNet50 achieved approximately:

- **Test Accuracy:** 88.55%
- **Macro F1:** 0.87
- **Weighted F1:** 0.89

Transfer learning produced a major improvement over the scratch CNN because pretrained ImageNet features provided stronger low-level and mid-level visual representations for a dataset of this size.

## Key Findings

- The dataset itself was clean and standardized; **class imbalance** was the larger modeling challenge.
- A CNN trained from scratch struggled because only about 1,058 images were available for training.
- Gradual fine-tuning performed better than immediately unfreezing an entire pretrained network.
- ResNet18 performed strongly and was close to ResNet50 despite being smaller.
- EfficientNet-B0 improved substantially over the baseline but did not outperform the ResNet models.
- ResNet50 provided the strongest overall test performance.
- Some classes such as `head_lamp` remained harder than others, indicating that future gains should focus on class-specific errors rather than blindly increasing model complexity.

## Repository Files

```text
Car Damage Prediction/
├── car_damage_classification_final.ipynb
├── Car Damage Prediction.pptx
├── class_to_idx.json
└── README.md
```

- `car_damage_classification_final.ipynb` — complete EDA, preprocessing, modeling, fine-tuning and evaluation workflow.
- `Car Damage Prediction.pptx` — project presentation summarizing methodology, experiments and findings.
- `class_to_idx.json` — label-to-index mapping used during training and inference.

## Class Mapping

```json
{
  "bumper_dent": 0,
  "bumper_scratch": 1,
  "door_dent": 2,
  "door_scratch": 3,
  "glass_shatter": 4,
  "head_lamp": 5,
  "tail_lamp": 6,
  "unknown": 7
}
```

## Tech Stack

- Python
- PyTorch
- Torchvision
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PIL
- Jupyter Notebook

## Running the Notebook

Install the main dependencies:

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn pillow jupyter
```

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
car_damage_classification_final.ipynb
```

The dataset is **not included in this repository**. Update the notebook's dataset paths to match the location of the dataset on your machine before running it.

## Future Improvements

- ResNet18 + ResNet50 probability ensemble
- ConvNeXt-Tiny transfer learning
- Test-time augmentation
- Targeted augmentation or additional samples for weaker classes
- Deployment as an API or web-based image classifier

## Project Outcome

The project demonstrates the full computer-vision workflow from data audit and preprocessing to baseline modeling, transfer learning, fine-tuning, error analysis and inference. The final ResNet50 model improved test accuracy from **30.40% with a scratch CNN to 88.55%**.
