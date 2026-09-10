# Hybrid-DenseNet121-ViT-B16
# Hybrid DenseNet121–ViT-B/16 with Cross-Attention for Skin Lesion Classification

A deep learning framework for **binary skin lesion classification** that combines **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** through a **Cross-Attention Fusion** mechanism.

The project is designed not only to train a high-performance classifier, but also to provide a comprehensive evaluation and interpretability pipeline including **Grad-CAM, feature visualization, statistical confidence intervals, McNemar's test, calibration analysis, threshold analysis, error analysis, and Decision Curve Analysis (DCA)**.

---

## 📌 Overview

This project proposes a hybrid architecture combining:

* **DenseNet121** for convolutional/local visual feature extraction.
* **ViT-B/16 (Vision Transformer)** for global image representation.
* **Cross-Attention Fusion** to enable interaction between CNN and Transformer feature representations.
* Extensive evaluation and statistical analysis to assess model performance beyond standard accuracy.

The pipeline covers the complete workflow from data preparation to model evaluation, interpretability, statistical analysis, and result saving.

---

## 🏗️ Model Architecture

The proposed architecture consists of three main components:

### 1. DenseNet121

DenseNet121 is used as the CNN branch to extract hierarchical and localized visual features from skin lesion images.

### 2. ViT-B/16

Vision Transformer (ViT-B/16) is used as the Transformer branch to capture global relationships and long-range dependencies within the image.

### 3. Cross-Attention Fusion

The extracted representations from DenseNet121 and ViT-B/16 are integrated using a **Cross-Attention mechanism**.

This allows the model to learn interactions between:

* Local CNN features
* Global Transformer features

The fused representation is then passed to the classification head for binary prediction.

```text
                 Input Image
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
     DenseNet121              ViT-B/16
          │                     │
          │ Local Features      │ Global Features
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
             Cross-Attention
                  Fusion
                     │
                     ▼
              Feature Fusion
                     │
                     ▼
             Classification Head
                     │
                     ▼
              Binary Prediction
```

---

## 🔬 Dataset

The project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset for skin lesion classification.

The dataset contains dermoscopic images representing different categories of skin lesions.

For the binary classification setting, the target task focuses on distinguishing:

* **Melanoma — Positive Class**
* **Non-Melanoma — Negative Class**

> The exact preprocessing, split strategy, and class mapping are implemented in the project notebook.

---

# 🚀 Pipeline

## 1. Data Preparation

The data preparation stage includes:

* Dataset loading
* Data inspection
* Train / validation / test splitting
* Class distribution analysis
* Image visualization
* Data augmentation
* Image preprocessing
* Dataset preparation for CNN and Transformer branches

---

## 2. Hybrid Model Construction

The hybrid architecture consists of:

* DenseNet121 CNN backbone
* ViT-B/16 Transformer backbone
* Cross-Attention fusion
* Classification head

The two branches extract complementary representations before being integrated through cross-attention.

---

## 3. Training

The model training pipeline includes:

* **AdamW optimizer**
* Learning-rate scheduling
* Early stopping
* Automatic Mixed Precision (AMP)
* Model checkpointing
* Best-model selection
* Training/validation monitoring

These techniques are used to improve training efficiency, stability, and generalization.

---

# 📊 Evaluation

The evaluation pipeline goes beyond standard classification metrics and includes a comprehensive binary classification analysis.

### Classification Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1-score
* Balanced Accuracy
* ROC-AUC
* PR-AUC
* Confusion Matrix

The **melanoma class is treated as the positive class** throughout the binary clinical evaluation.

---

# 📈 ROC and Precision-Recall Analysis

The project generates:

* ROC curves
* Precision-Recall curves
* ROC-AUC
* PR-AUC
* Confidence intervals for ROC-AUC

These analyses provide a more complete assessment of model discrimination, especially under class imbalance.

---

# 📐 Confidence Intervals

Confidence intervals are calculated to quantify the uncertainty around model performance estimates.

The analysis includes confidence intervals for:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1-score
* ROC-AUC
* Per-class performance

This provides a statistical perspective instead of relying only on single-point metric estimates.

---

# 🧪 McNemar's Test

**McNemar's test** is used to statistically compare the prediction errors of two paired classification models.

The test evaluates whether the difference between the models' classification errors is statistically significant.

This is particularly useful when comparing the proposed hybrid architecture against baseline models.

---

# 🎯 Decision Threshold Analysis

Instead of relying exclusively on the default classification threshold, the project investigates different decision thresholds.

The analysis evaluates how changing the threshold affects:

* Sensitivity
* Specificity
* Precision
* F1-score
* Accuracy
* Other clinically relevant metrics

This helps identify operating points that may be more suitable depending on the intended application.

---

# 🩺 Calibration Analysis

Model confidence is evaluated to determine whether predicted probabilities correspond to actual observed outcomes.

The calibration analysis includes:

* Reliability diagram
* Calibration curve
* Prediction confidence distribution
* Probability histograms
* Calibration-related metrics

The goal is to determine whether the model is:

* Overconfident
* Underconfident
* Well calibrated

---

# 📉 Decision Curve Analysis

**Decision Curve Analysis (DCA)** is performed to evaluate the potential clinical usefulness of the model across different probability thresholds.

The analysis considers:

* Net Benefit
* Treat-All strategy
* Treat-None strategy
* Model strategy

DCA provides an additional perspective on whether using the model could provide practical decision-making value.

---

# 🧠 Explainable AI — Grad-CAM

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to visualize the image regions that contribute most strongly to the model's predictions.

This helps investigate whether the model focuses on meaningful lesion regions rather than irrelevant background information.

The generated heatmaps provide visual interpretability for individual predictions.

---

# 🔍 Feature Visualization

The learned feature representations are explored using dimensionality-reduction techniques.

### t-SNE

t-SNE is used to project high-dimensional learned representations into a lower-dimensional space for visualization.

### UMAP

UMAP is used as an additional feature visualization technique to investigate the structure and separability of the learned representations.

These visualizations help analyze whether the learned feature space forms meaningful class-specific clusters.

---

# ❌ Error Analysis

A detailed error analysis is performed to understand where and why the model fails.

The analysis investigates:

* False Positives
* False Negatives
* Misclassified melanoma cases
* Misclassified non-melanoma cases
* Low-confidence predictions
* High-confidence incorrect predictions

---

# 🗂️ Error Categories

Misclassified samples are further analyzed according to potential visual characteristics and failure patterns.

Examples include:

* Difficult lesion morphology
* Similar visual appearance between classes
* Ambiguous cases
* Image artifacts
* Poor image quality
* Background interference
* High-confidence incorrect predictions

This analysis helps identify potential weaknesses of the model and directions for future improvement.

---

# 📊 Per-Class Confidence Analysis

Prediction confidence is analyzed separately for each class.

The analysis investigates:

* Confidence distribution
* Mean confidence
* Confidence variation
* Correct vs incorrect predictions
* Per-class confidence intervals

This helps determine whether the model behaves differently across melanoma and non-melanoma cases.

---

# 📚 Model Summary

The project includes a detailed model summary containing:

* Architecture structure
* Trainable parameters
* Non-trainable parameters
* Layer information
* Feature dimensions
* Classification head configuration

---

# 💾 Results Saving

The project automatically saves important outputs generated during training and evaluation.

Examples include:

```text
results/
│
├── checkpoints/
│   ├── best_model.pth
│   └── last_model.pth
│
├── metrics/
│   ├── classification_metrics.csv
│   ├── confidence_intervals.csv
│   └── per_class_metrics.csv
│
├── plots/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── calibration_curve.png
│   ├── reliability_diagram.png
│   ├── threshold_analysis.png
│   ├── decision_curve.png
│   ├── tsne.png
│   └── umap.png
│
├── gradcam/
│   └── ...
│
└── error_analysis/
    └── ...
```

The exact directory structure may vary depending on the experiment configuration.

---

# 🛠️ Technologies

The project is implemented using Python and modern deep learning and data science libraries.

### Core

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Scikit-learn

### Deep Learning

* DenseNet121
* Vision Transformer (ViT-B/16)
* Cross-Attention
* Automatic Mixed Precision (AMP)
* AdamW

### Explainability & Visualization

* Grad-CAM
* Matplotlib
* Seaborn
* t-SNE
* UMAP

### Statistical & Clinical Evaluation

* Confidence Intervals
* McNemar's Test
* Calibration Analysis
* Decision Curve Analysis
* Threshold Analysis
* Error Analysis

---

# 🔬 Experimental Analysis

The project is designed to evaluate the proposed architecture from multiple perspectives:

```text
                    Hybrid Model
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Performance      Interpretability   Statistics
        │                │                │
        ▼                ▼                ▼
 Accuracy           Grad-CAM          Confidence
 Precision          Feature Viz       Intervals
 Recall             t-SNE             McNemar
 Specificity        UMAP              Analysis
 ROC-AUC                              Calibration
 PR-AUC                               DCA
                                      Threshold
```

This provides a more comprehensive evaluation than reporting classification accuracy alone.

---

# 📌 Key Features

* ✅ DenseNet121 + ViT-B/16 hybrid architecture
* ✅ Cross-Attention feature fusion
* ✅ Binary melanoma classification
* ✅ Advanced training pipeline
* ✅ Automatic Mixed Precision
* ✅ Early stopping
* ✅ Checkpointing
* ✅ Comprehensive clinical metrics
* ✅ ROC / PR analysis
* ✅ ROC-AUC confidence intervals
* ✅ Per-class confidence intervals
* ✅ McNemar's statistical test
* ✅ Decision threshold analysis
* ✅ Calibration analysis
* ✅ Reliability diagrams
* ✅ Decision Curve Analysis
* ✅ Grad-CAM explainability
* ✅ t-SNE feature visualization
* ✅ UMAP feature visualization
* ✅ Detailed error analysis
* ✅ Error categorization
* ✅ Per-class confidence analysis
* ✅ Automated results saving

---

# 📁 Project Structure

```text
Hybrid-DenseNet-ViT/
│
├── data/
│   └── ...
│
├── notebooks/
│   └── hybrid_densenet_vit.ipynb
│
├── models/
│   └── ...
│
├── results/
│   ├── metrics/
│   ├── plots/
│   ├── gradcam/
│   └── error_analysis/
│
├── checkpoints/
│   └── ...
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ahmed-hosny755/Hybrid-DenseNet-ViT.git
cd Hybrid-DenseNet-ViT
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Open the main notebook:

```text
notebooks/hybrid_densenet_vit.ipynb
```

Then run the pipeline in the following order:

```text
Data Preparation
        ↓
Model Initialization
        ↓
Training
        ↓
Evaluation
        ↓
Grad-CAM
        ↓
Feature Visualization
        ↓
Error Analysis
        ↓
Statistical Analysis
        ↓
Calibration
        ↓
Decision Curve Analysis
        ↓
Results Saving
```

---

# ⚠️ Reproducibility

Random seeds are used where applicable to improve reproducibility.

However, exact results may vary depending on:

* GPU architecture
* CUDA version
* PyTorch version
* Library versions
* Random initialization
* Hardware-level numerical differences

---

# 📜 Disclaimer

This project is intended for **research and educational purposes**.

The model is not a clinically validated medical diagnostic system and should not be used as a substitute for professional medical diagnosis or clinical decision-making.

---

# 👨‍💻 Author

**Ahmed Hosny**

Computer Science — Aswan University

Interested in:

* Data Science
* Machine Learning
* Deep Learning
* Computer Vision
* Medical AI
* AI Research

---

# ⭐ Acknowledgements

This project builds upon established deep learning architectures and publicly available research resources, including DenseNet, Vision Transformers, Grad-CAM, and statistical evaluation methodologies for machine learning models.
