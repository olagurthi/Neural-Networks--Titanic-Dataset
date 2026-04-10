# 🚢 Titanic Survival Prediction — Neural Networks Deep Dive

> Binary classification with PyTorch: comparing a **Simple** vs **Deep** neural network on the classic Titanic dataset, with thorough hyperparameter analysis.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Experiments](#experiments)
- [Setup & Usage](#setup--usage)
- [File Structure](#file-structure)
- [Key Concepts](#key-concepts)

---

## Project Overview

This project builds and compares two neural network architectures to predict passenger survival on the Titanic. It goes beyond just training a model — it **visualises learning behaviour**, **measures quality with AUC**, **analyses confusion matrices**, and **systematically experiments with hyperparameters** including learning rate, batch size, regularisation, and epoch count.

The notebook is written with **detailed inline comments** designed to be educational for anyone learning neural networks.

---

## Dataset

**File:** `Titanic-Dataset.csv`  
**Source:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)  
**Size:** 891 passengers × 12 columns

| Feature | Description | Used? |
|---|---|---|
| `Survived` | Target: 0 = Died, 1 = Survived | ✅ (label) |
| `Pclass` | Ticket class (1st, 2nd, 3rd) | ✅ |
| `Sex` | Gender | ✅ |
| `Age` | Age in years | ✅ |
| `SibSp` | # siblings/spouses aboard | ✅ |
| `Parch` | # parents/children aboard | ✅ |
| `Fare` | Ticket fare | ✅ |
| `Embarked` | Port of embarkation | ✅ |
| `Name`, `Ticket`, `Cabin`, `PassengerId` | High-cardinality / low signal | ❌ dropped |

**Survival rate:** 38.4% (survived) — slightly imbalanced, handled via AUC metric.

---

## Models

### Model 1 — Simple Neural Network (Baseline)

```
Input (7) → Linear(7→16) → ReLU → Dropout → Linear(16→1) → Sigmoid
```

- **~128 parameters**
- Fast to train
- Good as a lower-bound baseline

### Model 2 — Deep Neural Network

```
Input (7)
  → Linear(7→64) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64→64) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64→32) → ReLU → Dropout(0.15)
  → Linear(32→16) → ReLU
  → Linear(16→1) → Sigmoid
```

- **~6,000 parameters**
- Batch Normalisation for training stability
- Dropout for regularisation
- LR scheduler (ReduceLROnPlateau)

---

## Results

| Metric | Simple NN | Deep NN |
|---|---|---|
| Validation Accuracy | ~80% | ~82% |
| AUC | ~0.85 | ~0.87 |
| Training Time | Fast | Moderate |
| Parameters | ~128 | ~6,000 |

> Exact numbers vary slightly per run due to random weight initialisation.

---

## Experiments

The notebook runs **4 hyperparameter sweeps**, each with a visualisation:

| Experiment | Values Tested | What to Watch For |
|---|---|---|
| **Learning Rate** | 1e-4, 5e-4, 1e-3, 5e-3, 1e-2 | Too low = slow; too high = unstable |
| **Batch Size** | 8, 16, 32, 64, 128 | Small = noisy; large = smoother |
| **Regularisation** | Dropout 0.0–0.7 + L2 0–1e-2 | Too much = underfits; too little = overfits |
| **Epochs** | 1–150 (long run) | Find where val loss starts rising |

---

## File Structure

```
titanic-nn-pytorch/
│
├── titanic_nn_analysis.ipynb   # Main notebook — all code and analysis
├── Titanic-Dataset.csv         # Dataset (place here before running)
└── README.md                   # This file
```

---

## Key Concepts

### Neural Network Terminology

| Term | Meaning |
|---|---|
| **Epoch** | One full pass through all training data |
| **Batch size** | Number of samples processed before a weight update |
| **Learning rate** | Step size for each gradient descent update |
| **ReLU** | Activation function: `max(0, x)` — introduces non-linearity |
| **Sigmoid** | Squashes output to `[0, 1]` — gives a probability |
| **Dropout** | Randomly zeroes neurons during training — prevents overfitting |
| **Batch Norm** | Normalises layer outputs per batch — stabilises training |
| **L2 / Weight Decay** | Penalises large weights — smoother decision boundaries |

### Evaluation Metrics

| Metric | Formula | When to use |
|---|---|---|
| **Accuracy** | (TP+TN) / Total | Balanced classes |
| **AUC** | Area under ROC curve | Imbalanced classes |
| **Precision** | TP / (TP+FP) | When false positives are costly |
| **Recall** | TP / (TP+FN) | When false negatives are costly |
| **F1** | 2·P·R / (P+R) | Balance of precision & recall |

### Diagnosing Training Curves

```
Train loss ≫ Val loss   →  Underfitting  (add layers / train longer)
Val loss ≫ Train loss   →  Overfitting   (add dropout / reduce model size)
Both decreasing         →  Learning ✓
Val loss starts rising  →  Stop here! (early stopping point)
```
