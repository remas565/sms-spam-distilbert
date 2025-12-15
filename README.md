# SMS Spam Classification using Fine-Tuned BERT

A comprehensive deep learning project for SMS spam detection using **BERT-base-uncased** and transfer learning.
The project systematically evaluates fine-tuning, optimizer choice, data augmentation, regularization techniques, and learning rate strategies, with detailed experimental analysis.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Installation](#installation)
* [Experimental Setup](#experimental-setup)
* [Experiments & Results](#experiments--results)
* [Model Interpretability](#model-interpretability)
* [Key Insights](#key-insights)
* [Acknowledgments](#acknowledgments)

---

## Overview

This project implements an **SMS Spam Detection system using BERT-base-uncased**, a transformer-based language model pre-trained on large-scale English corpora.
The model is adapted to the spam classification task through **task-specific fine-tuning**, while extensive experiments are conducted to evaluate optimization strategies and robustness improvements.

---

## Features

* **Baseline Evaluation** of pre-trained BERT without task-specific training
* **Transfer Learning** via fine-tuning on SMS spam data
* **Hyperparameter Tuning** (learning rate, batch size, epochs)
* **Optimizer Comparison** (AdamW, RMSprop, SGD)
* **Data Augmentation** (Synonym Replacement, Random Deletion)
* **Regularization Techniques** (Dropout, Early Stopping)
* **Learning Rate Scheduling** (Reduce on Plateau, Linear Warmup)
* **Gradient-based Token Importance Analysis**
* **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, Loss

---

## Dataset

**SMS Spam Collection Dataset**

* **Total Messages**: 5,572
* **Ham (Non-Spam)**: 4,825 (86.6%)
* **Spam**: 747 (13.4%)

### Balancing Strategy

To address class imbalance, the dataset was **balanced by random sampling**:

* 747 Spam
* 747 Ham

### Data Split

* **Training**: 70%
* **Validation**: 15%
* **Testing**: 15%

Stratified splitting was applied to preserve class distribution.

---

## Model Architecture

### Base Model

* **Model**: BERT-base-uncased
* **Layers**: 12 Transformer encoder layers
* **Hidden Size**: 768
* **Parameters**: ~110M
* **Tokenizer**: BertTokenizerFast
* **Max Sequence Length**: 128 tokens

### Classification Head

* Linear layer (768 → 2)
* Softmax activation for binary classification

### Training Strategy

* **Frozen Layers**: BERT encoder
* **Trainable Parameters**: Classification head only

---

## Installation

### Requirements

```bash
Python 3.8+
torch
transformers
datasets
scikit-learn
nlpaug
nltk
```

### Install Dependencies

```bash
pip install transformers datasets scikit-learn torch nlpaug nltk
```

### NLTK Resources

```python
import nltk
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")
```

---

## Experimental Setup

* **Batch Size**: 8
* **Epochs**: 5
* **Learning Rate**: 2e-5
* **Weight Decay**: 0.01
* **Optimizer (Baseline)**: AdamW
* **Evaluation Metric for Model Selection**: Accuracy / F1-score

---

## Experiments & Results

### 1. Baseline (Pre-trained BERT – No Fine-Tuning)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 72.61% |
| Precision | 27.59% |
| Recall    | 64.29% |
| F1-score  | 38.61% |
| Loss      | 0.6847 |

🔎 *Observation*: Pre-trained BERT without fine-tuning performs poorly on SMS spam detection.

---

### 2. Fine-Tuned BERT (Baseline Model)

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | **98.80%** |
| Precision | 95.54%     |
| Recall    | 95.54%     |
| F1-score  | 95.54%     |
| Loss      | 0.0418     |

Fine-tuning yields a **dramatic performance improvement**, validating transfer learning.

---

### 3. Optimizer Comparison

| Optimizer | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Loss  |
| --------- | ------------ | ------------- | ---------- | ------------ | ----- |
| AdamW     | **99.52**    | **100.00**    | 96.43      | **98.18**    | 0.028 |
| RMSprop   | 99.52        | 100.00        | 96.43      | 98.18        | 0.029 |
| SGD       | 98.56        | 96.30         | 92.86      | 94.55        | 0.045 |

 **Best Optimizer**: AdamW (highest stability and lowest loss).

---

### 4. Data Augmentation

#### Synonym Replacement

Improved robustness and maintained high F1-score.

#### Random Deletion

Provided minor robustness gains but slightly reduced stability compared to synonym augmentation.

---

### 5. Regularization Techniques

#### Dropout (p = 0.3)

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | **99.40%** |
| Precision | 99.08%     |
| Recall    | 96.43%     |
| F1-score  | **97.74%** |
| Loss      | 0.0315     |

**Best Overall Performance Achieved with Dropout**

#### Early Stopping

Reduced training time while maintaining comparable performance.

---

### 6. Learning Rate Strategies

| Strategy             | Accuracy (%) | F1-score (%) |
| -------------------- | ------------ | ------------ |
| Fixed LR (Baseline)  | 98.80        | 95.54        |
| Reduce LR on Plateau | ↑ Improved   | ↑ Improved   |
| Linear Warmup        | Comparable   | Stable       |

Learning rate scheduling improves convergence stability.

---

## Model Interpretability

A **gradient-based token importance analysis** was applied to individual SMS messages.

### Example (Spam)

Important tokens included:

* *congratulations*
* *iphone*
* *prize*
* *call*
* *claim*

### Example (Ham)

Important tokens included:

* *home*
* *dinner*
* *call*
* *later*

This confirms that the model learns **semantic spam cues**, not simple keyword rules.

---

## Key Insights

1. Fine-tuning is essential for transformer-based spam detection.
2. AdamW consistently provides the best optimizer performance.
3. Data augmentation improves robustness.
4. Dropout regularization yields the highest overall performance.
5. Learning rate scheduling enhances convergence stability.
6. The model bases predictions on meaningful semantic patterns.

---

## Acknowledgments

* **Dataset**: SMS Spam Collection Dataset
* **Model**: HuggingFace BERT-base-uncased
* **Libraries**: PyTorch, Transformers, scikit-learn, nlpaug

---
