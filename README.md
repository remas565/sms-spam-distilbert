# SMS Spam Detection using DistilBERT

A comprehensive deep learning project for detecting spam messages using transfer learning with DistilBERT. This project includes baseline evaluation, fine-tuning, optimization experiments, and multiple improvement techniques.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Installation](#installation)
* [Experiments & Results](#experiments--results)
* [Performance Metrics](#performance-metrics)
* [Key Insights](#key-insights)
* [Contributing](#contributing)
* [Acknowledgments](#acknowledgments)

---

## Overview

This project implements a state-of-the-art SMS spam detection system using DistilBERT, a distilled version of BERT that preserves most of BERT’s performance while significantly reducing computational cost. The system achieves high classification performance through task-specific fine-tuning, optimization strategies, and transfer learning evaluation.

---

## Features

* **Baseline Model Evaluation** using a pre-trained DistilBERT model
* **Transfer Learning** through fine-tuning on SMS spam data
* **Optimizer Comparison** (AdamW, RMSProp, SGD)
* **Learning Rate Analysis**
* **Data Augmentation** using synonym replacement
* **Dropout Regularization** for improved generalization
* **Cross-Dataset Transfer Learning Evaluation**
* **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, and Loss

---

## Dataset

The primary dataset used in this project is the SMS Spam Collection Dataset:

* **Total Messages**: 5,572
* **Ham (Non-spam)**: 4,825 (86.6%)
* **Spam**: 747 (13.4%)
* **Split Ratio**: 70% Training / 15% Validation / 15% Testing

An additional external SMS dataset is used to evaluate transfer learning and generalization performance.

---

## Model Architecture

### Base Model

* **Model**: DistilBERT-base-uncased
* **Parameters**: ~66 million
* **Tokenizer**: DistilBertTokenizerFast
* **Maximum Sequence Length**: 128 tokens

### Classification Head

* Fully connected linear layer (768 → 2)
* Softmax activation for binary classification

### Training Configuration

* **Frozen Layers**: DistilBERT encoder layers
* **Trainable Parameters**: Classification head only
* **Batch Size**: 16
* **Epochs**: 3–5 depending on experiment

---

## Installation

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended)
```

### Dependencies

```bash
pip install transformers datasets scikit-learn torch nlpaug nltk
```

### NLTK Resources

```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
```

---

## Experiments & Results

### 1. Baseline Evaluation (Before Fine-tuning)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 72.61% |
| Precision | 27.59% |
| Recall    | 64.29% |
| F1-Score  | 38.61% |
| Loss      | 0.6847 |

**Analysis**: The baseline pre-trained model performs poorly on SMS spam detection, highlighting the necessity of task-specific fine-tuning.

---

### 2. Fine-Tuned Model Performance

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 98.80% |
| Precision | 95.54% |
| Recall    | 95.54% |
| F1-Score  | 95.54% |
| Loss      | 0.0418 |

**Improvement**: Significant performance gains are achieved after fine-tuning, demonstrating the effectiveness of transfer learning.

---

### 3. Optimizer Comparison

| Optimizer | Accuracy | Precision | Recall | F1-Score | Loss   |
| --------- | -------- | --------- | ------ | -------- | ------ |
| AdamW     | 99.52%   | 100.00%   | 96.43% | 98.18%   | 0.0281 |
| RMSProp   | 99.52%   | 100.00%   | 96.43% | 98.18%   | 0.0291 |
| SGD       | 98.56%   | 96.30%    | 92.86% | 94.55%   | 0.0450 |

**Finding**: AdamW and RMSProp achieve the highest performance, while SGD shows slightly lower accuracy and F1-score.

---

### 4. Learning Rate Evaluation

| Learning Rate | Accuracy | Precision | Recall | F1-Score | Loss   |
| ------------- | -------- | --------- | ------ | -------- | ------ |
| 3e-5          | 98.56%   | 96.30%    | 92.86% | 94.55%   | 0.0411 |
| 5e-4          | 99.52%   | 100.00%   | 96.43% | 98.18%   | 0.0281 |
| 1e-3          | 99.52%   | 100.00%   | 96.43% | 98.18%   | 0.0269 |

**Finding**: Higher learning rates yield better convergence and lower loss.

---

### 5. Data Augmentation (Synonym Replacement)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 99.28% |
| Precision | 99.07% |
| Recall    | 95.54% |
| F1-Score  | 97.27% |
| Loss      | 0.0332 |

**Observation**: Data augmentation improves overall robustness and F1-score.

---

### 6. Dropout Regularization

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 99.40% |
| Precision | 99.08% |
| Recall    | 96.43% |
| F1-Score  | 97.74% |
| Loss      | 0.0315 |

**Result**: Dropout regularization achieves the best overall performance.

---

### 7. Transfer Learning on External Dataset

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 98.78% |
| Precision | 96.41% |
| Recall    | 94.32% |
| F1-Score  | 95.36% |
| Loss      | 0.0365 |

**Finding**: The model generalizes well to unseen SMS datasets.

---

## Performance Metrics

### Best Overall Model

```
Accuracy:  99.40%
Precision: 99.08%
Recall:    96.43%
F1-Score:  97.74%
Loss:      0.0315
```

---

## Key Insights

1. Fine-tuning significantly improves spam detection performance.
2. Optimizer choice has a noticeable impact on convergence and loss.
3. Data augmentation enhances robustness and class balance.
4. Dropout regularization reduces overfitting.
5. The model demonstrates strong generalization across datasets.

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## Acknowledgments

* **Dataset**: SMS Spam Collection Dataset
* **Model**: HuggingFace DistilBERT
* **Frameworks**: PyTorch, Transformers, scikit-learn
