# Classification of Left- and Right-Hand Motor Imagery from EEG Data

## Overview

This repository contains a machine learning pipeline for classifying left-hand versus right-hand motor imagery from electroencephalography (EEG) signals. The project is based on the PhysioNet EEG Motor Movement/Imagery dataset (EEGBCI) and follows a classical motor imagery brain-computer interface (MI-BCI) workflow.

The main goal is to decode a user’s imagined hand movement from non-invasive EEG recordings by exploiting changes in sensorimotor rhythms over the motor cortex. The project emphasizes interpretable signal processing and machine learning methods, especially Common Spatial Patterns (CSP) and Filter Bank Common Spatial Patterns (FBCSP), together with linear classifiers such as LDA and SVM.

## Problem Statement

Motor imagery classification is a core problem in brain-computer interface research. In this task, the model receives epoched EEG trials and predicts whether the subject is imagining moving the left hand or the right hand.

This problem is challenging because EEG data have:
- low signal-to-noise ratio,
- strong trial-to-trial variability,
- subject-specific patterns,
- sensitivity to artifacts such as eye movement and muscle activity.

Because of these properties, successful motor imagery decoding depends heavily on preprocessing quality, feature extraction, and robust evaluation design.

## Dataset

The project uses the **PhysioNet EEG Motor Movement/Imagery Dataset (EEGBCI)**.

Key dataset characteristics:
- EEG recordings collected with the **BCI2000** system,
- **64-channel** montage based on the 10–10 system,
- focus on channels around **C3/C4** and neighboring sensorimotor areas,
- EDF/EDF+ file format,
- sampling frequency standardized to **160 Hz** in the pipeline,
- motor imagery runs selected for **left-hand vs. right-hand imagery** classification.

Subjects with too few valid trials or only one remaining class after preprocessing are excluded to ensure meaningful training and evaluation.

## Methodology

The pipeline follows a standard MI-BCI structure:

1. **Load EEG recordings**
2. **Preprocess signals**
3. **Segment into epochs**
4. **Extract CSP/FBCSP features**
5. **Train linear classifiers**
6. **Evaluate cross-subject and subject-dependent performance**

### 1. Preprocessing

The preprocessing pipeline is designed to preserve motor imagery-related neural activity while reducing noise and artifacts.

Main steps:
- **Resampling** to 160 Hz
- **Band-pass filtering** using FIR filters
- **Epoching** from pre-stimulus to post-stimulus windows
- **Baseline correction**
- **Artifact rejection** using peak-to-peak amplitude thresholds

The frequency range of interest is mainly the **mu band (8–12 Hz)** and **beta band (12–30 Hz)**, which are strongly associated with motor imagery through event-related desynchronization/synchronization (ERD/ERS).

### 2. Feature Extraction

#### Common Spatial Patterns (CSP)

CSP is used to learn spatial filters that maximize variance for one class and minimize it for the other. This helps isolate discriminative sensorimotor activity from mixed scalp-level EEG measurements.

The extracted CSP features are typically log-variance values of the projected signals.

#### Filter Bank CSP (FBCSP)

FBCSP extends CSP by applying CSP separately on multiple frequency sub-bands and concatenating the resulting features.

In this project, the filter bank covers:
- **8–12 Hz** (mu band),
- **12–16 Hz** (low beta),
- **16–30 Hz** (high beta).

This allows the model to capture subject-specific spectral variations more effectively than single-band CSP.

### 3. Classification

The project evaluates several linear models:
- **Linear Discriminant Analysis (LDA)**
- **Linear Support Vector Machine (SVM)**
- **LDA + SVM** as a two-stage configuration

These models are chosen because CSP/FBCSP features are often close to linearly separable, making linear classifiers strong and interpretable baselines.

## Evaluation Strategy

Two evaluation settings are used:

### Cross-Subject Evaluation
Data from multiple subjects are pooled, then split into train/test sets. This setting tests generalization across subjects and is much more difficult because EEG distributions vary strongly between individuals.

### Subject-Dependent Evaluation
A separate model is trained and evaluated for each subject. This reflects a calibration-based practical BCI setting and usually gives better performance.

## Evaluation Metrics

The project reports more than just accuracy to provide a more complete view of performance.

### Accuracy
Measures the proportion of correctly classified trials.

### Macro F1-score
Balances precision and recall across classes and is useful when class distributions are not perfectly balanced after artifact rejection.

### Cohen’s Kappa
Measures agreement corrected for chance, which is particularly useful in BCI evaluations.

## Results

The experiments show the expected pattern for motor imagery EEG classification:

- **Subject-dependent models outperform cross-subject models**
- **FBCSP generally improves over single-band CSP**
- **Linear SVM performs slightly better than or similar to LDA**

Reported results include:

### Cross-Subject
- Best FBCSP + SVM accuracy: **about 0.6203**

### Subject-Dependent
- Best mean accuracy: **about 0.641**
- Macro F1-score: **about 0.619**
- Cohen’s kappa: **about 0.280**

These results indicate that:
1. classical CSP/FBCSP pipelines remain effective baselines,
2. subject variability is a major obstacle in cross-subject EEG decoding,
3. richer frequency-aware features improve motor imagery discrimination.

## Key Takeaways

This project highlights several important lessons in EEG motor imagery classification:

- good preprocessing matters as much as the classifier,
- spatial filtering is crucial for extracting useful EEG structure,
- frequency-specific feature design improves performance,
- cross-subject generalization remains difficult,
- simple linear models can still be strong baselines when feature engineering is done properly.

## Repository Purpose

This repository is intended to serve as:
- a reproducible academic project on MI-BCI,
- a reference implementation of CSP/FBCSP-based EEG classification,
- a learning resource for students starting with EEG machine learning,
- a baseline before exploring deep learning or domain adaptation methods.

## Limitations

Current limitations of the project include:
- strong dependence on subject-specific calibration,
- limited robustness to inter-subject variability,
- reliance on classical handcrafted features,
- basic artifact rejection rather than advanced denoising,
- no domain adaptation or transfer learning.

## Future Work

Possible extensions include:
- balanced accuracy and calibration analysis,
- Riemannian geometry-based covariance classifiers,
- transfer learning across subjects,
- adaptive online calibration,
- deep learning models for EEG decoding,
- more systematic ablation studies comparing CSP, single-band CSP, and FBCSP.

## Tech Stack

- **Python**
- **MNE**
- **scikit-learn**
- **NumPy / SciPy**
- **Matplotlib / visualization tools**

## Conclusion

This repository demonstrates a complete and interpretable pipeline for left- versus right-hand motor imagery EEG classification. By combining neurophysiology-informed preprocessing, CSP/FBCSP feature extraction, and linear classification, the project provides a solid baseline for motor imagery decoding and a practical foundation for more advanced BCI research.
