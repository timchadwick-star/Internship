# Machine Learning Task Specification for the HUSTmotor Multimodal Dataset

## 1. Task Title

Motor Health-State Classification Using Vibration and Acoustic Signals

## 2. Background

Electric motors may develop faults such as bearing damage, rotor bow, broken rotor bars, rotor misalignment, and voltage imbalance during long-term operation. These faults change the vibration and acoustic patterns of the motor. Machine learning methods can therefore be used to identify the motor health state from measured signals.

This task uses the HUSTmotor multimodal motor fault dataset. The dataset contains vibration and acoustic signals collected from motors under six health states and four operating conditions. The task focuses on classical machine learning methods and does not require complex deep learning models.

Dataset repository:

[HUSTmotor Multimodal Dataset on GitHub](https://github.com/CHAOZHAO-1/HUSTmotor-multi-modal-dataset)

## 3. Dataset Description

### 3.1 Health States

The dataset contains six classes:

| Class Code | Health State |
|---|---|
| H | Healthy |
| BF | Bearing fault |
| BOW | Bowed rotor |
| BROKEN | Broken rotor bars |
| MISAL | Rotor misalignment |
| UNBAL | Voltage imbalance |

### 3.2 Operating Conditions

The dataset contains four operating conditions:

| Condition ID | Operating Frequency |
|---|---|
| C1 | 5 Hz |
| C2 | 10 Hz |
| C3 | 20 Hz |
| C4 | 30 Hz |

### 3.3 Sampling Information

The sampling frequency is 25.6 kHz. Each recording contains 163,840 data points, corresponding to 6.4 seconds of continuous signal. The raw dataset contains 24 health-state and operating-condition combinations.

Before processing the data, read the dataset description carefully and confirm the vibration channels, acoustic channels, delimiters, headers, and measurement units. Channel meaning must not be inferred only from column position.

## 4. Task Objective

Build a six-class machine learning model that predicts the motor health state from a segment of vibration and acoustic signals.

The task should answer the following questions:

1. Can the six health states be identified using vibration signals only?
2. Can the six health states be identified using acoustic signals only?
3. Does feature-level fusion of vibration and acoustic signals improve classification performance?

The final model should also be evaluated on an operating condition that is not used during training, so that cross-condition classification performance can be measured.

## 5. Input and Output

### 5.1 Input

Each sample is a fixed-length signal window containing:

- a vibration-signal segment;
- an acoustic-signal segment from the same time interval;
- operating-condition information used only for data splitting and result analysis.

The operating condition should not be used as a main classification feature.

### 5.2 Output

The model should output one of the following six health-state labels:

```text
H, BF, BOW, BROKEN, MISAL, UNBAL
```

The model should also output class probabilities or decision scores when supported by the selected classifier.

## 6. Data Preprocessing

### 6.1 File Parsing

Extract the health-state label and operating condition from each file name. For example:

```text
H_5HZ
```

represents a healthy motor operating at 5 Hz.

Create a data index with the following structure:

| file_id | health_label | condition | modality | file_path |
|---|---|---|---|---|

### 6.2 Signal Validation

Check each file for the following problems:

- missing values, infinite values, or invalid characters;
- signal length inconsistent with the dataset description;
- time misalignment between vibration and acoustic signals;
- all-zero segments, saturated signals, or interrupted recordings;
- inconsistent sampling frequencies.

### 6.3 Mean Removal

Remove the direct-current component from each signal window:

\[
x'_i=x_i-\frac{1}{N}\sum_{j=1}^{N}x_j
\]

where \(N\) is the number of samples in the window.

### 6.4 Signal Segmentation

Use the following default settings:

- window duration: 0.5 seconds;
- window length: 12,800 samples;
- overlap ratio: 50%;
- sliding step: 6,400 samples.

Each 6.4-second file will produce about 24 windows. The full dataset will therefore produce about 576 samples. The exact sample count should be determined from the actual file lengths and the segmentation program.

Each window must retain its original `file_id`. Highly similar windows from the same raw file must not be placed in both the training and validation sets.

## 7. Feature Extraction

Extract features separately from the vibration and acoustic signals.

### 7.1 Time-Domain Features

Extract at least the following features from each signal window:

- mean;
- standard deviation;
- root mean square;
- maximum;
- minimum;
- peak-to-peak value;
- skewness;
- kurtosis;
- shape factor;
- crest factor;
- impulse factor.

Some ratio-based features may become unstable when the denominator is close to zero. Add a small positive constant \(\epsilon\) and check for abnormal values.

### 7.2 Frequency-Domain Features

Apply the fast Fourier transform to each mean-removed window and extract at least the following features:

- dominant frequency;
- spectral centroid;
- spectral bandwidth;
- spectral entropy;
- total spectral energy;
- energy ratios of eight equal-width frequency bands.

Use the one-sided frequency spectrum and construct the frequency axis using the actual sampling frequency.

### 7.3 Feature Standardization

Feature-standardization parameters must be calculated using the training data only:

\[
z=\frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}+\epsilon}
\]

The validation and test sets must use the mean and standard deviation calculated from the training set.

### 7.4 Multimodal Feature Fusion

Use feature-level fusion by concatenating vibration and acoustic features:

\[
\mathbf{f}_{\mathrm{fusion}}
=
[\mathbf{f}_{\mathrm{vibration}};
\mathbf{f}_{\mathrm{acoustic}}]
\]

Construct at least the following three input settings:

- Vibration: vibration features only;
- Acoustic: acoustic features only;
- Fusion: concatenated vibration and acoustic features.

## 8. Model Requirements

Implement at least the following three classifiers:

1. majority-class predictor as the minimum baseline;
2. support vector machine;
3. random forest.

Logistic regression or k-nearest neighbors may be added as optional models.

Recommended parameter ranges are shown below:

| Model | Suggested Parameters |
|---|---|
| Support vector machine | RBF kernel, \(C\in\{0.1,1,10,100\}\), and \(\gamma\) set to `scale` or selected through grid search |
| Random forest | 100, 300, or 500 trees; maximum depth of 5, 10, 20, or unlimited |
| Logistic regression | \(C\in\{0.1,1,10\}\), with at least 1,000 maximum iterations |
| k-nearest neighbors | \(k\in\{3,5,7,9\}\) |

Model parameters must be selected through cross-validation within the training data. The test set must not be used for parameter selection.

## 9. Experimental Design

### 9.1 Experiment 1: Single-Modality and Multimodal Comparison

Use the same data split and classifier for the following input settings:

| Experiment ID | Input |
|---|---|
| E1 | Vibration features |
| E2 | Acoustic features |
| E3 | Fused vibration and acoustic features |

This experiment evaluates the information provided by each modality and whether multimodal fusion improves classification performance.

### 9.2 Experiment 2: Leave-One-Condition-Out Evaluation

Perform four rounds of leave-one-condition-out evaluation:

| Round | Training Conditions | Test Condition |
|---|---|---|
| T1 | 10, 20, and 30 Hz | 5 Hz |
| T2 | 5, 20, and 30 Hz | 10 Hz |
| T3 | 5, 10, and 30 Hz | 20 Hz |
| T4 | 5, 10, and 20 Hz | 30 Hz |

In each round, all samples from the test condition must be used only for final testing.

Within the three training conditions, use grouped cross-validation based on `file_id` for model selection. Stratified group cross-validation is recommended so that all windows from the same raw file remain in the same fold.

After the four rounds, report the result for each test condition and the average result:

\[
\overline{M}=\frac{1}{4}\sum_{k=1}^{4}M_k
\]

where \(M_k\) is the evaluation metric obtained for the \(k\)-th test condition.

### 9.3 Data Leakage Restrictions

Do not randomly split all signal windows without grouping. Overlapping windows and windows from the same raw file are strongly correlated. Placing windows from the same file in both training and test sets may produce misleadingly high results.

The following operations also cause data leakage:

- calculating standardization parameters from the full dataset;
- selecting features using test data;
- adjusting model parameters based on test results;
- removing the original file identifier after segmentation;
- placing windows from the same raw file in different cross-validation folds.

## 10. Evaluation Metrics

Report at least the following metrics:

- Accuracy;
- Macro Precision;
- Macro Recall;
- Macro F1;
- per-class Recall;
- six-class confusion matrix.

Because data cleaning or segmentation may produce different numbers of samples for different classes, Macro F1 should be treated as the main metric. Accuracy should be used as a supporting metric.

For the four leave-one-condition-out rounds, report the mean and standard deviation.

## 11. Result Analysis Requirements

The report should answer the following questions:

1. Which single modality produces better classification results?
2. Does feature-level fusion consistently improve Macro F1?
3. Which operating condition is the most difficult test condition?
4. Which two fault classes are most often confused?
5. How do the random forest and support vector machine results differ?
6. Are the main errors caused by limited model capacity, weak features, or operating-condition changes?
7. Does one modality fail under a specific operating condition?

The report should include:

- a result table comparing the three input settings;
- a bar chart of Macro F1 for the four test conditions;
- a confusion matrix for the best model;
- random-forest feature importance or logistic-regression coefficients;
- time-domain and frequency-domain plots of selected correct and incorrect samples.

## 12. Minimum Completion Requirements

The basic task is complete when all of the following requirements are met:

- vibration and acoustic data are read correctly;
- signal segmentation and label generation are completed;
- time-domain and frequency-domain features are extracted;
- at least two effective classifiers are implemented;
- vibration, acoustic, and fusion experiments are completed;
- four leave-one-condition-out experiments are completed;
- Accuracy, Macro F1, and confusion matrices are reported;
- all results can be reproduced from the raw data;
- no clear data leakage is present.

A fixed accuracy threshold is not recommended as the only acceptance condition because the result depends on data cleaning, channel selection, window settings, and model parameters. The main checks should focus on experimental correctness, reproducibility, and reasonable result analysis.

## 13. Optional Extensions

After completing the basic task, choose one optional extension:

- compare window lengths of 0.25, 0.5, and 1 second;
- apply principal component analysis for dimensionality reduction;
- use feature selection to remove weak features;
- compare early feature fusion with late probability fusion;
- simulate a missing modality and test model stability;
- add a one-dimensional convolutional neural network for comparison;
- use t-SNE or UMAP to display feature distributions under different conditions.

All extension experiments must continue to use grouped data splitting based on raw files or operating conditions.

## 14. Submission Requirements

### 14.1 Code

A recommended project structure is:

```text
HUSTmotor_fault_classification/
├── data/
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   └── evaluate.py
├── configs/
│   └── default.yaml
├── results/
│   ├── metrics.csv
│   ├── confusion_matrix/
│   └── figures/
├── requirements.txt
├── README.md
└── report.pdf
```

### 14.2 Experimental Results

Submit the following result files:

- metrics for each leave-one-condition-out round;
- average metrics over the four rounds;
- Precision, Recall, and F1 for each class;
- confusion matrix for the best model;
- single-modality and multimodal comparison results;
- selected model parameters;
- random seed and software-version records.

### 14.3 Project Report

The report should contain:

1. task background;
2. dataset description;
3. data preprocessing;
4. feature-extraction methods;
5. classification models;
6. experimental settings;
7. experimental results;
8. error analysis;
9. conclusions and limitations.


## 15. Reproducibility Requirements

Record the following information:

- Python version;
- main library versions;
- random seed;
- window length and overlap ratio;
- feature list;
- data-splitting rules;
- model parameters;
- training and test conditions for each round;
- abnormal-data handling rules.

Use `requirements.txt` or an environment configuration file to save software dependencies. A main program should reproduce the complete process from raw-data loading to final result generation.
