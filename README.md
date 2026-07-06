# Donor Response Prediction and Used Clothes Image Clustering

This project demonstrates an end-to-end machine learning workflow that combines structured donor response prediction with unsupervised image clustering to support used-clothes condition review.

The structured-data task predicts whether a donor is likely to donate. The image task groups unlabelled used-clothes images into visually similar clusters that can support later manual review or semi-supervised labelling.

## Project Overview

The project contains two complementary machine learning workflows:

1. **Donor response prediction**
   - Builds supervised classification models using tabular donor data.
   - Removes target leakage by excluding `AMOUNT`.
   - Excludes `ID` from predictive features.
   - Applies train-only preprocessing before model evaluation.
   - Compares Logistic Regression and Random Forest.

2. **Used-clothes image clustering**
   - Extracts image features using a pretrained VGG16 convolutional neural network.
   - Reduces high-dimensional CNN embeddings with PCA.
   - Runs K-Means clustering across multiple values of `k`.
   - Evaluates cluster quality using Davies-Bouldin Index.
   - Reviews candidate cluster counts using DBI, PCA visualization, and representative images.

## Dataset Description

The structured dataset contains donor records with demographic, behavioural, and donation-history variables. The target variable is `DONATE`, converted into a binary label:

- `No` = 0
- `Yes` = 1

The image dataset contains unlabelled photos of used clothes. Since the images are not pre-labelled as damaged or undamaged, clustering is used to create groups for manual inspection.

Data files are not included in this project package because of copyright and redistribution restrictions. To run the notebook, place the local data files under `data/raw/`. See [`data/README.md`](data/README.md) for details.

## Methods

### Structured Data

- Median imputation for numeric variables including `AGE` and `WEALTH_RATING`.
- Mode imputation for `SES`, retained as a numeric/ordinal-style feature.
- Missing `URBANICITY` values filled with `"MISSING"` before One-Hot Encoding.
- One-Hot Encoding for categorical variables such as `HOME_OWNER`, `GENDER`, and `STATUS`.
- Stratified 80/20 train-test split.
- Logistic Regression and Random Forest classifiers.
- Evaluation with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, ROC curves, and feature importance.

### Image Clustering

- VGG16 pretrained on ImageNet for CNN feature extraction.
- Flattened CNN features reduced to 50 PCA components.
- First two PCA components used for two-dimensional visualization.
- K-Means evaluated for `k = 2` to `k = 20`.
- Candidate cluster counts fixed at `k = 3`, `k = 5`, and `k = 6` for detailed review.
- Final cluster count selected based on Davies-Bouldin Index, PCA separation, and visual consistency of representative images.
- Manual cluster-level assessment recorded as `damaged`, `undamaged`, or `unclear`.

## Key Outputs

The notebook produces:

- Classification metrics for Logistic Regression and Random Forest.
- Confusion matrices and ROC curves.
- Feature importance plots.
- Davies-Bouldin Index comparison across cluster counts.
- PCA cluster visualizations.
- Representative images closest to each cluster centroid.
- `outputs/clustering_results.csv`, containing each image filename, cluster label, and manual cluster assessment.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- notebooks/
|   `-- donor_prediction_and_image_clustering.ipynb
|-- data/
|   `-- README.md
|-- outputs/
|   `-- README.md
`-- assets/
    `-- README.md
```

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place local data files under `data/raw/`:

   ```text
   data/raw/DONOR.csv
   data/raw/used_clothes_images.zip
   ```

3. Open and run the notebook:

   ```text
   notebooks/donor_prediction_and_image_clustering.ipynb
   ```

4. Review the image clustering section:
   - The notebook allows users to review candidate cluster counts and assign manual cluster-level assessments.
   - For reproducibility, the reviewed final setting is controlled by the `FINAL_K` and `cluster_assessments` variables in the notebook.
