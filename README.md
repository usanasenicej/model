# Iris Species Classification Model

This project implements a Machine Learning model using **R**, the **caret** package, and **ggplot2** for visualization. It uses a Random Forest classifier to predict the species of an iris flower based on its physical measurements.

## Features
- **Exploratory Data Analysis (EDA)**: Visualizes relationships between features using `GGally::ggpairs`.
- **Random Forest Classification**: Robust and accurate model training with cross-validation.
- **Model Evaluation**: Detailed performance metrics including a Confusion Matrix.
- **Beautiful Visualizations**: Professional-grade plots for understanding model performance and feature importance.

## Prerequisites
Ensure you have R installed and the following packages:
```r
install.packages(c("caret", "ggplot2", "GGally", "randomForest"))
```

## How to Run
1. Open your R environment or terminal.
2. Run the script:
   ```bash
   Rscript ml_model.R
   ```
3. Check the output files generated in the project directory:
   - `iris_pairs.png`: Pairwise plot of the dataset.
   - `confusion_matrix.png`: Heatmap of model predictions.
   - `feature_importance.png`: Ranking of features by their predictive power.

## Model Summary
- **Dataset**: Iris (150 observations, 5 variables)
- **Algorithm**: Random Forest
- **Validation**: 5-fold Cross-Validation
- **Target**: `Species` (Setosa, Versicolor, Virginica)
