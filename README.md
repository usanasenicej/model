# Iris Species Intelligence - Interactive ML Dashboard

This is a complex Machine Learning project that pairs an **R-based Random Forest model** with a **High-Performance Shiny Web Application**. 

## ✨ Key Features
- **Interactive Prediction Lab**: Real-time species prediction using physical measurements.
- **Visual Analytics Dashboard**:
    - **Pairwise EDA**: Deep dive into feature correlations using `GGally`.
    - **Feature Importance**: Understand what drives the model's decisions.
- **Model Diagnostics**: Live confusion matrix and accuracy metrics from cross-validation.
- **Modern UI**: Styled with the **Lux** premium theme, featuring glassmorphism elements and responsive layouts via `bslib`.
- **Data Explorer**: Interactive table for browsing the raw Iris dataset.

## 🚀 Prerequisites
To run this application, you need R installed along with these professional grade packages:
```r
install.packages(c("shiny", "bslib", "caret", "ggplot2", "randomForest", "GGally", "DT", "shinycssloaders", "bsicons"))
```

## 🛠️ How to Launch
1. Open R or RStudio.
2. Run the following command:
   ```r
   shiny::runApp("app.R")
   ```
3. The dashboard will open in your default browser at `http://127.0.0.1:xxxx`.

## 📦 Project Structure
- `app.R`: The main interactive application (UI + Server).
- `ml_model.R`: A standalone script for batch training and static plot generation.
- `iris_pairs.png`, `confusion_matrix.png`, `feature_importance.png`: Sample outputs from the static script.
