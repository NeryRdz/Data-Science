<div align="center">

# 🧠 Data Science Repository

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A comprehensive collection of Data Science notebooks, scripts, and projects — covering the full pipeline from raw data to deployed models.

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Topics Covered](#topics-covered)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Overview

This repository serves as a **personal knowledge base and portfolio** for Data Science work. It is organized to follow the natural flow of a DS project:

> **Data → Cleaning → Exploration → Visualization → Modeling → Evaluation**

Whether you're revisiting fundamentals or exploring advanced techniques, everything lives here — documented, reproducible, and ready to run.

---

## 🗂️ Repository Structure

```
📦 data-science/
├── 📁 01_data_manipulation/
│   ├── pandas_basics.ipynb
│   ├── numpy_operations.ipynb
│   ├── data_cleaning.ipynb
│   └── feature_engineering.ipynb
│
├── 📁 02_exploratory_data_analysis/
│   ├── eda_template.ipynb
│   ├── statistical_summaries.ipynb
│   └── correlation_analysis.ipynb
│
├── 📁 03_data_visualization/
│   ├── matplotlib_fundamentals.ipynb
│   ├── seaborn_charts.ipynb
│   ├── plotly_interactive.ipynb
│   └── storytelling_with_data.ipynb
│
├── 📁 04_machine_learning/
│   ├── 📁 supervised/
│   │   ├── linear_regression.ipynb
│   │   ├── logistic_regression.ipynb
│   │   ├── decision_trees.ipynb
│   │   ├── random_forest.ipynb
│   │   ├── gradient_boosting.ipynb
│   │   └── svm.ipynb
│   │
│   ├── 📁 unsupervised/
│   │   ├── kmeans_clustering.ipynb
│   │   ├── pca_dimensionality.ipynb
│   │   └── dbscan.ipynb
│   │
│   └── 📁 model_evaluation/
│       ├── cross_validation.ipynb
│       ├── metrics_classification.ipynb
│       ├── metrics_regression.ipynb
│       └── hyperparameter_tuning.ipynb
│
├── 📁 05_pipelines/
│   ├── sklearn_pipelines.ipynb
│   └── preprocessing_pipeline.ipynb
│
├── 📁 06_projects/
│   └── (end-to-end DS projects)
│
├── 📁 data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
└── README.md
```

---

## 🧩 Topics Covered

### 🔧 1. Data Manipulation
- **Pandas**: DataFrames, Series, merging, groupby, pivoting, time series
- **NumPy**: Arrays, broadcasting, vectorized operations, linear algebra
- **Data Cleaning**: Handling missing values, duplicates, outliers, type casting
- **Feature Engineering**: Encoding, scaling, binning, creating new features

### 🔍 2. Exploratory Data Analysis (EDA)
- Descriptive statistics and distributions
- Correlation matrices and heatmaps
- Outlier detection methods
- Hypothesis testing basics

### 📊 3. Data Visualization
| Library | Use Case |
|---|---|
| `Matplotlib` | Fine-grained, publication-quality plots |
| `Seaborn` | Statistical visualizations with style |
| `Plotly` | Interactive charts and dashboards |

### 🤖 4. Machine Learning with Scikit-learn
**Supervised Learning**
- Regression: Linear, Ridge, Lasso, ElasticNet
- Classification: Logistic Regression, SVM, Decision Trees, Random Forest, Gradient Boosting (XGBoost, LightGBM)

**Unsupervised Learning**
- Clustering: K-Means, DBSCAN, Hierarchical
- Dimensionality Reduction: PCA, t-SNE, UMAP

**Model Evaluation & Tuning**
- Cross-validation strategies (KFold, StratifiedKFold)
- Metrics: Accuracy, F1, ROC-AUC, RMSE, MAE, R²
- Hyperparameter tuning: GridSearchCV, RandomizedSearchCV, Optuna

### ⚙️ 5. Sklearn Pipelines
- Building end-to-end `Pipeline` objects
- `ColumnTransformer` for mixed-type preprocessing
- Saving and loading pipelines with `joblib`

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip` or `conda`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/data-science.git
cd data-science

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

### `requirements.txt` snapshot

```
numpy>=1.26
pandas>=2.0
matplotlib>=3.8
seaborn>=0.13
plotly>=5.20
scikit-learn>=1.4
jupyter>=1.0
joblib>=1.3
scipy>=1.12
xgboost>=2.0
lightgbm>=4.3
optuna>=3.6
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **Jupyter Notebook** | Interactive development |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn / Plotly** | Visualization |
| **Scikit-learn** | ML modeling & pipelines |
| **XGBoost / LightGBM** | Gradient boosting models |
| **Optuna** | Hyperparameter optimization |
| **Joblib** | Model serialization |

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-topic`
3. Commit your changes: `git commit -m "Add notebook on topic X"`
4. Push to the branch: `git push origin feature/your-topic`
5. Open a Pull Request

Please keep notebooks **clean and well-documented** — restart the kernel and run all cells before committing.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ and a lot of <code>df.head()</code>
</div>
