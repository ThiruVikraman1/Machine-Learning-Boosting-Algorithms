# Machine Learning: Boosting Algorithms

This repository is dedicated to the study, implementation, and comparison of advanced Boosting algorithms. The project focuses on improving predictive accuracy for regression tasks using sequential ensemble learning techniques.

## 📂 Project Overview
In this assignment, I explore how Boosting algorithms can minimize residual errors to outperform traditional bagging methods. The primary dataset used is the **Insurance Premium Dataset**, where the goal is to predict medical charges based on patient attributes.

## 🧠 Algorithms Explored
The following state-of-the-art boosting models are implemented and tuned in this repository:

1. **AdaBoost (Adaptive Boosting)**: Focuses on "hard-to-predict" instances by adjusting weights sequentially.
2. **XGBoost (Extreme Gradient Boosting)**: A high-performance implementation featuring L1/L2 regularization and parallel processing.
3. **LightGBM (Light Gradient Boosting Machine)**: A memory-efficient model using leaf-wise tree growth for faster training on large data.

## 📊 Dataset: Insurance Prediction
- **Input Features**: Age, Sex, BMI, Children, Smoker status.
- **Target Variable**: Insurance Charges.
- **Goal**: To exceed the 87% $R^2$ score baseline established in previous Random Forest assignments.

**🔬 Comparison Strategy**
For each model, I perform a grid search over key hyperparameters:

Learning Rate (eta): To control the step size of each iteration.

n_estimators: To find the optimal number of boosting rounds.

max_depth: To manage model complexity and prevent overfitting.

**📈 Key Findings**
XGBoost provided the best balance between training speed and accuracy due to its built-in regularization.

LightGBM demonstrated superior performance in terms of computational memory usage.

## 🛠️ Installation & Setup
To run these notebooks, you will need to install the following specific boosting libraries:

```bash
pip install xgboost
pip install lightgbm

Author: Thiru Vikraman
