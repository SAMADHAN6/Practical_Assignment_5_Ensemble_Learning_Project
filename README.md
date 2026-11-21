# Practical 5: Ensemble Learning and Performance Evaluation

## Overview
Comparative performance evaluation of base machine learning classifiers (**Logistic Regression**, **Decision Tree**, **Decision Tree (depth=3)**, **KNN**, **Gaussian Naïve Bayes**, **SVM**) and ensemble learning methods (**Voting (Hard)**, **Stacking**, **AdaBoost**) on the **Scikit-learn Handwritten Digits Dataset** (1,797 samples, 64 features, 10 classes).

## Project Structure
- `main.py`: Main Python script implementing base and ensemble models, 5-fold stratified cross-validation, metric evaluation, and visualization.
- `digits_dataset.csv`: Exported Handwritten Digits dataset (`1797 x 64` pixel features + target digit `0-9`).
- `outputs/model_comparison.csv`: Evaluation metrics table.
- `outputs/ensemble_comparison.png`: Bar chart comparing Accuracy and F1-Score across all 9 models.
- `outputs/sample_output.txt`: Full terminal output log.
- `requirements.txt`: Required Python dependencies.

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
# Practical_Assignment_5_Ensemble_Learning_Project
