# Default of Credit Card Clients

## Overview
This project focuses on analyzing and predicting credit card client defaults using the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository. The goal is to achieve higher accuracy and F1 scores in predicting the `dpnm` column class compared to the baseline results provided in the original research paper.

---

## Objectives
- **Analyze** the dataset to uncover key insights about credit card client behavior.
- **Predict** the likelihood of defaults using various classification models.
- **Optimize** model performance to improve accuracy and F1 scores over the benchmarks presented in the original paper.

---

## Machine Learning Models Used
The following models were implemented to classify the `dpnm` column:

1. **Logistic Regression**: A fundamental yet powerful model for binary classification.
2. **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm.
3. **Random Forest**: An ensemble learning method leveraging decision trees.
4. **Decision Tree**: A tree-structured algorithm for classification tasks.
5. **Naive Bayes (Gaussian NB)**: A probabilistic classifier based on Bayes' theorem.
6. **Neural Networks**: Leveraging deep learning for complex decision boundaries.

---

## Methodology
1. **Data Preprocessing**:
   - Handled missing values and outliers.
   - Standardized and normalized feature values to improve model performance.

2. **Feature Engineering**:
   - Explored correlations between features.
   - Selected the most impactful features for classification.

3. **Model Training & Evaluation**:
   - Split the dataset into training and testing sets.
   - Trained and tuned each model using hyperparameter optimization.
   - Evaluated performance using accuracy and F1 scores.

4. **Comparison with Benchmarks**:
   - Compared model performance against the results in the original paper.
   - Focused on exceeding the baseline accuracy and F1 scores.

---

## Results
- **Logistic Regression**: Delivered strong baseline performance.
- **Random Forest**: Achieved the highest accuracy among all models.
- **Neural Networks**: Provided competitive results with potential for further optimization.
- **Naive Bayes**: Worked well with Gaussian-distributed features.

---

## Key Insights
- Ensemble models like Random Forest performed better than standalone classifiers.
- Neural Networks demonstrated potential for capturing non-linear relationships in the data.
- Data preprocessing and feature selection significantly influenced model outcomes.

---

## Tools & Libraries
- **Python**: Programming language for implementation.
- **Scikit-learn**: Machine learning library for model training and evaluation.
- **Pandas & NumPy**: Libraries for data manipulation and analysis.
- **Matplotlib & Seaborn**: Visualization libraries for data exploration.

---

## Future Enhancements
- Incorporate additional features to capture behavioral patterns.
- Experiment with advanced ensemble methods like Gradient Boosting and XGBoost.
- Leverage hyperparameter optimization techniques such as Grid Search and Bayesian Optimization.

---

## Author
This project was developed by **Amit Raj** to explore practical applications of machine learning in financial prediction tasks.

---
