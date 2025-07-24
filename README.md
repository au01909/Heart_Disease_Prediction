
# Heart Disease Prediction

This project provides a simplified pipeline for predicting heart disease risk using machine learning (XGBoost) on the classic UCI Heart Disease dataset. It is designed for reproducible experiments and model logging using MLflow.

## Features

* Loads and preprocesses the UCI Heart Disease dataset (from CSV or UCI repository)
* Handles missing values with imputation
* Selects key features for prediction
* Splits data into training, validation, and test sets
* Scales features for model training
* Trains an XGBoost classifier and logs experiments via MLflow
* Evaluates the model with accuracy, AUC, confusion matrix, and classification report
* Generates ROC and confusion matrix plots
* Predicts heart disease risk for new patient data

## Getting Started

### Prerequisites

* Python 3.x
* Required Python packages:

  * pandas
  * numpy
  * scikit-learn
  * xgboost
  * mlflow
  * imbalanced-learn
  * matplotlib
  * seaborn
  * ucimlrepo
  * joblib

Install all dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost mlflow imbalanced-learn matplotlib seaborn ucimlrepo joblib
```

### Dataset

The notebook expects a CSV file named `heart.csv` in the project directory. You can fetch the dataset using the `ucimlrepo` package or manually download it from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).

### Running the Notebook

Open and run `Heart_Disease_Prediction.ipynb` in Jupyter Notebook, Google Colab, or any compatible IDE.

The notebook will:

* Load and preprocess the data
* Train and evaluate the model
* Log metrics and artifacts with MLflow
* Show sample predictions for new patient data

## Example Prediction

```python
new_patient_data = {
    'age': 52,
    'sex': 1,
    'cp': 0,
    'trestbps': 125,
    'chol': 212,
    'thalach': 168
}
probability, prediction = predict_heart_disease(model, new_patient_data, imputer, scaler, feature_names)
print(f"Probability of Heart Disease: {probability:.4f}")
print(f"Prediction: {'Heart Disease Present' if prediction == 1 else 'No Heart Disease'}")
```

## MLflow Tracking

All model runs and metrics are tracked with MLflow. To launch the MLflow UI:

```bash
mlflow ui
```

## Results

Sample output:

```
Accuracy: 0.9805, AUC: 1.0000
Confusion Matrix:
[[ 96   4]
 [  0 105]]

Classification Report:
              precision    recall  f1-score   support
           0       1.00      0.96      0.98       100
           1       0.96      1.00      0.98       105

Prediction Results:
  Probability of Heart Disease: 0.1401
  Prediction: No Heart Disease
```

## Project Structure

```
├── Heart_Disease_Prediction.ipynb
├── heart.csv
├── README.md
└── mlruns/  (MLflow artifacts and logs)
```

## License

This project is for educational purposes. Refer to the [UCI dataset license](https://archive.ics.uci.edu/ml/about.html) for terms regarding data usage.

## References

* [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
* [scikit-learn documentation](https://scikit-learn.org/)
* [MLflow documentation](https://mlflow.org/)
* [XGBoost documentation](https://xgboost.readthedocs.io/)


