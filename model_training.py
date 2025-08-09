import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Setup logging
logging.basicConfig(
    filename="model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    # Load dataset
    df = pd.read_csv("heart.csv")
    logging.info(f"Dataset loaded successfully with shape {df.shape}")

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df.fillna(df.median(), inplace=True)
        logging.warning("Missing values found and filled with median")

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Feature selection - top 5
    selector = SelectKBest(score_func=f_classif, k=5)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    logging.info(f"Selected top features: {selected_features}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X[selected_features], y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    # Train & cross-validate
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        avg_score = np.mean(scores)
        logging.info(f"{name} CV accuracy: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_model_name = name

    # Fit best model
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    logging.info(f"Best model: {best_model_name} | Test Accuracy: {test_acc:.4f}")

    # Save model & scaler
    pickle.dump(best_model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(selected_features, open("features.pkl", "wb"))
    logging.info("Model, scaler & features saved successfully.")

except Exception as e:
    logging.error("Error during model training", exc_info=True)
    print(f"Model training failed: {e}")
