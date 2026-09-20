"""
Model Training - Production Script (runs inside the GitHub Actions job)
-------------------------------------------------------------------------
Loads the Xtrain/Xtest/ytrain/ytest splits produced by prep.py, builds a
shared preprocessing pipeline (scaling + one-hot encoding), then trains
and tunes five candidate algorithms - Decision Tree, Random Forest,
AdaBoost, Gradient Boosting and XGBoost - with GridSearchCV. Every
parameter combination tried is logged to MLflow as a nested run, the
best configuration of each algorithm is logged as its own run, and the
overall best model (by test ROC-AUC) is registered by saving it into
tourism_project/deployment/ so the Streamlit app can load it directly.
"""
# for data manipulation
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")   # local MLflow server started by the workflow job
mlflow.set_experiment("tourism-wellness-package-prediction")     # MLflow experiment name (same as the dev experimentation cell)

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "PitchSatisfactionScore",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]   # all numerical feature names (same as in prep.py)

categorical_features = [
    "TypeofContact", "CityTier", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Passport", "OwnCar", "Designation",
]   # all categorical feature names (same as in prep.py)

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Set classification threshold
# Chosen to favour recall on the minority (buyer) class: for a sales-targeting
# use case, the cost of skipping a genuine buyer (false negative) outweighs the
# cost of an extra follow-up call to a customer who was not going to buy.
classification_threshold = 0.45

# --- Candidate algorithms and their tuning grids -------------------------
CANDIDATES = {
    "decision_tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {
            "decisiontreeclassifier__max_depth": [4, 6, 8, 10, None],
            "decisiontreeclassifier__min_samples_leaf": [1, 5, 10],
            "decisiontreeclassifier__criterion": ["gini", "entropy"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {
            "randomforestclassifier__n_estimators": [200, 400],
            "randomforestclassifier__max_depth": [8, 12, None],
            "randomforestclassifier__min_samples_leaf": [1, 3],
        },
    ),
    "adaboost": (
        AdaBoostClassifier(random_state=42),
        {
            "adaboostclassifier__n_estimators": [100, 200, 300],
            "adaboostclassifier__learning_rate": [0.05, 0.1, 0.5, 1.0],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "gradientboostingclassifier__n_estimators": [150, 250],
            "gradientboostingclassifier__max_depth": [2, 3, 4],
            "gradientboostingclassifier__learning_rate": [0.05, 0.1],
        },
    ),
    "xgboost": (
        xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss"),
        {
            # Number of boosting trees. More trees can improve performance but increase training time.
            'xgbclassifier__n_estimators': [200, 300],
            # Maximum depth of each tree. Higher values increase model complexity and risk of overfitting.
            'xgbclassifier__max_depth': [3, 4, 5],
            # Fraction of features sampled when building each tree.
            'xgbclassifier__colsample_bytree': [0.8, 1.0],
            # Fraction of features sampled at each tree level.
            'xgbclassifier__colsample_bylevel': [0.8, 1.0],
            # Step size used during boosting. Smaller values may improve generalization but require more trees.
            'xgbclassifier__learning_rate': [0.05, 0.1],
            # L2 regularization strength. Higher values help reduce overfitting.
            'xgbclassifier__reg_lambda': [1.0, 2.0],
        },
    ),
}

overall_results = {}
overall_best = {"name": None, "roc_auc": -1, "estimator": None}

with mlflow.start_run(run_name="model_comparison_parent"):
    for name, (estimator, param_grid) in CANDIDATES.items():
        # Model pipeline: chain preprocessor and the candidate estimator
        model_pipeline = make_pipeline(preprocessor, estimator)

        with mlflow.start_run(run_name=name, nested=True):
            # Hyperparameter tuning with GridSearchCV
            grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
            grid_search.fit(Xtrain, ytrain)

            # Log every parameter combination tried during the search as a nested run,
            # so all experiments can be compared side by side in the MLflow UI
            results = grid_search.cv_results_
            for i in range(len(results["params"])):
                with mlflow.start_run(run_name=f"{name}_trial_{i}", nested=True):
                    mlflow.log_params(results["params"][i])
                    mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
                    mlflow.log_metric("std_test_score", results["std_test_score"][i])

            # Log the best hyperparameters in this algorithm's main run
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("algorithm", name)

            # Store the best model for this algorithm
            best_model = grid_search.best_estimator_

            # Make predictions on the training and test data
            y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
            y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

            y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
            y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

            # Evaluation
            train_report = classification_report(ytrain, y_pred_train, output_dict=True)
            test_report = classification_report(ytest, y_pred_test, output_dict=True)
            test_roc_auc = roc_auc_score(ytest, y_pred_test_proba)

            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_report['accuracy'],
                "train_precision": train_report['1']['precision'],
                "train_recall": train_report['1']['recall'],
                "train_f1-score": train_report['1']['f1-score'],
                "test_accuracy": test_report['accuracy'],
                "test_precision": test_report['1']['precision'],
                "test_recall": test_report['1']['recall'],
                "test_f1-score": test_report['1']['f1-score'],
                "test_roc_auc": test_roc_auc,
            })

            overall_results[name] = {
                "test_accuracy": test_report['accuracy'],
                "test_precision": test_report['1']['precision'],
                "test_recall": test_report['1']['recall'],
                "test_f1-score": test_report['1']['f1-score'],
                "test_roc_auc": test_roc_auc,
                "best_params": grid_search.best_params_,
            }
            print(f"[{name}] test_roc_auc={test_roc_auc:.4f} test_f1={test_report['1']['f1-score']:.4f}")

            if test_roc_auc > overall_best["roc_auc"]:
                overall_best = {"name": name, "roc_auc": test_roc_auc, "estimator": best_model}

    # Log the champion algorithm on the parent run
    mlflow.log_param("best_algorithm", overall_best["name"])
    mlflow.log_metric("best_test_roc_auc", overall_best["roc_auc"])

    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability ("model registration")
    model_path = "tourism_project/deployment/best_model.joblib"   # local file path (inside tourism_project/deployment/) where the trained model is saved
    joblib.dump(overall_best["estimator"], model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"\nBest algorithm: {overall_best['name']} (test ROC-AUC = {overall_best['roc_auc']:.4f})")
    print(f"Model saved to {model_path}")

# Persist a small comparison table alongside the model for the notebook/README/deck
pd.DataFrame(overall_results).T.to_csv("tourism_project/deployment/model_comparison_summary.csv")
print("\nComparison summary written to tourism_project/deployment/model_comparison_summary.csv")
