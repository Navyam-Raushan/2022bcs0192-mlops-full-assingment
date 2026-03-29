import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris   # ✅ THIS WAS MISSING
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import argparse

mlflow.set_tracking_uri("http://127.0.0.1:5000")

NAME = "Raushan Kumar"
ROLL = "2022bcs0192"

mlflow.set_experiment(f"{ROLL}_experiment")

def train(n_estimators=100, max_depth=3, use_subset=False):

    data = load_iris()

    X = data.data
    y = data.target

    # 🔥 Feature selection
    if use_subset:
        X = X[:, :2]  # use only first 2 features
        selected_features = "first_2_features"
    else:
        selected_features = "all_features"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    with mlflow.start_run():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')

        # ✅ Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features_used", selected_features)

        # ✅ Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # ✅ Save model
        import os
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        # ✅ Log model
        mlflow.sklearn.log_model(model, "model")

        # 🔥 REQUIRED for assignment (Name + Roll)
        print({
            "accuracy": acc,
            "f1_score": f1,
            "features_used": selected_features,
            "name": "Raushan Kumar",
            "roll": "2022bcs0192"
        })

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--use_subset", action="store_true")  # ✅ important

    args = parser.parse_args()

    train(args.n_estimators, args.max_depth, args.use_subset)