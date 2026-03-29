import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import argparse

NAME = "Raushan Kumar"
ROLL = "2022bcs0192"

mlflow.set_experiment(f"{ROLL}_experiment")

def train(n_estimators=100, max_depth=3):

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    with mlflow.start_run():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        mlflow.sklearn.log_model(model, "model")

        print({
            "accuracy": acc,
            "f1": f1,
            "name": NAME,
            "roll": ROLL
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)

    args = parser.parse_args()

    train(args.n_estimators, args.max_depth)