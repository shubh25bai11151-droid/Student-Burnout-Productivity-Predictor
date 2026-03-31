import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).with_name(".vendor")
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).with_name("burnout_data.csv")
MODEL_PATH = Path(__file__).with_name("best_burnout_model.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURE_COLUMNS = [
    "study_hours_per_day",
    "sleep_hours",
    "deadlines_per_week",
    "social_media_hours",
    "attendance_percentage",
    "exercise",
]
TARGET_COLUMN = "burnout_level"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=4,
            random_state=RANDOM_STATE,
        ),
    }


def rank_results(results: dict[str, dict[str, object]]) -> list[tuple[str, dict[str, object]]]:
    return sorted(
        results.items(),
        key=lambda item: (
            item[1]["accuracy"],
            item[1]["macro_f1"],
            item[1]["macro_precision"],
        ),
        reverse=True,
    )


def train_and_evaluate_models(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results: dict[str, dict[str, object]] = {}
    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report_dict = classification_report(y_test, predictions, output_dict=True)

        results[model_name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": report_dict["macro avg"]["f1-score"],
            "macro_precision": report_dict["macro avg"]["precision"],
            "classification_report": classification_report(y_test, predictions, digits=3),
        }

    return results


def fit_best_model_on_full_dataset(df: pd.DataFrame, best_model_name: str) -> object:
    best_model = build_models()[best_model_name]
    best_model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    return best_model


def save_model_bundle(model: object, model_name: str, output_path: Path = MODEL_PATH) -> Path:
    model_bundle = {
        "model_name": model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "model": model,
    }
    joblib.dump(model_bundle, output_path)
    return output_path


def load_saved_model(output_path: Path = MODEL_PATH) -> dict[str, object]:
    if not output_path.exists():
        raise FileNotFoundError(f"Saved model not found at {output_path}")
    return joblib.load(output_path)


def predict_burnout(new_data: pd.DataFrame, output_path: Path = MODEL_PATH) -> list[str]:
    model_bundle = load_saved_model(output_path)
    predictions = model_bundle["model"].predict(new_data[model_bundle["feature_columns"]])
    return predictions.tolist()


def print_results(results: dict[str, dict[str, object]]) -> None:
    for model_name, metrics in results.items():
        print(model_name)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print("Classification Report:")
        print(metrics["classification_report"])
        print("-" * 60)

    print("Better Model Summary")
    ranked_models = rank_results(results)
    top_name, top_metrics = ranked_models[0]
    second_name, second_metrics = ranked_models[1]

    if top_metrics["accuracy"] == second_metrics["accuracy"]:
        print(
            f"Both models reached the same test accuracy of {top_metrics['accuracy']:.3f}."
        )
        print(
            f"If we break the tie using overall class balance metrics, {top_name} comes out slightly ahead."
        )

        if top_name == "Decision Tree Classifier":
            print(
                "Simple reason: the burnout labels were created from rule-like patterns, "
                "so the tree matches those if-then splits very naturally."
            )
        else:
            print(
                "Simple reason: logistic regression matched the same accuracy while staying simpler, "
                "and its class-wise precision was a little more consistent."
            )
        return

    print(
        f"{top_name} performed better with an accuracy of {top_metrics['accuracy']:.3f}, "
        f"compared with {second_name} at {second_metrics['accuracy']:.3f}."
    )

    if top_name == "Decision Tree Classifier":
        print(
            "Simple reason: the dataset uses rule-like patterns to assign burnout levels, "
            "and decision trees are very good at learning if-then style rules."
        )
    else:
        print(
            "Simple reason: the burnout classes are separated in a fairly smooth way, "
            "so logistic regression generalized better on the test data."
        )


def save_best_model(df: pd.DataFrame, results: dict[str, dict[str, object]]) -> Path:
    best_model_name, _ = rank_results(results)[0]
    best_model = fit_best_model_on_full_dataset(df, best_model_name)
    output_path = save_model_bundle(best_model, best_model_name)
    return output_path


if __name__ == "__main__":
    burnout_df = load_dataset()
    model_results = train_and_evaluate_models(burnout_df)
    print_results(model_results)
    saved_model_path = save_best_model(burnout_df, model_results)
    print()
    print(f"Saved best model to {saved_model_path}")
