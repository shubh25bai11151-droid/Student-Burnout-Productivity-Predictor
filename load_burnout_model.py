import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).with_name(".vendor")
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import pandas as pd

from train_burnout_models import MODEL_PATH, load_saved_model, predict_burnout


def main() -> None:
    model_bundle = load_saved_model(MODEL_PATH)

    sample_student = pd.DataFrame(
        [
            {
                "study_hours_per_day": 8.5,
                "sleep_hours": 5.0,
                "deadlines_per_week": 7,
                "social_media_hours": 5.5,
                "attendance_percentage": 68,
                "exercise": 0,
            }
        ]
    )

    prediction = predict_burnout(sample_student, MODEL_PATH)[0]

    print(f"Loaded model: {model_bundle['model_name']}")
    print("Input for prediction:")
    print(sample_student.to_string(index=False))
    print()
    print(f"Predicted burnout level: {prediction}")


if __name__ == "__main__":
    main()
