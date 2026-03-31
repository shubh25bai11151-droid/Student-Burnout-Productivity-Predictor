import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_DIR / ".vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import pandas as pd
import streamlit as st

from train_burnout_models import FEATURE_COLUMNS, MODEL_PATH, load_saved_model


st.set_page_config(page_title="Student Burnout Prediction", layout="centered")


@st.cache_resource
def get_model_bundle() -> dict[str, object]:
    return load_saved_model(MODEL_PATH)


def build_input_frame(
    study_hours: float,
    sleep_hours: float,
    deadlines_per_week: int,
    social_media_hours: float,
    attendance_percentage: int,
    exercise: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_hours_per_day": study_hours,
                "sleep_hours": sleep_hours,
                "deadlines_per_week": deadlines_per_week,
                "social_media_hours": social_media_hours,
                "attendance_percentage": attendance_percentage,
                "exercise": exercise,
            }
        ]
    )


def show_prediction(prediction: str) -> None:
    if prediction == "Low":
        st.success("Predicted Burnout Level: Low")
    elif prediction == "Medium":
        st.warning("Predicted Burnout Level: Medium")
    else:
        st.error("Predicted Burnout Level: High")


def main() -> None:
    st.title("Student Burnout Prediction")
    st.write("Enter student lifestyle details to estimate the burnout level.")

    try:
        model_bundle = get_model_bundle()
    except FileNotFoundError:
        st.error("Saved model not found. Run `python3 train_burnout_models.py` first.")
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.slider("Study hours per day", 0.0, 12.0, 6.0, 0.5)
            sleep_hours = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
            deadlines_per_week = st.slider("Deadlines per week", 0, 10, 4)

        with col2:
            social_media_hours = st.slider("Social media usage (hours)", 0.0, 8.0, 2.0, 0.5)
            attendance_percentage = st.slider("Attendance percentage", 40, 100, 80)
            exercise_label = st.radio("Exercise", ["No", "Yes"], horizontal=True)

        predict_clicked = st.form_submit_button("Predict Burnout Level", use_container_width=True)

    if not predict_clicked:
        st.info("Fill in the values and click the button to see the prediction.")
        return

    student_data = build_input_frame(
        study_hours=study_hours,
        sleep_hours=sleep_hours,
        deadlines_per_week=deadlines_per_week,
        social_media_hours=social_media_hours,
        attendance_percentage=attendance_percentage,
        exercise=1 if exercise_label == "Yes" else 0,
    )

    prediction = model_bundle["model"].predict(student_data[FEATURE_COLUMNS])[0]

    st.subheader("Prediction Result")
    show_prediction(prediction)
    st.caption(f"Model loaded: {model_bundle['model_name']}")

    with st.expander("View input summary"):
        st.dataframe(student_data, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
