import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).with_name(".vendor")
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import pandas as pd


def _uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    return np.round(rng.uniform(low, high, size), 1)


def _integers(rng: np.random.Generator, low: int, high: int, size: int) -> np.ndarray:
    return rng.integers(low, high + 1, size=size)


def _sample_profile(profile: str, count: int, rng: np.random.Generator) -> pd.DataFrame:
    if profile == "balanced":
        return pd.DataFrame(
            {
                "study_hours_per_day": _uniform(rng, 3.0, 7.0, count),
                "sleep_hours": _uniform(rng, 7.0, 9.5, count),
                "deadlines_per_week": _integers(rng, 0, 4, count),
                "social_media_hours": _uniform(rng, 0.0, 3.0, count),
                "attendance_percentage": _integers(rng, 80, 100, count),
                "exercise": rng.choice([0, 1], size=count, p=[0.2, 0.8]),
            }
        )

    if profile == "steady_pressure":
        return pd.DataFrame(
            {
                "study_hours_per_day": _uniform(rng, 5.0, 9.0, count),
                "sleep_hours": _uniform(rng, 5.5, 7.0, count),
                "deadlines_per_week": _integers(rng, 3, 7, count),
                "social_media_hours": _uniform(rng, 1.0, 5.0, count),
                "attendance_percentage": _integers(rng, 70, 95, count),
                "exercise": rng.choice([0, 1], size=count, p=[0.45, 0.55]),
            }
        )

    if profile == "cramming":
        return pd.DataFrame(
            {
                "study_hours_per_day": _uniform(rng, 7.5, 12.0, count),
                "sleep_hours": _uniform(rng, 3.0, 6.0, count),
                "deadlines_per_week": _integers(rng, 6, 10, count),
                "social_media_hours": _uniform(rng, 2.0, 7.0, count),
                "attendance_percentage": _integers(rng, 50, 90, count),
                "exercise": rng.choice([0, 1], size=count, p=[0.75, 0.25]),
            }
        )

    if profile == "distracted":
        return pd.DataFrame(
            {
                "study_hours_per_day": _uniform(rng, 1.0, 6.0, count),
                "sleep_hours": _uniform(rng, 4.5, 8.0, count),
                "deadlines_per_week": _integers(rng, 2, 8, count),
                "social_media_hours": _uniform(rng, 4.0, 8.0, count),
                "attendance_percentage": _integers(rng, 40, 85, count),
                "exercise": rng.choice([0, 1], size=count, p=[0.7, 0.3]),
            }
        )

    raise ValueError(f"Unknown profile: {profile}")


def _assign_burnout_level(df: pd.DataFrame) -> pd.Categorical:
    score = np.zeros(len(df), dtype=float)

    score += np.select(
        [df["sleep_hours"] < 5, df["sleep_hours"] < 6, df["sleep_hours"] < 7],
        [3, 2, 1],
        default=0,
    )
    score += np.select(
        [
            df["deadlines_per_week"] >= 8,
            df["deadlines_per_week"] >= 5,
            df["deadlines_per_week"] >= 3,
        ],
        [3, 2, 1],
        default=0,
    )
    score += np.select(
        [df["social_media_hours"] >= 6, df["social_media_hours"] >= 4],
        [2, 1],
        default=0,
    )
    score += np.select(
        [df["study_hours_per_day"] >= 10, df["study_hours_per_day"] >= 8],
        [2, 1],
        default=0,
    )
    score += np.select(
        [df["attendance_percentage"] < 60, df["attendance_percentage"] < 75],
        [2, 1],
        default=0,
    )
    score += np.where(df["exercise"] == 0, 1, -1)

    overload_pattern = (
        (df["sleep_hours"] < 5.5)
        & (df["deadlines_per_week"] >= 7)
        & (df["social_media_hours"] >= 5)
    )
    balanced_pattern = (
        df["sleep_hours"].between(7.0, 9.5)
        & df["study_hours_per_day"].between(3.5, 7.5)
        & (df["deadlines_per_week"] <= 4)
        & (df["social_media_hours"] <= 3)
        & (df["attendance_percentage"] >= 80)
        & (df["exercise"] == 1)
    )

    score += np.where(overload_pattern, 3, 0)
    score -= np.where(balanced_pattern, 3, 0)

    burnout_level = np.select(
        [score <= 2, score <= 6],
        ["Low", "Medium"],
        default="High",
    )
    return pd.Categorical(burnout_level, categories=["Low", "Medium", "High"], ordered=True)


def build_burnout_dataset(num_rows: int = 200, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    profile_names = ["balanced", "steady_pressure", "cramming", "distracted"]
    profile_weights = [0.30, 0.28, 0.22, 0.20]
    profile_counts = rng.multinomial(num_rows, profile_weights)

    frames = [
        _sample_profile(profile_name, count, rng)
        for profile_name, count in zip(profile_names, profile_counts)
        if count > 0
    ]

    burnout_df = pd.concat(frames, ignore_index=True)
    burnout_df["burnout_level"] = _assign_burnout_level(burnout_df)
    burnout_df = burnout_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return burnout_df


burnout_df = build_burnout_dataset()


if __name__ == "__main__":
    output_path = Path(__file__).with_name("burnout_data.csv")
    burnout_df.to_csv(output_path, index=False)

    print(f"Saved {len(burnout_df)} rows to {output_path}")
    print()
    print("Burnout level distribution:")
    print(burnout_df["burnout_level"].value_counts().sort_index())
    print()
    print(burnout_df.head().to_string(index=False))
