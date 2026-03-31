# Student Burnout & Productivity Predictor

## Project Overview

The Student Burnout & Productivity Predictor is a beginner-friendly machine learning project that predicts a student's burnout level based on daily habits and academic pressure.

This project uses a synthetic dataset with student lifestyle factors such as study time, sleep, deadlines, social media usage, attendance, and exercise. A machine learning model is trained on this data and then used inside a simple Streamlit web app for interactive predictions.

## Problem Statement

Student burnout is often influenced by a mix of academic workload and lifestyle habits. Long study hours, low sleep, frequent deadlines, poor attendance, and unhealthy routines can increase stress and reduce productivity.

The goal of this project is to:

- analyze common student behavior patterns
- predict burnout level as `Low`, `Medium`, or `High`
- provide a simple interface for testing different student profiles
- demonstrate a complete machine learning workflow from data generation to deployment

## Features

- Synthetic burnout dataset generation using realistic rules
- Multi-class burnout prediction with `Low`, `Medium`, and `High` labels
- Training and evaluation of two models:
  Logistic Regression
  Decision Tree Classifier
- Accuracy and classification report comparison
- Best model saved using `joblib`
- Separate script for loading the trained model and making predictions
- Simple and clean Streamlit app for interactive use

## Tech Stack

- Python
- pandas
- scikit-learn
- joblib
- Streamlit

## Project Files

- `generate_burnout_data.py` - creates the synthetic dataset
- `burnout_data.csv` - generated dataset used for training
- `train_burnout_models.py` - trains, evaluates, and saves the best model
- `best_burnout_model.joblib` - saved trained model
- `load_burnout_model.py` - loads the saved model and predicts on sample input
- `streamlit_app.py` - Streamlit frontend for burnout prediction
- `run_streamlit_app.sh` - helper script to launch the Streamlit app

## Installation Steps

### 1. Clone or open the project folder

```bash
cd "student-burnout-productivity-predictor"
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required packages

```bash
pip install pandas scikit-learn joblib streamlit
```

If you want to install packages into a local project folder like this workspace uses, you can run:

```bash
python3 -m pip install pandas scikit-learn joblib streamlit -t .vendor
```

## How To Run the Project

### 1. Generate the dataset

```bash
python3 generate_burnout_data.py
```

This creates `burnout_data.csv` with 200 rows of student burnout data.

### 2. Train and evaluate the models

```bash
python3 train_burnout_models.py
```

This will:

- train Logistic Regression and Decision Tree models
- compare their performance
- print accuracy and classification reports
- save the best model as `best_burnout_model.joblib`

### 3. Load the saved model for prediction

```bash
python3 load_burnout_model.py
```

### 4. Run the Streamlit app

If you installed packages normally:

```bash
streamlit run streamlit_app.py
```

If you are using the local `.vendor` setup in this project:

```bash
./run_streamlit_app.sh
```

Then open the app in your browser, usually at:

```text
http://localhost:8501
```

## Example Usage

Example student input:

- Study hours per day: `8.5`
- Sleep hours: `5.0`
- Deadlines per week: `7`
- Social media hours: `5.5`
- Attendance percentage: `68`
- Exercise: `No`

Expected prediction:

```text
High
```

In the Streamlit app, enter the values, click `Predict Burnout Level`, and the result will be displayed clearly as:

- `Low`
- `Medium`
- `High`

## Model Summary

Both trained models reached strong performance on the synthetic dataset. In this project, Logistic Regression was selected as the best saved model because it matched the top accuracy while remaining simpler and slightly more consistent across classes.

## Learning Outcomes

This project is useful for beginners who want to learn:

- how to create a synthetic dataset
- how to train and compare multiple machine learning models
- how to save and load trained models
- how to build a simple machine learning web app with Streamlit

## Future Improvements

- Add a real-world dataset instead of synthetic data
- Add visualizations for burnout trends
- Deploy the Streamlit app online
- Add more student lifestyle and mental health features

## Author

This project was created as a machine learning and Streamlit practice project for student burnout prediction.
