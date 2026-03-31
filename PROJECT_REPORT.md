# Student Burnout & Productivity Predictor

## 1. Introduction

Student life often involves balancing academics, deadlines, social activities, health, and personal habits. When this balance is disturbed for a long period, students may experience burnout. Burnout can affect concentration, motivation, attendance, and overall productivity.

The Student Burnout & Productivity Predictor project was developed to study how daily routines and academic pressure can influence burnout level. The project uses machine learning to predict whether a student's burnout level is `Low`, `Medium`, or `High` based on a small set of lifestyle and academic factors.

This project is designed as a beginner-friendly end-to-end machine learning system. It includes dataset generation, model training, model comparison, model saving, and a Streamlit web application for interactive prediction.

## 2. Problem Statement

Many students face stress due to long study hours, poor sleep, too many deadlines, lack of exercise, and excessive social media use. These factors can reduce productivity and affect well-being. However, burnout is not always easy to identify early.

The problem addressed in this project is to build a simple prediction system that can estimate burnout level using common student-related inputs. Such a system can help demonstrate how data science can be used to understand academic stress patterns and support healthier routines.

## 3. Objectives

The main objectives of this project are:

- to create a meaningful dataset for student burnout prediction
- to define logical relationships between lifestyle habits and burnout level
- to train machine learning models on the dataset
- to compare model performance using accuracy and classification reports
- to save the best model for future use
- to build a simple Streamlit interface for easy user interaction
- to understand the full workflow of a machine learning project from start to deployment

## 4. Dataset Description

The dataset used in this project is a synthetic dataset created specifically for burnout prediction. It contains 200 rows, where each row represents one student's academic and lifestyle profile.

### Dataset Features

The dataset includes the following input columns:

| Column Name | Description | Range |
| --- | --- | --- |
| `study_hours_per_day` | Average study time per day | 0 to 12 |
| `sleep_hours` | Average sleep per day | 3 to 10 |
| `deadlines_per_week` | Number of deadlines in a week | 0 to 10 |
| `social_media_hours` | Daily social media usage | 0 to 8 |
| `attendance_percentage` | Attendance level | 40 to 100 |
| `exercise` | Whether the student exercises | 0 or 1 |

The target column is:

| Column Name | Description |
| --- | --- |
| `burnout_level` | Burnout class: `Low`, `Medium`, or `High` |

### Logic Used to Create the Target

The dataset was not generated randomly without meaning. A rule-based scoring approach was used so that the burnout label follows realistic patterns. For example:

- low sleep, many deadlines, and high social media usage increase burnout
- balanced sleep, moderate study time, good attendance, and exercise reduce burnout
- overloaded schedules push students toward `High` burnout
- healthier routines push students toward `Low` burnout

This made the dataset more useful for model training and closer to a real educational scenario.

## 5. Methodology

### 5.1 Data Preprocessing

The preprocessing stage was simple because the dataset was synthetic and clean by design. The following steps were followed:

- the generated dataset was loaded from `burnout_data.csv`
- input features and target labels were separated
- the dataset was split into training and testing sets
- an 80:20 train-test split was used
- stratified splitting was applied so that all three burnout classes remained balanced in both sets

No missing value handling was required because the generated data did not contain null values. No complex feature engineering was needed because the features were already meaningful and numeric.

### 5.2 Models Used

Two machine learning models were trained and compared:

#### Logistic Regression

Logistic Regression is a simple and widely used classification algorithm. In this project, it was used for multi-class classification to predict `Low`, `Medium`, or `High` burnout.

Before training Logistic Regression, feature scaling was applied using `StandardScaler`. This helps the model perform better when input values are on different ranges, such as study hours and attendance percentage.

#### Decision Tree Classifier

Decision Tree Classifier is a rule-based model that learns decision paths from the training data. It is useful for problems where class predictions depend on combinations of conditions, such as low sleep and high deadlines.

The decision tree in this project was trained with controlled settings such as limited depth and minimum split size to reduce the risk of overfitting.

### 5.3 Model Evaluation

Both models were tested on unseen test data. Their performance was evaluated using:

- accuracy
- precision
- recall
- F1-score
- classification report

After comparison, the better model was selected and saved using `joblib`.

## 6. Results

After training and testing, the following accuracy scores were obtained:

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 0.950 |
| Decision Tree Classifier | 0.950 |

Both models achieved the same overall accuracy of 95% on the test dataset. However, Logistic Regression was selected as the best saved model because it showed slightly more consistent class-wise precision and remained simpler than the Decision Tree model.

### Result Interpretation

- both models performed strongly on the synthetic dataset
- Logistic Regression gave very stable overall performance
- Decision Tree also worked well because the dataset followed rule-like burnout patterns
- since the accuracy was tied, the simpler model was preferred

This result suggests that the burnout classes in the dataset were separated clearly enough for both linear and rule-based models to learn the patterns effectively.

## 7. Challenges Faced

Several practical challenges were faced during the project:

- creating a synthetic dataset that looked logical instead of completely random
- making sure the target labels reflected meaningful burnout behavior
- balancing the dataset so that `Low`, `Medium`, and `High` classes were all represented
- comparing two models fairly using the same train-test split
- handling local dependency setup for packages such as `pandas`, `scikit-learn`, and `Streamlit`
- ensuring that the saved model could be loaded again correctly for later predictions
- launching and verifying the Streamlit app in a restricted local environment

These challenges were helpful because they reflected real problems that often happen in machine learning projects, even in small academic applications.

## 8. Learnings

This project provided several important learnings:

- machine learning works better when the dataset reflects realistic logic
- model accuracy alone is not always enough to choose the best model
- simpler models can be strong choices when performance is similar
- preprocessing steps such as scaling can improve some algorithms
- saving trained models is essential for deployment
- building a user interface helps turn a model into a usable application
- Streamlit is a simple and effective way to present machine learning predictions

The project also improved understanding of the complete pipeline, including data generation, preprocessing, model evaluation, persistence, and deployment.

## 9. Future Scope

This project can be extended in several ways:

- use real student survey data instead of synthetic data
- include more features such as screen time, mental health scores, exam frequency, or diet
- add visual dashboards to show burnout trends and feature impact
- test more machine learning models such as Random Forest, SVM, or XGBoost
- improve the app with charts, recommendations, and personalized suggestions
- deploy the Streamlit application on a cloud platform for public access
- connect the prediction system with counseling or wellness support tools

In the future, this project can move from being a learning prototype to a more practical student well-being assistant.

## Conclusion

The Student Burnout & Productivity Predictor successfully demonstrates how machine learning can be used to estimate burnout level from simple student lifestyle inputs. The project combined logical dataset creation, model training, model comparison, model saving, and a clean Streamlit interface into one complete workflow.

Overall, the project shows that even a beginner-friendly machine learning system can provide useful insights when it is structured carefully and built around a clear problem statement.
