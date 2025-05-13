# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Science and Machine Learning Concepts Explained ---

"""
# 🌸 Iris Data Science & Machine Learning Explorer

This app demonstrates key concepts in **data science** and **machine learning** using Python and Streamlit.

---

## What is Data Science?

Data science is the process of extracting insights and knowledge from data. It involves:
- Collecting data
- Cleaning and preparing data
- Analyzing and visualizing data
- Building predictive models (machine learning)
- Communicating results

## What is Machine Learning?

Machine learning is a subset of AI focused on building algorithms that can learn from data and make predictions or decisions without being explicitly programmed for each task[2][3][5][6].

**Types of machine learning:**
- **Supervised Learning:** Learn from labeled data (e.g., classification, regression)
- **Unsupervised Learning:** Find patterns in unlabeled data (e.g., clustering)
- **Reinforcement Learning:** Learn by trial and error

In this app, we'll use **supervised learning** for classification.
"""

# --- LOAD DATA ---

@st.cache_data
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(
        iris.data, columns=iris.feature_names
    )
    df['target'] = iris.target
    df['target_name'] = df['target'].map(dict(enumerate(iris.target_names)))
    return df, iris

df, iris = load_data()

# --- DATA EXPLORATION ---

st.header("1. Data Exploration")

st.write("The Iris dataset contains measurements for 150 iris flowers from 3 species.")
st.dataframe(df.head())

# Data summary
st.subheader("Summary Statistics")
st.write(df.describe())

# --- DATA VISUALIZATION ---

st.header("2. Data Visualization")

st.write("Visualizing data helps us understand patterns and relationships.")

# Pairplot
st.subheader("Pairplot (Scatterplot Matrix)")
fig = sns.pairplot(df, hue="target_name")
st.pyplot(fig)

# Correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig2, ax2 = plt.subplots()
sns.heatmap(df.iloc[:, :-2].corr(), annot=True, cmap="Blues", ax=ax2)
st.pyplot(fig2)

# --- MACHINE LEARNING PIPELINE ---

st.header("3. Machine Learning Model")

st.write("""
Let's train a **Random Forest Classifier** to predict the iris species from the measurements.

### Steps:
1. **Split the data** into training and testing sets.
2. **Train** the model on the training data.
3. **Test** the model on unseen data.
4. **Evaluate** the model's accuracy.
""")

# Feature selection
features = st.multiselect(
    "Choose features for model training:",
    iris.feature_names,
    default=iris.feature_names
)
if len(features) < 2:
    st.warning("Select at least 2 features.")
else:
    X = df[features]
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Prediction
    y_pred = clf.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    st.subheader(f"Model Accuracy: {acc:.2f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
                xticklabels=iris.target_names, yticklabels=iris.target_names, cmap="Greens", ax=ax3)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig3)

# --- USER PREDICTION ---

st.header("4. Try It Yourself!")

st.write("Input measurements and the model will predict the species:")

user_input = []
for feature in iris.feature_names:
    val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    user_input.append(val)

if st.button("Predict Species"):
    user_pred = clf.predict([user_input])[0]
    st.success(f"Predicted species: **{iris.target_names[user_pred].capitalize()}**")

# --- FURTHER LEARNING ---

st.header("5. Learn More")

st.markdown("""
- [W3Schools: Python Machine Learning][2]
- [Real Python: Data Science Tutorials][6]
- [DataCamp: Machine Learning with scikit-learn][4]
- [Harvard: Data Science with Python][5]
""")

# --- END OF APP ---

"""
---

## Key Concepts Recap

- **Data Science**: Extracting knowledge from data using programming, statistics, and domain expertise.
- **Machine Learning**: Training algorithms to make predictions from data.
- **Python Libraries**: pandas (data handling), matplotlib/seaborn (visualization), scikit-learn (machine learning).
- **Model Evaluation**: Accuracy, confusion matrix, and classification report.

Explore, visualize, and build models with your data!
"""

