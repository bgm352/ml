import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Step 1: Load the dataset and explain
st.title("Interactive Machine Learning App: Iris Classifier")
st.write("""
This app demonstrates a machine learning workflow using the classic Iris dataset.
You can adjust the feature values below to see how the model predicts the species.
""")

# Step 2: Data Exploration
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
st.write("### Sample of the dataset", df.head())

# Step 3: User input widgets
st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Step 4: Model Training
st.write("### Training a RandomForestClassifier")
clf = RandomForestClassifier()
clf.fit(df, iris.target)
st.write("Model trained on the full Iris dataset.")

# Step 5: Prediction
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

st.write("### User Input Features")
st.write(input_df)

st.write("### Prediction")
st.write(f"Predicted species: **{iris.target_names[prediction][0]}**")
st.write("Prediction probabilities:")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))

# Step 6: Explanation
st.write("""
**How this works:**
- The app loads the Iris dataset and trains a RandomForestClassifier.
- You can adjust the feature values using the sliders.
- The model predicts the species and shows the probabilities for each class.
- This demonstrates the end-to-end process: data input, model training, prediction, and explanation.
""")
