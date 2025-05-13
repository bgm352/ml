import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Iris ML Explorer", page_icon="🌸", layout="wide")

# Title and Introduction
st.title("🔬 Interactive Machine Learning App: Iris Classifier")
st.write("""
Welcome! This app walks you through a complete machine learning workflow using the classic Iris dataset.
You'll see how data is loaded, how features are selected, how a model is trained, and how predictions are made.
Each step is explained in detail below.
""")

# Step 1: Data Loading and Explanation
st.header("Step 1: Load the Dataset")
st.markdown("""
We use the **Iris dataset**, a famous dataset in machine learning. It contains 150 samples of iris flowers,
each described by four features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Each sample is labeled as one of three species: **setosa, versicolor, or virginica**.
""")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
st.dataframe(df.head(), use_container_width=True)
st.markdown("*Above: The first five rows of the dataset. Each row is a flower sample with its features.*")

# Step 2: Data Exploration
st.header("Step 2: Explore the Data")
st.markdown("Let's look at some basic statistics to understand the data better.")
st.write(df.describe())

# Visualization: Pairplot
with st.expander("🔍 Click to view feature relationships (pairplot)"):
    st.markdown("**Pairplot:** Explore how features relate to each other and the species.")
    fig = sns.pairplot(df, hue="species")
    st.pyplot(fig)

st.markdown("""
- **Mean, min, and max** values help us understand the range of each feature.
- This information is useful for setting up the input sliders in the next step.
""")

# Step 3: User Input Widgets
st.header("Step 3: Input Flower Features")
st.markdown("""
Use the sliders in the sidebar to set the features of a hypothetical iris flower.
The model will use these values to predict the species.
""")
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider(
        'Sepal length (cm)',
        float(df['sepal length (cm)'].min()),
        float(df['sepal length (cm)'].max()),
        float(df['sepal length (cm)'].mean())
    )
    sepal_width = st.sidebar.slider(
        'Sepal width (cm)',
        float(df['sepal width (cm)'].min()),
        float(df['sepal width (cm)'].max()),
        float(df['sepal width (cm)'].mean())
    )
    petal_length = st.sidebar.slider(
        'Petal length (cm)',
        float(df['petal length (cm)'].min()),
        float(df['petal length (cm)'].max()),
        float(df['petal length (cm)'].mean())
    )
    petal_width = st.sidebar.slider(
        'Petal width (cm)',
        float(df['petal width (cm)'].min()),
        float(df['petal width (cm)'].max()),
        float(df['petal width (cm)'].mean())
    )
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.write("#### Your Selected Features")
st.write(input_df)
st.markdown("*Above: The feature values you selected using the sliders.*")

# Step 4: Model Training
st.header("Step 4: Train the Machine Learning Model")
st.markdown("""
We use a **Random Forest Classifier**, a popular machine learning algorithm that combines the results of multiple decision trees for better accuracy and robustness.

- The model is trained on the entire Iris dataset.
- It learns the relationship between the flower features and their species.
""")
clf = RandomForestClassifier(random_state=42)
clf.fit(df[iris.feature_names], iris.target)
st.success("The Random Forest model has been trained on the dataset.")

# Model Interpretation: Feature Importances
with st.expander("🔎 View feature importances"):
    st.markdown("**Feature Importances:** The model assigns importance scores to each feature based on how useful they are for making predictions.")
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': iris.feature_names,
        'Importance': importances
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(imp_df.set_index('Feature'))

# Step 5: Make Predictions
st.header("Step 5: Make a Prediction")
st.markdown("""
Now, the trained model predicts the species of your hypothetical flower based on your input features.
""")
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

st.write(f"**Predicted species:** {iris.target_names[prediction][0].capitalize()}")
st.write("**Prediction probabilities:**")
proba_df = pd.DataFrame(
    prediction_proba,
    columns=[name.capitalize() for name in iris.target_names]
)
st.write(proba_df)

# Step 6: Explain the Results
st.header("Step 6: Understanding the Results")
st.markdown("""
- The **predicted species** is the one with the highest probability.
- The **probabilities** show how confident the model is about each possible species.
- If the probabilities are close, it means the model is less certain.
- The model uses the patterns it learned during training to make these predictions.

**Why Random Forest?**
- Random Forest is robust to overfitting and works well with small datasets like Iris.
- It can handle complex relationships between features and the target variable.

**Next Steps:**
- Try changing the sliders to see how different feature values affect the prediction.
- This is similar to how data scientists test models with new data!
""")

st.info("""
This app demonstrates the full machine learning workflow:
1. **Data Loading**
2. **Exploration**
3. **Feature Input**
4. **Model Training**
5. **Prediction**
6. **Interpretation**

You can use this template for your own datasets and models!
""")

# --- New Section: People, Resources, and Example Projects ---

st.header("Learn More & Get Inspired")
st.markdown("""
### 👤 **People to Follow**
- **Max Wiertz (@maxwiertz):** Shares rapid Streamlit prototypes and tips.
- **Robert John | MLOps (@trojrobert):** Focuses on ML operations and Streamlit integration.
- **Shubham Chaudhari (@Shubham28698):** Posts about deploying Streamlit apps for data science.
- **Vinay Babu (@min2bro):** Advocates for Streamlit in ML demos.
- **Ayoub Nainia (@nainia_ayoub):** Insights on productionizing ML models with Streamlit.

### 📚 **Resources**
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit App Gallery](https://streamlit.io/gallery)
- [Udemy Streamlit Courses](https://www.udemy.com/topic/streamlit/)
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [GitHub: Streamlit Projects](https://github.com/topics/streamlit)

### 💡 **Example Projects**
- **PCA Dashboard by Ben Jack:** Interactive PCA dashboard with Streamlit & Plotly ([GitHub](https://github.com/benjaminjack/streamlit-pca))
- **ML Model Demos:** Many apps let users input data and see live predictions.
- **Streamlit App Gallery:** Browse dashboards, ML demos, and more for inspiration.

### 🛠️ **Other Tools to Explore**
- **Dash by Plotly:** For advanced dashboards and visualizations.
- **Gradio:** For quick, shareable ML model demos.
""")

# --- New Section: Code Transparency ---
st.header("See the Code Used")
with st.expander("Show code for model training and prediction"):
    st.code("""
# Model Training
clf = RandomForestClassifier(random_state=42)
clf.fit(df[iris.feature_names], iris.target)

# Prediction
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)
    """, language="python")

# --- Tips & Best Practices ---
st.header("Tips for Your Own Data Science Projects")
st.markdown("""
- **Start simple:** Use simple models and visualizations first, then add complexity.
- **Visualize your data:** Always explore your data visually before modeling.
- **Interpret results:** Use feature importances and probability outputs to understand model behavior.
- **Document your workflow:** Clear explanations help others (and your future self) understand your process.
- **Share and collaborate:** Use platforms like GitHub and Streamlit Cloud to share your apps and get feedback.
""")

# --- Contact & Contribution ---
st.header("Contribute or Get in Touch")
st.markdown("""
- **Found this app useful?** Fork it on GitHub and add your own features!
- **Questions or feedback?** [Open an issue](https://github.com/) or connect with the Streamlit community.
- **Happy learning and building! 🚀**
""")


