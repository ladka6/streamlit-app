import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# Define a Streamlit app title
st.title("Tesla Stock Analysis")

# Load the data


@st.cache_data  # Caching the data to improve performance
def load_data():
    df = pd.read_csv('./content/Tesla.csv')
    return df


df = load_data()

# Display the first few rows of the dataset
st.subheader("Tesla Stock Data")
st.write(df.head())

# Data Preprocessing
st.subheader("Data Preprocessing")

# Remove 'Adj Close' column
df = df.drop(['Adj Close'], axis=1)

# Check for missing values
missing_values = df.isnull().sum()
st.write("Missing Values:")
st.write(missing_values)

# Data Visualization
st.subheader("Data Visualization")

features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20, 10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.distplot(df[col])

st.pyplot(plt)

# Feature Engineering
st.subheader("Feature Engineering")

splitted = df['Date'].str.split('-', expand=True)

df['day'] = splitted[2].astype('int')
df['month'] = splitted[1].astype('int')
df['year'] = splitted[0].astype('int')

df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Display the modified DataFrame
st.write(df.head())

# Model Training
st.subheader("Model Training")

# Prepare features and target
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and validation sets
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Display model performance metrics before adding new data
st.write("Initial Model Performance:")
st.write(f'Model: {model}')
st.write('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, model.predict_proba(X_train)[:, 1]))
st.write('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, model.predict_proba(X_valid)[:, 1]))

# Form to add new data
st.subheader("Add New Data to Training Set")
with st.form("new_data_form"):
    open_val = st.number_input("Open Value")
    close_val = st.number_input("Close Value")
    low_high_val = st.number_input("Low-High Value")
    is_quarter_end_val = st.radio("Is Quarter End?", ["Yes", "No"])

    submitted = st.form_submit_button("Add Data")

if submitted:
    # Convert radio button value to 1 for "Yes" and 0 for "No"
    is_quarter_end_val = 1 if is_quarter_end_val == "Yes" else 0

    # Add the new data to the training set
    new_data = [[open_val - close_val, low_high_val, is_quarter_end_val]]
    X_train = np.concatenate((X_train, new_data), axis=0)
    # Assume the new data always has a positive target (1)
    Y_train = np.append(Y_train, 1)

    # Retrain the model with the updated training data
    model.fit(X_train, Y_train)

    # Display model performance metrics after adding new data
    st.write("Updated Model Performance:")
    st.write(f'Model: {model}')
    st.write('Training Accuracy : ', metrics.roc_auc_score(
        Y_train, model.predict_proba(X_train)[:, 1]))
    st.write('Validation Accuracy : ', metrics.roc_auc_score(
        Y_valid, model.predict_proba(X_valid)[:, 1]))

    st.success(
        "New data has been added to the training set and the model has been retrained.")
