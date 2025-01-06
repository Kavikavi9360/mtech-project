import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense

# Load the dataset
data = pd.read_csv(r'C:\Users\ADMIN\Documents\final app\70 accu.csv')

# Preview the dataset
st.write("Dataset Preview:")
st.write(data.head())

# Step 2: Define features and target
y = data['Self-Reported Creativity Score']
X = data.drop(columns=['Self-Reported Creativity Score'])

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].astype(str)

onehotencoder = OneHotEncoder(sparse_output=False)
X_encoded = onehotencoder.fit_transform(X[categorical_cols])

X_numeric = X.select_dtypes(exclude=['object'])
X_combined = np.concatenate((X_numeric, X_encoded), axis=1)

# Encode target variable if it’s categorical
y_encoded = y - 1

# Step 3: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)

# Step 4: Train SVM and RandomForest models
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Get prediction probabilities from both models
svm_train_pred_proba = svm_model.predict_proba(X_train)
svm_test_pred_proba = svm_model.predict_proba(X_test)
rf_train_pred_proba = rf_model.predict_proba(X_train)
rf_test_pred_proba = rf_model.predict_proba(X_test)

# Combine probabilities for RCNN input
train_pred_proba = np.hstack((svm_train_pred_proba, rf_train_pred_proba))
test_pred_proba = np.hstack((svm_test_pred_proba, rf_test_pred_proba))

# Step 6: Build and train RCNN model
X_train_rcnn = train_pred_proba.reshape(train_pred_proba.shape[0], train_pred_proba.shape[1], 1)
X_test_rcnn = test_pred_proba.reshape(test_pred_proba.shape[0], test_pred_proba.shape[1], 1)

rcnn_model = Sequential()
rcnn_model.add(Conv1D(32, 2, activation='relu', input_shape=(X_train_rcnn.shape[1], 1)))
rcnn_model.add(MaxPooling1D(pool_size=2))
rcnn_model.add(LSTM(64, return_sequences=False))
rcnn_model.add(Dense(64, activation='relu'))
rcnn_model.add(Dense(5, activation='softmax'))

rcnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model with checks
st.write("Training RCNN model...")
history = rcnn_model.fit(X_train_rcnn, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 7: Evaluate RCNN model
test_loss, test_accuracy = rcnn_model.evaluate(X_test_rcnn, y_test)

# Make predictions on test set with checks
st.write("Making predictions on test data...")
y_pred_test = rcnn_model.predict(X_test_rcnn)

if y_pred_test is None:
    st.write("Prediction failed: Model returned None.")
else:
    st.write(f"Prediction shape: {y_pred_test.shape}")
    if y_pred_test.shape[0] == 0:
        st.write("No predictions were made. Please check the input data.")
    else:
        y_pred_test = np.argmax(y_pred_test, axis=1)

# Calculate precision, recall, F1 score
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

# Display precision, recall, and F1 score in Streamlit
st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# Function to predict creativity score
def predict_creativity_score(sample_data):
    sample_df = pd.DataFrame([sample_data])
    sample_encoded = onehotencoder.transform(sample_df[categorical_cols])
    sample_combined = np.concatenate((sample_df.select_dtypes(exclude=['object']).values, sample_encoded), axis=1)
    rf_sample_pred_proba = rf_model.predict_proba(sample_combined)
    svm_sample_pred_proba = svm_model.predict_proba(sample_combined)
    sample_pred_combined = np.hstack((svm_sample_pred_proba, rf_sample_pred_proba))
    sample_rcnn_input = sample_pred_combined.reshape(sample_pred_combined.shape[0], sample_pred_combined.shape[1], 1)
    sample_prediction = rcnn_model.predict(sample_rcnn_input)
    predicted_score = np.argmax(sample_prediction) + 1  # Assuming 1-5 creativity score scale
    return predicted_score

# Function to get user input and show prediction
def get_user_input_and_predict():
    age = st.selectbox("Enter Age (e.g., '18-20','21-22','23-24')", ['18-20', '21-22', '23-24'])
    daily_screen_time = st.selectbox("Enter Daily Screen Time (e.g., 'Less than 2 hours', '2-4 hours', 'More than 4 hours')", ['Less than 2 hours', '2-4 hours', 'More than 4 hours'])
    use_of_ai_tools = st.selectbox("Do you use AI Tools for Creativity? (Yes or No)", ['Yes', 'No'])
    ai_tools_used = st.text_input("List AI Tools Used for Creativity (comma-separated)")
    reliance_on_technology = st.selectbox("How often do you rely on technology for ideas? (Always, Often, Sometimes, Rarely, Never)", ['Always', 'Often', 'Sometimes', 'Rarely', 'Never'])
    technology_impact = st.selectbox("What is the impact of technology on your creativity? (Positive , Negative , No impact)", ['Positive', 'Negative', 'No impact'])
    problem_solving_preference = st.selectbox("Preference for Independent Problem-Solving? (Independently or With technology)", ['Independently', 'With technology'])

    # Create a dictionary for the user input
    sample_input = {
        'Age ': age,
        'Daily Screen Time': daily_screen_time,
        'Use of AI Tools for Creativity': use_of_ai_tools,
        'AI Tools Used': ai_tools_used,
        'Reliance on Technology for Ideas': reliance_on_technology,
        "Technology's Impact on Creativity": technology_impact,
        'Preference for Independent Problem-Solving': problem_solving_preference
    }

    if st.button('Predict Creativity Score'):
        predicted_score = predict_creativity_score(sample_input)
        score_labels = {
            1: "Very Low - Your creativity score is low. Consider engaging in more creative activities without technology reliance.",
            2: "Low - Your creativity score is below average. Try exploring new hobbies or activities to boost your creativity.",
            3: "Moderate - Your creativity score is average. Keep balancing your technology usage and creative tasks.",
            4: "High - Your creativity score is good! You're effectively using technology to enhance your creativity.",
            5: "Very High - Excellent creativity score! You have a strong creative mindset and effectively utilize technology as a tool."
        }
        st.write(f"Predicted Creativity Score: {predicted_score} - {score_labels[predicted_score]}")

# Display AI Tools graph and pie chart
def display_ai_tools_graph():
    # Handle missing or non-string values in the 'AI Tools Used' column
    data['AI Tools Used'] = data['AI Tools Used'].fillna('')
    tools_series = data['AI Tools Used'].apply(lambda x: x.split(',') if isinstance(x, str) else []).explode()
    tools_series = tools_series.str.strip()
    tool_counts = tools_series.value_counts()

    top_5_tools = tool_counts.head(5)
    colors = ['#FF5733' if tool == 'ChatGPT' else '#3498db' for tool in top_5_tools.index]

    # Bar chart
    plt.figure(figsize=(10, 6))
    top_5_tools.plot(kind='bar', color=colors)
    plt.title('Top 5 Most Used AI Tools for Creativity Among Young Adults (16-24)')
    plt.xlabel('AI Tools')
    plt.ylabel('Frequency of Use')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

    # Pie chart
    ai_tool_usage_counts = data['Use of AI Tools for Creativity'].value_counts()
    labels = ['Yes', 'No']
    colors = ['#DE3163', '#DFFF00']  # Red for 'No', Green for 'Yes'

    # Create pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(ai_tool_usage_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Usage of AI Tools for Creativity (Yes/No)')
    plt.show()
    st.pyplot(plt)

# Main Streamlit Interface
st.title('Creativity Score Prediction and AI Tool Analysis')

get_user_input_and_predict()

# Show AI tools usage graph and pie chart on button click
if st.button('Show Top 5 Most Used AI Tools and Pie Chart'):
    display_ai_tools_graph()
