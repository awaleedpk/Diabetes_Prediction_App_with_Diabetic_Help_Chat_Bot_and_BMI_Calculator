# Diabetes Prediction App

This GitHub repository hosts a Diabetes Prediction App developed with Python, leveraging machine learning techniques and the Streamlit library. The application enables users to predict the likelihood of diabetes based on various health metrics. The predictive model is built using the k-nearest neighbors algorithm, and the app provides an interactive interface for users to input their health information and receive instant predictions.

## Features

### 1. Machine Learning Model
- **K-Nearest Neighbors Algorithm:** The app utilizes the k-nearest neighbors algorithm to predict the likelihood of diabetes based on user-input health metrics.
- **Data Preprocessing:** The dataset is preprocessed by handling missing values and standardizing feature values to enhance the performance of the machine learning model.

### 2. Streamlit Web Application
- **User-Friendly Interface:** Streamlit is employed to create an intuitive and interactive web interface.
- **Input Fields:** Users can input various health metrics, including pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age.

### 3. Real-time Predictions
- **Instant Results:** Upon user input, the app provides immediate predictions on whether the user is likely to have diabetes or not.
- **Prediction Display:** The app displays a clear prediction result, categorizing the user as having "No Diabetes" or "Diabetes."

### 4. Sidebar Navigation
- **Sidebar Menu:** Users can navigate between different functionalities, including the Diabetes Prediction App, Diabetes Help Chatbot, and BMI Calculator, using a sidebar menu.

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies using the provided `requirements.txt` file.
3. Run the Streamlit app using the command `streamlit run app.py`.
4. Access the Diabetes Prediction App through your web browser at the specified address.
5. Input your health metrics in the provided fields and click "Predict" to receive instant predictions.


# Diabetes Help Chatbot with OpenAI Integration

This GitHub repository hosts an interactive Diabetes Help Chatbot built with Streamlit and powered by the OpenAI language model. The chatbot is designed to provide information and answers related to diabetic-related topics, including diet, blood sugar levels, and complications.

## Features

### 1. Chatbot Interface
- **User-Friendly Input:** Users can ask questions about diabetic-related topics using a simple text input field.
- **Dynamic Responses:** The chatbot dynamically generates responses based on user inquiries, creating an engaging and informative interaction.

### 2. OpenAI Integration
- **OpenAI Language Model:** Utilizes the OpenAI language model to process and generate responses to user queries.
- **Custom Prompt Template:** Implements a custom prompt template to structure the initial input prompt for the language model.

### 3. Memory Storage
- **Conversation History:** Utilizes a conversation buffer memory to store the history of the chat, enabling contextualized responses based on previous interactions.

### 4. Streamlit App
- **Web Application:** Developed with Streamlit, allowing for a seamless and interactive user experience.
- **Real-time Interaction:** Users receive immediate responses to their queries within the web application.

## Usage
1. Clone the repository to your local machine.
2. Set the OpenAI API key in the constants file (`openai_key`).
3. Install the required dependencies using the provided `requirements.txt` file.
4. Run the Streamlit app using the command `streamlit run app.py`.
5. Access the Diabetes Help Chatbot through your web browser at the specified address.
6. Enter your questions about diabetic-related topics in the provided text input field.

Empower yourself with knowledge about diabetes through interactive conversations with the Diabetes Help Chatbot!

# BMI Calculator using Streamlit

This GitHub repository hosts a simple yet effective BMI Calculator implemented in Python using the Streamlit library. Body Mass Index (BMI) is a widely used indicator for assessing an individual's body weight in relation to their height. This application allows users to input their weight and height, calculates their BMI, and provides a basic health category interpretation.

## Features

### 1. BMI Calculation
- **Simple Formula:** The BMI is calculated using the standard formula: BMI = weight (kg) / (height (m))^2.
- **User-Friendly Interface:** Intuitive input fields for weight and height make it easy for users to calculate their BMI.

### 2. Health Category Interpretation
- **Interpretation Function:** The application interprets the calculated BMI and assigns it to a basic health category (Underweight, Normal Weight, Overweight, or Obese).
- **Clear Results:** Users receive immediate feedback on their BMI and corresponding health category.

### 3. Streamlit App
- **Interactive Web Application:** Utilizes Streamlit, a Python library for creating web applications with minimal code.
- **Dynamic User Interface:** The app dynamically updates based on user input, providing a seamless user experience.

## Usage
1. Clone the repository to your local machine.
2. Install the required dependencies using the provided `requirements.txt` file.
3. Run the Streamlit app using the command `streamlit run app.py`.
4. Access the BMI Calculator through your web browser at the specified address.

## How to Calculate BMI
1. Enter your weight in kilograms.
2. Enter your height in centimeters.
3. Click the "Calculate BMI" button to get your BMI and its corresponding health category.

Take control of your health with the BMI Calculator!
