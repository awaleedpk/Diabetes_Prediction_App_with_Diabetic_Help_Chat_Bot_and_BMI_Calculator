# Import necessary libraries
import streamlit as st  # Importing the Streamlit library for creating web applications
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import os
from constants import openai_key  # Importing the OpenAI API key from a constants file
from langchain.llms import OpenAI  # Importing the OpenAI language model
from langchain import PromptTemplate  # Importing a template for creating prompts
from langchain.chains import LLMChain  # Importing a chain to link language models
from langchain.memory import ConversationBufferMemory  # Importing memory for storing conversation history

# Import necessary libraries (duplicate import statement removed for clarity)
# Comment: The code imports essential libraries for data manipulation, machine learning, and interfacing with OpenAI language models.




# Load the diabetes dataset
data = pd.read_csv('diabetes.csv')

# Create a deep copy of the dataset to avoid modifying the original data
diabetes_data_copy = data.copy(deep=True)

# Replace zero values in specific columns with NaN
diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
    diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# Impute missing values with mean or median in their respective columns
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)

# Standardize the feature values using StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"], axis=1)),
                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                          'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = diabetes_data_copy.Outcome

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42, stratify=y)

# Lists to store training and testing accuracy scores for different k values
test_scores = []
train_scores = []

# Iterate over different values of k for the k-nearest neighbors algorithm
for i in range(1, 15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
    test_scores.append(accuracy_score(y_test, knn.predict(X_test)))


with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',
                               
                              ['Diabetes Prediction','Diabetes Help Chatbot','BMI Calculator'],
                          
                              icons=['prescription2','robot','calculator'],
                              
                              menu_icon='hospital',
                          
                              default_index =0  
                             )



if(selected == 'Diabetes Prediction'):
    st.title('Diabetes Prediction')


    # Input Boxes for user to input feature values
    pregnancies = st.number_input("Pregnancies: Number of times pregnant", min_value=0, max_value=17, value=0)
    glucose = st.number_input("Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test", min_value=0, max_value=1000, value=0)
    blood_pressure = st.number_input("BloodPressure: Diastolic blood pressure (mm Hg)", min_value=0, max_value=300, value=0)
    skin_thickness = st.number_input("Skin Thickness: Triceps skin fold thickness (mm)", min_value=0, max_value=99, value=0)
    insulin = st.number_input("Insulin: 2-Hour serum insulin (mu U/ml)", min_value=0, max_value=846, value=79)
    bmi = st.number_input("BMI: Body mass index (weight in kg/(height in m)^2)", min_value=0, max_value=100, value=0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.5, value=0.000)
    age = st.number_input("Age (years):", min_value=21, max_value=110, value=21)

    # Make a prediction using the trained k-nearest neighbors model
    input_data = sc_X.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = knn.predict(input_data)[0]

    # Display the prediction result
    st.subheader("Prediction:")
    if prediction == 0:
        st.write("No Diabetes")
    else:
        st.write("Diabetes")
        
        

# Set OpenAI API key from the constants file
os.environ["OPENAI_API_KEY"] = openai_key

if(selected == 'Diabetes Help Chatbot'):
    # Set the title for the Streamlit web application
    st.title('Diabetes Help Bot')

    # Get user input for diabetic-related topics
    input_text = st.text_input("Ask about diabetic-related topics: Ask about diet? Blood Sugar Level? Complications?")

    # Define a template for the initial input prompt
    first_input_prompt = PromptTemplate(
        input_variables=['prompt'],
        template="reply this question {prompt} in the context of a diabetic patient."
    )

    # Set up memory for storing the conversation history
    person_memory = ConversationBufferMemory(input_key='prompt', memory_key='chat_history')

    # Create an instance of the OpenAI language model (LLM)
    llm = OpenAI(temperature=0.8)

    # Create a language model chain with the specified prompt template and memory
    chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True)

    # Check if there is user input
    if input_text:
        # Run the language model chain with the user input and display the result
        st.write(chain.run(input_text))


# BMI
import streamlit as st

def calculate_bmi(weight, height):
    """
    Calculate BMI using the formula: BMI = weight (kg) / (height (m))^2
    """
    height_in_meters = height / 100  # Convert height from centimeters to meters
    bmi = weight / (height_in_meters ** 2)
    return bmi

def interpret_bmi(bmi):
    """
    Interpret BMI and provide a basic health category.
    """
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal Weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

# Streamlit app
if(selected == 'BMI Calculator'):

    st.title("BMI Calculator")

# User input for weight and height
    weight = st.number_input("Enter your weight (kg)", min_value=0.0, step=0.1)
    height = st.number_input("Enter your height (cm)", min_value=0.0, step=0.1)

# Calculate BMI
if st.button("Calculate BMI"):
    bmi = calculate_bmi(weight, height)
    st.write(f"Your BMI is: {bmi:.2f}")
    category = interpret_bmi(bmi)
    st.write(f"Category: {category}")
