import base64
import streamlit as st
import pandas as pd
import joblib
import dill
from sklearn.pipeline import Pipeline

# Load the pretrained model
with open("pipeline.pkl") as file:
    model = dill.load(file)

# Load the feature dictionary
with open("my_feature_dict.pkl", "rb") as file:
    my_feature_dict = joblib.load(file)

# Function to predict churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction[0]

# Streamlit page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation Buttons
pages = {"Home": "home", "Churn Prediction": "input", "Model Info": "info", "About Us": "about"}
selected_page = st.sidebar.radio("", list(pages.keys()), key="nav")

# CSS for Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpapers.com/images/hd/dark-gray-background-31zgslm940epcocw.jpg"); /* Replace with your image URL */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white; /* Ensures text is visible */
    }

    /* Targeting the title bar */
    .css-1e5im15 {
        background-image: url("https://wallpapers.com/images/hd/dark-gray-background-31zgslm940epcocw.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Optional: Ensure text inside title bar remains visible */
    .css-1e5im15 h1, .css-1e5im15 h2 {
        color: #ff9700;
    }
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: black;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: white;
        font-size: 18px;
        margin-bottom: 10px;
        padding: 8px 12px;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #ff9700;
        color: white;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > div {
        margin-bottom: 8px;
    }

    /* Custom styles for prediction button */
    div.stButton > button:first-child {
        background-color: #ff9700;
        color: white;
        font-size: 20px;
        height: 2em;
        width: 20em;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff9700;
    }

    /* Custom styles for alert box */
    .alert-box {
        background-color: rgba(255, 0, 0, 0.8); /* Red background with transparency */
        color: white; /* Text color */
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Home Page
if selected_page == "Home":
    st.markdown(
        """
        <style>
        .main-header {
            font-size:40px;
            text-align:center;
            font-weight: bold;
            color:#ff9700;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<p class="main-header">WELCOME TO CUSTOMER CHURN PREDICTION APP</p>', unsafe_allow_html=True)
    st.markdown(
        """
        This app is designed to help businesses predict customer churn rates. Using advanced machine learning algorithms, 
        this app processes input data about customers' demographics, behaviors, and engagement patterns to predict whether 
        a customer is likely to stay or leave the service.  
        
        **Who is this app for?**
        - Business Analysts
        - Marketing Teams
        - Data Scientists
        - Anyone looking to improve customer retention strategies
        
        With real-time predictions and detailed input analysis, this app provides insights to make informed decisions, 
        reduce churn, and boost customer satisfaction.
        """,
        unsafe_allow_html=True
    )

# Churn Prediction Page
elif selected_page == "Churn Prediction":
    st.markdown("### Churn Prediction 📝")
    st.markdown("#### Categorical Features 📊")
    categorical_input = my_feature_dict.get('CATEGORICAL')
    categorical_input_vals = {}
    for i, col in enumerate(categorical_input.get('Column Name').values()):
        options = categorical_input.get('Members')[i]
        categorical_input_vals[col] = st.selectbox(
            col,
            options,
            help=f"Choose the most relevant option for {col}"
        )
    
    st.markdown("#### Numerical Features 🔢")
    numerical_input = my_feature_dict.get('NUMERICAL')
    numerical_input_vals = {}
    for col in numerical_input.get('Column Name'):
        if col == 'AGE':
            min_value = 20
            max_value = 100
        elif col == 'EXPERIENCEINCURRENTDOMAIN':
            min_value = 0
            max_value = 20
        else:
            min_value = 2010
            max_value = 2030
        numerical_input_vals[col] = st.slider(
            f"Adjust {col}:",
            min_value,
            max_value,
            min_value,
            help=f"Slide to set the value for {col}"
        )

    # Combine Input Data
    input_data = {**categorical_input_vals, **numerical_input_vals}
    input_data = pd.DataFrame([input_data])

    # Display Input Data
    with st.expander("See Input Data 📋", expanded=False):
        st.write("Preview the input data before making predictions:")
        st.write(input_data)

    # Prediction Button
    if st.button('Predict Churn 🚀'):
        with st.spinner("Predicting..."):
            prediction = predict_churn(input_data)
            translation_dict = {1: 'Tier 1', 2: 'Tier 2', 3: 'Tier 3'}
            prediction_translate = translation_dict.get(prediction, "Unknown")
            
            if prediction == 'LEAVE':  # Employee likely to leave
                st.markdown(
                    '<div class="alert-box">🔴 ALERT: The employee is likely to LEAVE!</div>',
                    unsafe_allow_html=True
                )
            else:  # Employee likely to stay
                st.markdown(
                    '<div class="alert-box" style="background-color: rgba(0, 128, 0, 0.8);">🟢 GREAT NEWS: The employee is likely to STAY!</div>',
                    unsafe_allow_html=True
                )
                st.balloons()


# Model Info Section
elif selected_page == "Model Info":
    st.markdown("### Model Information 📚")
    st.write("This app uses a pretrained machine learning pipeline for predicting customer churn.")
    st.write("Details:")
    st.write("- Model Type: Machine Learning Pipeline")
    st.write("- Libraries: Scikit-learn, Dill, Joblib")
    st.write("- Input Features: Categorical and Numerical Variables")
    st.write("- Output: Tier Predictions (e.g., Tier 1, Tier 2, etc.)")

# About Page
elif selected_page == "About Us":
    import base64

    # Function to encode the image to base64
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Encode your profile picture
    profile_pic_base64 = encode_image_to_base64("C:/Users/Mehdi Abbas/Downloads/churnpredict/My profile picture professional.jpeg")

    st.markdown("### About Us 🛠️")
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/jpeg;base64,{profile_pic_base64}" 
                 alt="Creator's Photo" 
                 style="width:150px; height:150px; margin-right: 20px; border-radius: 50%;">
            <div>
                <p>Hello! My name is <b>Syed Mehdi Abbas Rizvi</b>, the creator of this app. I specialize in Data Analytics, 
                Machine Learning, and Power BI Expert. With a passion for solving real-world problems through innovative solutions, 
                I designed this app to help businesses predict and understand customer churn.</p>
                <p>Feel free to reach out for collaborations, feedback, or support!</p>
                <p>Email: <a href="mailto:mehdiabbas2017@gmail.com">mehdiabbas2017@gmail.com</a></p>
		<p>Contact: +92 331 3469106 </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
