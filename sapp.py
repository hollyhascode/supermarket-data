import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


# Set page title and icon
st.set_page_config(page_title="Airline Passenger Satisfation Data Explore", page_icon="🌸")

# Sidebar for navigation
# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions!", "Extras"])

# Import data
df = pd.read_csv("cleaned_train.csv")  
test = pd.read_csv("cleaned_test.csv") 
# Get the numerical and categorical columns for visualization
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
obj_cols = df.select_dtypes(include=['object']).columns



# Home Page
if page == "Home":
    
    st.title("📊 Airline Customer Satisfaction Dataset Explorer")
    st.subheader("Welcome to our Airline Customer Satisfaction  dataset explorer app!")
    st.write("""
        This app provides an interactive platform to explore the Airline Customer Satisfaction dataset.
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!
        Use the sidebar to navigate through the sections.
    """)
    st.image('https://www.retently.com/wp-content/uploads/2018/08/Airline-satisfaction-cover-1.png', caption="Airplane Customer Satification Data Analytics")

# Data Overview Page
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("""
        The Airplane Customer Satisfaction Dataset goes over 103,904 different customer experiences and stats. Using this data we can explore factors that lead to positive and negative customer experiences.   
    """)
    st.image("https://upgradedpoints.com/wp-content/uploads/2023/03/upgradedpoints-airlinecustomerservice-graphic-v4_og_1920x1080.png?auto=webp&disable=upscale&width=1200")

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)

    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Exploratory Data Analysis (EDA) Page
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])

    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by satisfaction"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x='satisfaction', y=b_selected_col, title=chart_title, color='satisfaction'))

    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols)
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='satisfaction', title=chart_title))

    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox("Select a categorical variable:", obj_cols)
        if selected_col:
            chart_title = f'Distribution of {selected_col.title()}'
            st.plotly_chart(px.histogram(df, x=selected_col, color='satisfaction', title=chart_title))

# Extras Page
elif page == "Extras":
    st.title("Extras")
    st.subheader("Adding New Columns")
    st.write("""
        Here, you can add new columns or perform other custom transformations on the dataset.
    """)
    new_col_name = st.text_input("Enter the new column name:")
    if new_col_name:
        df[new_col_name] = 0  # Create a new column with default value 0
        st.write(f"New column `{new_col_name}` added to the dataset.")
        st.dataframe(df)

    st.title("Adding Columns")

elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

    # Make Predictions Page
elif page == "Make Predictions!":
    st.title("🌸 Make Predictions")

    st.subheader("Adjust the values below to make predictions on the Airplane Customer Satisfaction dataset:")

    # User inputs for prediction
    #satisfaction = st.slider("satisfaction", min_value=4.0, max_value=8.0, value=5.1)
    travel_type = st.slider("Type of Travel", min_value=2.0, max_value=4.5, value=3.5, step=1.0)
    #1 is personal travel, #2 is business 
    classType = st.slider("Class", min_value=0.0, max_value=2.0, value=1.0, step=1.0)
    #0 is business, 1 is eco, 2 is eco plus 
    dept_delay = st.slider("Departure Delay in Minutes ", min_value=0.1, max_value=300.0, value=0.2)
    id = st.slider("passenger number", min_value=2.0, max_value=4.5, value=3.5)
    Gender = st.slider("Gender", min_value=0.0, max_value=1.0, value=0.0, step=1.0)
    #1 is male and 0 is female 
    CustomerType = st.slider("Customer Type", min_value=0.0, max_value=1.0, value=0.0, step=1.0)
    #1 is personal travel , 0 is business travel 
    Age = st.slider("Age", min_value=0.0, max_value=100.0, value=35.0, step=1.0)
    ArrivalDelay = st.slider("Arrival Delay in Minutes", min_value=0.0, max_value=300.0, value=0.0)
    baghandle = st.slider("Baggage handling", min_value=1.0, max_value=5.0, value=1.0, step=1.0)
    clean = st.slider("Cleanliness", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
    easyleavetime = st.slider("Departure/Arrival time convenient", min_value=1.0, max_value=4.5, value=3.5, step=1.0)
    easywebbook = st.slider("Ease of Online booking", min_value=1.0, max_value=5.0, value=3.5, step=1.0)
    distance = st.slider("Flight Distance", min_value=1.0, max_value=5.0, value=3.0, step=1.0) 
    Checkin = st.slider("Checkin service", min_value=1.0, max_value=5.0, value=3.0, step=1.0) 
    foodDrink = st.slider("Food and drink", min_value=1.0, max_value=5.0, value = 3.0, step=1.0) 
    gatelocation = st.slider("Gate location", min_value=1.0, max_value=5.0, value = 3.0, step=1.0) 
    
    Inflightentertainment = st.slider("Inflight entertainment", min_value=1.0, max_value=5.0, value = 3.5, step=1.0) 
    Inflightservice = st.slider("Inflight service", min_value=1.0, max_value=5.0, value = 3.5, step=1.0) 
    Inflightwifiservice = st.slider("Inflight wifi service", min_value=1.0, max_value=5.0, value = 3.5, step=1.0) 
    Legroomservice = st.slider("Leg room service", min_value=1.0, max_value=5.0, value = 3.5, step=1.0) 
    Onboardservice = st.slider("On-board service", min_value=1.0, max_value=5.0, value = 3.5, step=1.0)
    Onlineboarding = st.slider("Online boarding", min_value=1.0, max_value=5.0, value = 3.5, step=1.0)
    Seatcomfort = st.slider("Seat comfort", min_value=1.0, max_value=5.0, value = 3.5, step=1.0)
    id = st.slider("id", min_value=2.0, max_value=4.5, value = 3.5)
      
    # User input dataframe
    user_input = pd.DataFrame({
       
        "id": [id],
        'Gender': [Gender], 
        'Customer Type': [CustomerType], 
        'Age': [Age], 
        'Type of Travel': [travel_type],
        'Class': [classType],
        "Flight Distance": [distance], 
        "Inflight wifi service": Inflightwifiservice,
        "Departure/Arrival time convenient": [easyleavetime],  
        "Ease of Online booking": [easywebbook],
        "Gate location": [gatelocation],
        "Food and drink": [foodDrink],
        "Online boarding": [Onlineboarding],  
        "Seat comfort": [Seatcomfort], 
        "Inflight entertainment": [Inflightentertainment],
        "On-board service": [Onboardservice], 
        "Leg room service": [Legroomservice], 
        'Baggage handling': [baghandle], 
        "Checkin service": [Checkin],
        "Inflight service": [Inflightservice],  
        "Cleanliness": [clean], 
        'Departure Delay in Minutes': [dept_delay],
        'Arrival Delay in Minutes': [ArrivalDelay],   
        #'satisfaction': [satisfaction]
        
       
        #'Customer Type': [CustomerType],  
         
       
       
       
        
        
       
   
      
        
  
        
      
        
       
        
        
       
       

    })

    st.write("### Your Input Values")
    st.dataframe(user_input)

    # Use KNN (k=9) as the model for predictions
    model = KNeighborsClassifier(n_neighbors=9)
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction = model.predict(user_input_scaled)[0]


    # Display the result
    st.write(f"The model predicts that the passenger is of the Satisfaction (0 is neutral or unsatisfied, 1 is satisfied) : **{prediction}**")
    st.balloons()
