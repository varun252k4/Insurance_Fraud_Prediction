import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to load data
@st.cache_data
def load_data():
    file_path = './InsuranceFraud.xlsx'
    if os.path.exists(file_path):
        data = pd.read_excel(file_path)
        return data
    else:
        st.error("File not found. Please check the file path.")
        return None

def encode_categorical_features(df):
    if df is not None:
        encoder_dict = {}
        categorical_columns = df.select_dtypes(include=["object"]).columns
        encoder = LabelEncoder()
        for col in categorical_columns:
        
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoder_dict[col] = encoder.classes_  # Store the classes for each column

        return df, encoder_dict
    else:
        st.error("DataFrame is None, cannot encode categorical features.")
        return None, None


# Function to preprocess data
def data_preprocessor(df):
    if df is not None:
        df = df.copy()
        df.dropna(inplace=True)

         # Replace '?' with NaN for easier handling
        df.replace('?', np.nan, inplace=True)

        # Handle missing values by filling with the mode of each column
        for col in df.select_dtypes(include=["object"]).columns:
            if col in df.columns:
                mode_value = df[col].mode()[0]  # Get the most frequent value
                df[col].fillna(mode_value, inplace=True)
        
        columns_to_drop = [
            'policy_number', 'policy_bind_date', 'policy_csl', 'insured_zip', 'incident_date', 
            'auto_model', 'auto_year', 'injury_claim', 'property_claim', 'vehicle_claim', 
            'bodily_injuries', 'incident_hour_of_the_day', 'incident_location', 
            'capital_gains', 'capital_loss', 'policy_deductable', 'number_of_vehicles_involved', 'capital-gains','capital-loss'
        ]
        df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
        
        df = df[~df['collision_type'].isin(['?'])]
        df = df[~df['property_damage'].isin(['?'])]
        df = df[~df['police_report_available'].isin(['?'])]
        
        # Convert specified columns to integers
        integer_columns = ['months_as_customer', 'age', 'witnesses']
        for col in integer_columns:
            df[col] = df[col].astype(int)
    
        return df
    else:
        st.error("DataFrame is None, cannot preprocess data.")
        return None, None

# Categorical field options
categorical_options = {
    'policy_state': ['OH', 'IL', 'IN'],
    'insured_sex': ['MALE', 'FEMALE'],
    'insured_education_level': ['MD', 'PhD', 'High School', 'College', 'Masters', 'JD', 'Associate'],
    'insured_occupation': ['craft-repair', 'sales', 'tech-support', 'other-service', 'exec-managerial', 'protective-serv', 'machine-op-inspct', 'transport-moving', 'prof-specialty', 'adm-clerical', 'handlers-cleaners', 'armed-forces', 'farming-fishing', 'priv-house-serv'],
    'insured_hobbies': ['sleeping', 'board-games', 'bungie-jumping', 'golf', 'skydiving', 'reading', 'movies', 'yachting', 'paintball', 'kayaking', 'polo', 'basketball', 'hiking', 'video-games', 'chess', 'cross-fit', 'exercise', 'dancing', 'camping', 'base-jumping'],
    'insured_relationship': ['husband', 'own-child', 'unmarried', 'other-relative', 'wife', 'not-in-family'],
    'incident_type': ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Parked Car', 'Vehicle Theft'],
    'collision_type': ['Side Collision', 'Rear Collision', 'Front Collision'],
    'incident_severity': ['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage'],
    'authorities_contacted': ['Police', 'Fire', 'Ambulance', 'Other'],
    'incident_state': ['SC', 'NY', 'WV', 'VA', 'OH', 'PA', 'NC'],
    'incident_city': ['Columbus', 'Arlington', 'Springfield', 'Northbend', 'Hillsdale', 'Northbrook', 'Riverwood'],
    'property_damage': ['YES', 'NO'],
    'police_report_available': ['YES', 'NO'],
    'auto_make': ['Saab', 'Dodge', 'Toyota', 'Audi', 'Accura', 'Suburu', 'Ford', 'BMW', 'Mercedes', 'Chevrolet', 'Honda', 'Nissan', 'Volkswagen', 'Jeep']
}

# Streamlit app layout
st.title("Insurance Fraud Detection")

# Load and process data
data = load_data()

if data is not None:
    processed_data  = data_preprocessor(data)  # Get encoder_dict here

    if processed_data is not None:
        # Display the preprocessed DataFrame
        st.subheader("Preprocessed Data (Before Encoding)")
        st.dataframe(processed_data)

        # Now encode categorical features
        processed_data, encoder_dict = encode_categorical_features(processed_data)
        # Define features and target
        features = processed_data.columns.drop('fraud_reported')
        X = processed_data[features]
        y = processed_data['fraud_reported']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        dt_model = DecisionTreeClassifier(max_depth=10,random_state=42)
        rf_model = RandomForestClassifier(n_estimators=35,max_depth=10,random_state=42)
        nb_model = GaussianNB()

        models = {
            "Decision Tree": dt_model,
            "Random Forest": rf_model,
            "Naive Bayes": nb_model
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
        
        mod = rf_model

        

        # Sidebar for navigation
        tab = st.sidebar.selectbox("Select Tab", ["Model Comparison", "Prediction"])

        if tab == "Model Comparison":

            st.header("Model Comparison")
            
            results = []
            
            for name, model in models.items():
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, pos_label=1)
                recall = recall_score(y_test, y_pred, pos_label=1)
                f1 = f1_score(y_test, y_pred, pos_label=1)
                results.append([name, accuracy, precision, recall, f1])

            results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
            st.write(results_df)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df.set_index("Model").plot(kind="bar", ax=ax)
            plt.title("Model Performance Comparison")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Confusion Matrix for best model
            best_model_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
            best_model = models[best_model_name]
            y_pred = best_model.predict(X_test)
            
            st.write(f"Confusion Matrix for {best_model_name}:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Not Fraud', 'Fraud'],
                        yticklabels=['Not Fraud', 'Fraud'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

        elif tab == "Prediction":
            st.header("Make Predictions")

            # Create input fields
            input_data = {}
   
            for feature in features:
                if feature in categorical_options:
                    # Determine the number of options
                    options = categorical_options[feature]
                    if len(options) <= 4:
                        # Use radio buttons for 4 or fewer options
                        input_data[feature] = st.radio(f"Select {feature.replace('_', ' ').title()}", options)
                    else:
                        # Use select box for more than 4 options
                        input_data[feature] = st.selectbox(f"Select {feature.replace('_', ' ').title()}", options)
                elif feature in ['months_as_customer', 'age', 'witnesses']:
                    # Use sliders for these numerical features
                    min_value = int(X[feature].min())
                    max_value = int(X[feature].max())
                    default_value = int(X[feature].mean())
                    input_data[feature] = st.slider(f"Enter {feature.replace('_', ' ').title()}", min_value=min_value, max_value=max_value, value=default_value, step=1)
                else:
                    # Keep number input for other features
                    default_value = X[feature].mean()
                    input_data[feature] = st.number_input(f"Enter {feature.replace('_', ' ').title()}", value=float(default_value))

            if st.button("Predict"):
                # Create input DataFrame
                input_df = pd.DataFrame([input_data])

                # Transform categorical fields
                for col in input_df.columns:
                    if col in encoder_dict:
                        if input_df[col].iloc[0] not in encoder_dict[col]:
                            st.error(f"Invalid value for {col}: {input_df[col].iloc[0]}")
                        else:
                            encoder = LabelEncoder()  # Create a new encoder instance to avoid issues
                            encoder.fit(encoder_dict[col])  # Fit on the learned classes
                            input_df[col] = encoder.transform([input_df[col].iloc[0]])


                # Ensure that the input DataFrame is consistent with the training data
                input_df = input_df[features]  # Reorder and filter to match training features

                # Make predictions
                
                prediction = mod.predict(input_df)
                probability = mod.predict_proba(input_df)[0][1]  # Probability of fraud
                st.write(f"Random Forest Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
                st.write(f"Random Forest Fraud Probability: {probability:.2f}")

    else:
        st.error("Error processing data.")
else:
    st.error("Error loading data.")